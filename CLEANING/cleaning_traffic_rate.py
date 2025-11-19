# CLEANING/cleaning_traffic_rate.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import os
import sys
import pickle
import numpy as np
import pandas as pd

class CleaningTrafficRate:
    """
    Grupo 4: Ritmo de Tráfico — Preprocesamiento mínimo + PCA

    Pasos (exactamente como en el texto):
      1) Coerción numérica: tasas/intervalos/períodos -> float; no finitos -> NaN.
      2) Imputación conservadora por mediana (aprendida en entrenamiento).
      3) Transformación log (log1p) para variables NO negativas y fuertemente asimétricas.
      4) Estandarización z-score (media y std del entrenamiento).
      5) PCA sobre TODO el subespacio de ritmo ya transformado:
           - Selección de componentes: mínimo k tal que varianza acumulada ≥ 0.80,
             con tope de 4 (3–4 típicamente).
      6) Reemplazo: se dropean las columnas originales del bloque ritmo y se agregan
         traffic_pca_1..k (+ traffic_pca_var_expl_total).
      7) Artefactos persistidos en CLEANING/.artifacts/traffic_rate_prep.pkl

    Modo de trabajo:
      - Si existen artefactos -> aplica (inferencia).
      - Si no existen   -> ajusta y guarda (entrenamiento).

    Notas:
      - No mezcla información de clase en el ajuste de PCA/escala más que para
        definir columnas; las estadísticas se calculan sobre TODO el set que se use
        como entrenamiento en esta etapa del pipeline.
    """

    name: str = "04_traffic_rate"
    depends_on: list[str] = ["02_flow_intensity"]

    # Dónde guardamos artefactos (relativo a esta carpeta CLEANING)
    ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), ".artifacts")
    ARTIFACTS_FILE = os.path.join(ARTIFACTS_DIR, "traffic_rate_prep.pkl")

    # Heurísticas de selección de columnas del bloque "ritmo"
    # (se detectan por nombres típicos; si tu dataset usa otros, ajustá aquí)
    EXPLICIT_COLS = [
        "flow_bytes_s", "flow_pkts_s",
        "active.tot", "idle.tot",
        "active.mean", "idle.mean", "active.max", "idle.max", "active.min", "idle.min",
        "flow_iat.avg", "flow_iat.min", "flow_iat.max", "flow_iat.std", "flow_iat.tot"
    ]
    # Además, incluimos cualquier columna cuyo nombre contenga estos patrones:
    CONTAINS_PATTERNS = ["iat", "active", "idle"]

    SKEW_THRESHOLD = 1.0   # "fuerte asimetría" ~ skewness > 1
    VAR_EXPL_TARGET = 0.80 # 80% de varianza acumulada
    PCA_MAX_COMPONENTS = 4
    KEEP_ORIGINAL = False  # Reemplazamos por las componentes

    # ---------------- Utilidades internas ----------------
    def _collect_rate_cols(self, df: pd.DataFrame) -> List[str]:
        cols = [c for c in self.EXPLICIT_COLS if c in df.columns]
        for c in df.columns:
            name = str(c).lower()
            if any(p in name for p in self.CONTAINS_PATTERNS):
                if c not in cols and pd.api.types.is_numeric_dtype(df[c]):
                    cols.append(c)
        # Quitar duplicados conservando orden
        seen = set(); ordered = []
        for c in cols:
            if c not in seen:
                ordered.append(c); seen.add(c)
        return ordered

    def _to_numeric(self, df: pd.DataFrame, cols: List[str]) -> None:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def _fit_artifacts(self, df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
        """Aprende medianas, qué columnas log-transformar, medias/std y PCA."""
        from sklearn.decomposition import PCA

        # 1) Medianas para imputación
        medians = {c: float(df[c].median(skipna=True)) for c in cols}

        # 2) Imputar para poder evaluar skew/log y estandarizar
        X = df[cols].copy()
        for c in cols:
            X[c] = X[c].fillna(medians[c])

        # 3) Selección de columnas para log1p:
        #    - no negativas (min >= 0 luego de imputar)
        #    - asimetría fuerte (skew > SKEW_THRESHOLD)
        log_cols: List[str] = []
        for c in cols:
            series = X[c]
            if series.min() >= 0:
                # skew robusto: Fisher-Pearson (pandas .skew usa momento de orden 3)
                skew = float(series.skew(skipna=True))
                if np.isfinite(skew) and skew > self.SKEW_THRESHOLD:
                    log_cols.append(c)

        # aplicar log1p en copia X
        for c in log_cols:
            X[c] = np.log1p(X[c])

        # 4) z-score (media y std)
        means = {c: float(X[c].mean(skipna=True)) for c in cols}
        stds  = {c: float(X[c].std(ddof=0)) for c in cols}
        Z = pd.DataFrame(index=X.index)
        for c in cols:
            std = stds[c]
            Z[c] = (X[c] - means[c]) / std if std and np.isfinite(std) and std > 0 else 0.0

        # 5) PCA: elegir k mínimo con var.expl. acumulada >= 80%, tope 4
        pca = PCA(n_components=min(len(cols), self.PCA_MAX_COMPONENTS))
        pca.fit(Z[cols].values)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum, self.VAR_EXPL_TARGET) + 1)
        k = max(1, min(k, self.PCA_MAX_COMPONENTS, len(cols)))

        # Volver a ajustar con k definitivo
        pca = PCA(n_components=k)
        pca.fit(Z[cols].values)

        artifacts = {
            "cols": cols,
            "medians": medians,
            "log_cols": log_cols,
            "means": means,
            "stds": stds,
            "pca": pca,
            "var_expl_ratio": [float(x) for x in pca.explained_variance_ratio_],
        }
        return artifacts

    def _apply_artifacts(self, df: pd.DataFrame, art: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Aplica imputación por mediana, log1p y z-score con artefactos; luego PCA."""
        cols = art["cols"]
        medians = art["medians"]
        log_cols = set(art["log_cols"])
        means = art["means"]
        stds = art["stds"]
        pca = art["pca"]

        # Asegurar columnas presentes (si falta alguna, crearla como NaN)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        X = df[cols].copy()
        # Imputar medianas
        for c in cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians[c])

        # log1p solo en columnas definidas
        for c in cols:
            if c in log_cols:
                X[c] = np.log1p(X[c])

        # z-score
        Z = pd.DataFrame(index=X.index)
        for c in cols:
            std = stds[c]
            Z[c] = (X[c] - means[c]) / std if std and np.isfinite(std) and std > 0 else 0.0

        # PCA transform
        comps = pca.transform(Z[cols].values)
        k = pca.n_components_
        for i in range(k):
            df[f"traffic_pca_{i+1}"] = comps[:, i]
        df["traffic_pca_var_expl_total"] = float(np.sum(pca.explained_variance_ratio_))

        return df, {
            "n_features_in": len(cols),
            "n_components": k,
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        }

    # ---------------- API Stage ----------------
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = df.copy()
        before_rows = len(df)

        report: Dict[str, Any] = {
            "rows_before": before_rows,
            "rows_after": None,
            "dropped": 0,
            "rate_cols_used": [],
            "log_cols": [],
            "imputed_with_median": True,
            "standardized": True,
            "pca": {"n_components": 0, "explained_variance_ratio": []},
            "artifacts_mode": "apply" if os.path.exists(self.ARTIFACTS_FILE) else "fit",
            "notes": [],
        }

        # 0) Selección de columnas del bloque ritmo
        rate_cols = self._collect_rate_cols(df)
        # Forzar numérico en las candidatas (no finitos -> NaN)
        self._to_numeric(df, rate_cols)

        if not rate_cols:
            report["notes"].append("No se detectaron columnas de ritmo; módulo 4 omitido.")
            report["rows_after"] = len(df)
            return df, report

        report["rate_cols_used"] = list(rate_cols)

        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)

        # 1) Fit o Apply
        if not os.path.exists(self.ARTIFACTS_FILE):
            # ---- ENTRENAMIENTO: aprender artefactos y guardarlos ----
            try:
                artifacts = self._fit_artifacts(df, rate_cols)
            except Exception as e:
                report["notes"].append(f"Falló _fit_artifacts: {e}")
                report["rows_after"] = len(df)
                return df, report

            # Guardar artefactos
            with open(self.ARTIFACTS_FILE, "wb") as f:
                pickle.dump(artifacts, f)

            # Aplicar inmediatamente lo aprendido
            df, pca_info = self._apply_artifacts(df, artifacts)
            report["pca"] = pca_info
            report["log_cols"] = artifacts["log_cols"]
            report["notes"].append(f"Artefactos entrenados y guardados en {os.path.relpath(self.ARTIFACTS_FILE, os.path.dirname(__file__))}.")
        else:
            # ---- INFERENCIA: cargar artefactos y aplicar ----
            with open(self.ARTIFACTS_FILE, "rb") as f:
                artifacts = pickle.load(f)
            # Si las columnas actuales difieren, extendemos df con NaN y seguimos
            missing = [c for c in artifacts["cols"] if c not in df.columns]
            if missing:
                report["notes"].append(f"Se agregaron columnas faltantes para aplicar PCA: {missing}")
            df, pca_info = self._apply_artifacts(df, artifacts)
            report["pca"] = pca_info
            report["log_cols"] = artifacts["log_cols"]

        # 2) Drop de columnas originales del bloque ritmo (reemplazo por PCA)
        if not self.KEEP_ORIGINAL:
            df = df.drop(columns=[c for c in rate_cols if c in df.columns], errors="ignore")
            report["notes"].append("Columnas originales de ritmo reemplazadas por traffic_pca_*.")

        # 3) Final
        after_rows = len(df)
        report["rows_after"] = after_rows
        report["dropped"] = before_rows - after_rows  # este módulo no dropea filas
        report["notes"].extend([
            "Coerción numérica + NaN en no-finitos.",
            "Imputación por mediana (entrenamiento) y reuso en inferencia.",
            "Log1p en columnas no negativas y sesgadas (skew>1).",
            "Z-score con medias/std del entrenamiento.",
            "PCA aplicado; #comp. elegido por ≥80% varianza (máx 4).",
        ])
        return df, report

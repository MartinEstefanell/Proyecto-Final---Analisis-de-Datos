# CLEANING/cleaning_flow_intensity.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import os
import pickle
import numpy as np
import pandas as pd

class CleaningFlowIntensity:
    """
    Grupo 2: Intensidad y Tamaño de Flujo — Limpieza + PCA (genera nuevas variables)
    - Elimina filas físicamente inválidas (duración/volumen/tasa <= 0).
    - Verifica coherencias (3σ y ±15%) y elimina inconsistentes.
    - Elimina columnas redundantes: fwd_pkts_tot, bwd_pkts_tot, flow_pkts_per_sec.
    - Outliers SOLO en NORMAL: recorte P99 + imputación mediana.
    - Prep. PCA: imputación mediana (entreno), log1p en asimétricas, z-score (media/std entreno).
    - PCA: k mínimo con var.exp. acumulada ≥ 85% (máx 4) → intensity_pca_1..k (+ var total).
    - Reemplaza columnas del bloque por PCs y (opcional) conserva representativas fuera del PCA.

    Artefactos:
      CLEANING/.artifacts/flow_intensity_prep.pkl

    Variables representativas (fuera del PCA, opcional):
      - flow_pkts_payload.tot (magnitud)
      - payload_bytes_per_second (ritmo)
      - down_up_ratio (direccional) si existe
    """

    name: str = "02_flow_intensity"
    depends_on: list[str] = ["01_comm_type"]

    ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), ".artifacts")
    ARTIFACTS_FILE = os.path.join(ARTIFACTS_DIR, "flow_intensity_prep.pkl")

    # Selección de columnas candidatas del bloque intensidad
    INTENSITY_COLS = [
        "flow_duration",
        "flow_pkts_payload.tot",
        "payload_bytes_per_second",
        "flow_pkts_per_sec",           # redundante (se eliminará tras coherencias)
        "fwd_pkts_tot", "bwd_pkts_tot",# redundantes (se eliminarán)
        "flow_iat.avg", "flow_iat.std",
        "active.avg", "idle.avg",
        "active.std", "idle.std",
        # si existen otras métricas de actividad/iat, se pueden sumar aquí
    ]

    VAR_EXPL_TARGET = 0.85
    PCA_MAX_COMPONENTS = 4
    SKEW_THRESHOLD = 1.0
    KEEP_REPRESENTATIVES = True   # conservar fuera del PCA: magnitud/ritmo/direccional

    # ---------------- utilidades ----------------
    def _to_numeric(self, df: pd.DataFrame, cols: List[str]) -> None:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    def _attack_col(self, df: pd.DataFrame) -> str:
        if "attack_type" in df.columns: return "attack_type"
        if "Attack_type" in df.columns: return "Attack_type"
        df["attack_type"] = "unknown"; return "attack_type"

    # ------------- reglas de consistencia -------------
    def _drop_physical_invalid(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        must_pos = ["flow_duration", "flow_pkts_payload.tot", "payload_bytes_per_second"]
        present = [c for c in must_pos if c in df.columns]
        if not present:
            return df, 0
        mask = pd.Series(False, index=df.index)
        for c in present:
            mask |= (pd.to_numeric(df[c], errors="coerce").fillna(0) <= 0)
        dropped = int(mask.sum())
        return df.loc[~mask].copy(), dropped

    def _drop_3sigma_duration_consistency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        needed = ["flow_iat.avg", "flow_pkts_payload.tot", "flow_duration"]
        if not all(c in df.columns for c in needed):
            return df, 0
        iat = pd.to_numeric(df["flow_iat.avg"], errors="coerce")
        pkts = pd.to_numeric(df["flow_pkts_payload.tot"], errors="coerce")
        dur  = pd.to_numeric(df["flow_duration"], errors="coerce")
        dif = iat * pkts - dur
        std = np.nanstd(dif.values, ddof=0)
        thr = 3 * std if np.isfinite(std) and std > 0 else np.inf
        bad = dif.abs() > thr
        dropped = int(bad.sum())
        return df.loc[~bad].copy(), dropped

    def _drop_ratio_15pct(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """±15%: payload_bytes_per_second ≈ flow_pkts_payload.tot / flow_duration
                 flow_pkts_per_sec     ≈ (fwd_pkts_tot + bwd_pkts_tot) / flow_duration (si existe)"""
        out = {"dropped_ratio_pbs": 0, "dropped_ratio_pkts_rate": 0}
        if all(c in df.columns for c in ["payload_bytes_per_second","flow_pkts_payload.tot","flow_duration"]):
            exp = (pd.to_numeric(df["flow_pkts_payload.tot"], errors="coerce") /
                   pd.to_numeric(df["flow_duration"], errors="coerce"))
            rel = (pd.to_numeric(df["payload_bytes_per_second"], errors="coerce") - exp).abs() / exp.replace(0, np.nan).abs()
            bad = (rel > 0.15).fillna(False)
            out["dropped_ratio_pbs"] = int(bad.sum())
            df = df.loc[~bad].copy()
        if all(c in df.columns for c in ["flow_pkts_per_sec","fwd_pkts_tot","bwd_pkts_tot","flow_duration"]):
            exp = ((pd.to_numeric(df["fwd_pkts_tot"], errors="coerce").fillna(0) +
                    pd.to_numeric(df["bwd_pkts_tot"], errors="coerce").fillna(0)) /
                   pd.to_numeric(df["flow_duration"], errors="coerce"))
            rel = (pd.to_numeric(df["flow_pkts_per_sec"], errors="coerce") - exp).abs() / exp.replace(0, np.nan).abs()
            bad = (rel > 0.15).fillna(False)
            out["dropped_ratio_pkts_rate"] = int(bad.sum())
            df = df.loc[~bad].copy()
        return df, out

    # ------------- outliers (solo NORMAL) -------------
    def _cap_p99_normals_and_impute(self, df: pd.DataFrame, cols: List[str], is_normal: pd.Series) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for c in cols:
            if c not in df.columns: 
                continue
            vals = pd.to_numeric(df.loc[is_normal, c], errors="coerce")
            if vals.notna().sum() == 0:
                info[c] = {"p99": np.nan, "n_capped": 0}
                continue
            p99 = float(np.nanpercentile(vals.values, 99))
            over = is_normal & (pd.to_numeric(df[c], errors="coerce") > p99)
            ncap = int(over.sum())
            df.loc[over, c] = p99
            # “imputación mediana” posterior (por si quedan NaN): guardamos la mediana
            med = float(pd.to_numeric(df.loc[is_normal, c], errors="coerce").median())
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med)
            info[c] = {"p99": p99, "n_capped": ncap, "median_used": med}
        return info

    # ------------- artefactos: fit/apply (para PCA) -------------
    def _fit_artifacts(self, df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
        from sklearn.decomposition import PCA
        medians = {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in cols}

        X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").fillna(medians[c]) for c in cols})

        # decidir log1p en variables no negativas y sesgadas
        log_cols: List[str] = []
        for c in cols:
            series = X[c]
            if series.min() >= 0:
                sk = float(series.skew())
                if np.isfinite(sk) and sk > self.SKEW_THRESHOLD:
                    log_cols.append(c)
        for c in log_cols:
            X[c] = np.log1p(X[c])

        means = {c: float(X[c].mean()) for c in cols}
        stds  = {c: float(X[c].std(ddof=0)) for c in cols}
        Z = pd.DataFrame({c: ((X[c]-means[c]) / stds[c] if stds[c] and np.isfinite(stds[c]) and stds[c] > 0 else 0.0)
                          for c in cols})

        # PCA preliminar para elegir k
        pca = PCA(n_components=min(len(cols), self.PCA_MAX_COMPONENTS))
        pca.fit(Z[cols].values)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum, self.VAR_EXPL_TARGET) + 1)
        k = max(1, min(k, self.PCA_MAX_COMPONENTS, len(cols)))

        pca = PCA(n_components=k)
        pca.fit(Z[cols].values)

        return {
            "cols": cols,
            "medians": medians,
            "log_cols": log_cols,
            "means": means,
            "stds": stds,
            "pca": pca,
            "var_expl_ratio": [float(x) for x in pca.explained_variance_ratio_],
        }

    def _apply_artifacts(self, df: pd.DataFrame, art: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        cols     = art["cols"]
        medians  = art["medians"]
        log_cols = set(art["log_cols"])
        means    = art["means"]
        stds     = art["stds"]
        pca      = art["pca"]

        # asegurar columnas
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").fillna(medians[c]) for c in cols})
        for c in cols:
            if c in log_cols:
                X[c] = np.log1p(X[c])
        Z = pd.DataFrame({c: ((X[c]-means[c]) / stds[c] if stds[c] and np.isfinite(stds[c]) and stds[c] > 0 else 0.0)
                          for c in cols})

        comps = pca.transform(Z[cols].values)
        k = pca.n_components_
        for i in range(k):
            df[f"intensity_pca_{i+1}"] = comps[:, i]
        df["intensity_pca_var_expl_total"] = float(np.sum(pca.explained_variance_ratio_))
        return df, {"n_features_in": len(cols), "n_components": k,
                    "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_]}

    # ------------- stage API -------------
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = df.copy()
        before = len(df)
        report: Dict[str, Any] = {
            "rows_before": before,
            "rows_after": None,
            "dropped": 0,
            "dropped_physical_invalid": 0,
            "dropped_3sigma": 0,
            "dropped_ratio_checks": {},
            "removed_redundant_columns": [],
            "outliers_normals_cap_p99": {},
            "artifacts_mode": "apply" if os.path.exists(self.ARTIFACTS_FILE) else "fit",
            "intensity_cols_used": [],
            "pca": {"n_components": 0, "explained_variance_ratio": []},
            "notes": [],
        }

        # 0) coerción numérica base
        self._to_numeric(df, [c for c in self.INTENSITY_COLS if c in df.columns])

        # 1) eliminar filas físicamente inválidas
        df, n_drop = self._drop_physical_invalid(df)
        report["dropped_physical_invalid"] = n_drop

        # 2) coherencias
        df, n3 = self._drop_3sigma_duration_consistency(df)
        report["dropped_3sigma"] = n3

        df, rch = self._drop_ratio_15pct(df)
        report["dropped_ratio_checks"] = rch

        # 3) redundancias -> eliminar columnas
        to_remove = [c for c in ["fwd_pkts_tot", "bwd_pkts_tot", "flow_pkts_per_sec"] if c in df.columns]
        if to_remove:
            df = df.drop(columns=to_remove)
            report["removed_redundant_columns"] = to_remove

        # 4) preparar columnas para PCA (sin las eliminadas)
        intensity_cols = [c for c in self.INTENSITY_COLS
                          if c in df.columns and c not in ["fwd_pkts_tot","bwd_pkts_tot","flow_pkts_per_sec"]]
        # quitar flow_duration de PCA (es denominador de razones y ya verificada física)
        if "flow_duration" in intensity_cols:
            intensity_cols.remove("flow_duration")

        if not intensity_cols:
            report["notes"].append("No hay columnas de intensidad suficientes para PCA; módulo 2 se limita a limpieza.")
            report["rows_after"] = len(df); report["dropped"] = before - len(df)
            return df, report

        report["intensity_cols_used"] = list(intensity_cols)

        # 5) Outliers SOLO en NORMAL (cap P99 + imputación mediana)
        attack_col = self._attack_col(df)
        is_normal = df[attack_col].astype(str).str.lower().eq("normal")
        cap_targets = [c for c in ["payload_bytes_per_second", "flow_pkts_payload.tot"] if c in df.columns]
        report["outliers_normals_cap_p99"] = self._cap_p99_normals_and_impute(df, cap_targets, is_normal)

        # 6) Fit/Apply artefactos para PCA
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        if not os.path.exists(self.ARTIFACTS_FILE):
            # ENTRENAMIENTO
            artifacts = self._fit_artifacts(df, intensity_cols)
            with open(self.ARTIFACTS_FILE, "wb") as f:
                pickle.dump(artifacts, f)
            df, pinfo = self._apply_artifacts(df, artifacts)
            report["pca"] = pinfo
            report["notes"].append("Artefactos de Flow Intensity entrenados y guardados.")
        else:
            # INFERENCIA
            with open(self.ARTIFACTS_FILE, "rb") as f:
                artifacts = pickle.load(f)
            missing = [c for c in artifacts["cols"] if c not in df.columns]
            if missing:
                report["notes"].append(f"Se agregaron columnas faltantes para aplicar PCA: {missing}")
                for c in missing: df[c] = np.nan
            df, pinfo = self._apply_artifacts(df, artifacts)
            report["pca"] = pinfo

        # 7) Reemplazo: dropear columnas originales del bloque intensidad y conservar PCs
        #    + (opcional) conservar representativas fuera del PCA
        keep = set()
        if self.KEEP_REPRESENTATIVES:
            if "flow_pkts_payload.tot" in df.columns: keep.add("flow_pkts_payload.tot")
            if "payload_bytes_per_second" in df.columns: keep.add("payload_bytes_per_second")
            if "down_up_ratio" in df.columns: keep.add("down_up_ratio")

        drop_cols = [c for c in intensity_cols if c not in keep]
        df = df.drop(columns=drop_cols, errors="ignore")
        if drop_cols:
            report["notes"].append(f"Columnas de intensidad reemplazadas por intensity_pca_*: {drop_cols}")

        # 8) Final
        after = len(df)
        report["rows_after"] = after
        report["dropped"] = before - after
        report["notes"].extend([
            "Eliminadas redundancias (fwd/bwd_pkts_tot, flow_pkts_per_sec).",
            "Outliers en NORMAL cap P99 + imputados por mediana; ATTACK sin cap.",
            "Imputación por mediana + log1p (variables sesgadas) + z-score (entrenamiento).",
            "PCA aplicado; k elegido por ≥85% var.exp. (máx 4).",
            "PCs intensity_pca_* añadidas; columnas originales del bloque intensidad eliminadas (salvo representativas).",
        ])
        return df, report

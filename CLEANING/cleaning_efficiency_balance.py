# CLEANING/cleaning_efficiency_balance.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import os
import pickle
import numpy as np
import pandas as pd

class CleaningEfficiencyBalance:
    """
    Grupo 6: Eficiencia y Balance (ratios, simetrías y PCA del subespacio)

    Depende de:
      - 02_flow_intensity (duración y métricas base)
      - 03_payload_features (totales/avg payload consistentes, payload_ratio_log disponible)
      - 05_tcp_control (flags/ventanas saneados)

    Qué hace (resumen):
      1) Fuerza tipo numérico y no-finitos -> NaN.
      2) Deriva métricas clave (todas 'suaves' y reproducibles):
         - payload_efficiency = flow_bytes_payload.tot / (flow_bytes_payload.tot + flow_header_size_tot)   ∈ [0,1]
         - bytes_per_pkt_flow = flow_bytes_payload.tot / flow_pkts_payload.tot                              ≥ 0
         - sym_bytes = (fwd_bytes_payload.tot - bwd_bytes_payload.tot) / (fwd+bwd)                         ∈ [-1,1]
         - sym_pkts  = (fwd_pkts_payload.tot  - bwd_pkts_payload.tot ) / (fwd+bwd)                         ∈ [-1,1]
         - rate_ratio_log = log( (fwd_pkts_per_sec + 1) / (bwd_pkts_per_sec + 1) )  (si existen)
         - header_payload_ratio = flow_header_size_tot / (flow_bytes_payload.tot + 1)                       ≥ 0
         - Reusa payload_ratio_log de Grupo 3 si existe (no se recalcula).
      3) Coherencia física:
         - ratios en [0,1] y simetrías en [-1,1]; tolerancia 0.05 abs:
             * NORMAL: si |exceso|>0.05 -> drop
             * ATTACK/UNKNOWN: flag *_incoherent=True
         - Denominadores 0 manejados -> NaN (no drop por esto solo).
      4) Outliers SOLO en NORMAL:
         - P99 para variables no acotadas (bytes_per_pkt_flow, header_payload_ratio).
      5) Preprocesamiento PCA (igual esquema grupos 3/4):
         - Imputación por mediana (ENTRENO) y reuso en inferencia.
         - log1p en variables no negativas y sesgadas (skew>1).
         - z-score (media/std de entrenamiento).
         - PCA: k mínimo con varianza acumulada ≥ 80% (máx 3).
      6) Reemplazo:
         - Añade eff_pca_1..k (+ eff_pca_var_expl_total).
         - Por defecto CONSERVA fuera del PCA señales clave: payload_ratio_log (G3), sym_bytes, sym_pkts.
         - El resto de columnas del subespacio se dropean (configurable).
      7) Artefactos persistidos en CLEANING/.artifacts/efficiency_balance_prep.pkl
    """

    name: str = "06_efficiency_balance"
    depends_on: list[str] = ["02_flow_intensity", "03_payload_features", "05_tcp_control"]

    ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), ".artifacts")
    ARTIFACTS_FILE = os.path.join(ARTIFACTS_DIR, "efficiency_balance_prep.pkl")

    VAR_EXPL_TARGET = 0.80
    PCA_MAX_COMPONENTS = 3
    SKEW_THRESHOLD = 1.0
    KEEP_SIGNALS_OUTSIDE_PCA = True  # mantener payload_ratio_log, sym_bytes, sym_pkts

    # Columnas base esperadas para derivar ratios/simetrías (si faltan, se salta)
    BASE_COLS = [
        "flow_bytes_payload.tot", "fwd_bytes_payload.tot", "bwd_bytes_payload.tot",
        "flow_pkts_payload.tot",  "fwd_pkts_payload.tot",  "bwd_pkts_payload.tot",
        "flow_header_size_tot",
        "fwd_pkts_per_sec", "bwd_pkts_per_sec",  # opcionales
    ]

    # --- Utils numéricas ---
    def _to_numeric(self, df: pd.DataFrame, cols: List[str]) -> None:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    def _attack_col(self, df: pd.DataFrame) -> str:
        if "attack_type" in df.columns: return "attack_type"
        if "Attack_type" in df.columns: return "Attack_type"
        df["attack_type"] = "unknown"; return "attack_type"

    # --- Derivados del bloque ---
    def _derive_features(self, df: pd.DataFrame) -> List[str]:
        derived: List[str] = []
        # payload_efficiency in [0,1]
        if {"flow_bytes_payload.tot", "flow_header_size_tot"}.issubset(df.columns):
            num = pd.to_numeric(df["flow_bytes_payload.tot"], errors="coerce").clip(lower=0)
            den = (num + pd.to_numeric(df["flow_header_size_tot"], errors="coerce").clip(lower=0))
            eff = num / den.replace(0, np.nan)
            df["payload_efficiency"] = eff
            derived.append("payload_efficiency")

            # header / payload (no acotada, para PCA)
            df["header_payload_ratio"] = pd.to_numeric(df["flow_header_size_tot"], errors="coerce").clip(lower=0) / (num + 1.0)
            derived.append("header_payload_ratio")

        # bytes_per_pkt_flow >= 0
        if {"flow_bytes_payload.tot", "flow_pkts_payload.tot"}.issubset(df.columns):
            num = pd.to_numeric(df["flow_bytes_payload.tot"], errors="coerce").clip(lower=0)
            den = pd.to_numeric(df["flow_pkts_payload.tot"], errors="coerce")
            df["bytes_per_pkt_flow"] = num / den.replace(0, np.nan)
            derived.append("bytes_per_pkt_flow")

        # symmetries in [-1,1]
        if {"fwd_bytes_payload.tot", "bwd_bytes_payload.tot"}.issubset(df.columns):
            f = pd.to_numeric(df["fwd_bytes_payload.tot"], errors="coerce").clip(lower=0)
            b = pd.to_numeric(df["bwd_bytes_payload.tot"], errors="coerce").clip(lower=0)
            s = (f - b) / (f + b).replace(0, np.nan)
            df["sym_bytes"] = s
            derived.append("sym_bytes")
        if {"fwd_pkts_payload.tot", "bwd_pkts_payload.tot"}.issubset(df.columns):
            f = pd.to_numeric(df["fwd_pkts_payload.tot"], errors="coerce").clip(lower=0)
            b = pd.to_numeric(df["bwd_pkts_payload.tot"], errors="coerce").clip(lower=0)
            s = (f - b) / (f + b).replace(0, np.nan)
            df["sym_pkts"] = s
            derived.append("sym_pkts")

        # rate_ratio_log (opcional)
        if {"fwd_pkts_per_sec", "bwd_pkts_per_sec"}.issubset(df.columns):
            f = pd.to_numeric(df["fwd_pkts_per_sec"], errors="coerce").clip(lower=0)
            b = pd.to_numeric(df["bwd_pkts_per_sec"], errors="coerce").clip(lower=0)
            df["rate_ratio_log"] = np.log( (f + 1.0) / (b + 1.0) )
            derived.append("rate_ratio_log")

        return derived

    # --- Coherencia física ---
    def _physical_consistency(self, df: pd.DataFrame, attack_col: str) -> Dict[str, int]:
        stats = {"dropped_bounds": 0, "flagged_bounds": 0}
        is_normal = df[attack_col].astype(str).str.lower().eq("normal")
        not_normal = ~is_normal

        def _bound_check(series: pd.Series, low: float, high: float, tol: float = 0.05, name: str = ""):
            nonlocal df, stats
            if series.name not in df.columns: return
            s = pd.to_numeric(df[series.name], errors="coerce")
            low_violate  = s < (low - tol)
            high_violate = s > (high + tol)
            bad = (low_violate | high_violate)
            drop_mask = bad & is_normal
            stats["dropped_bounds"] += int(drop_mask.sum())
            keep_flag = bad & not_normal
            if f"{series.name}_incoherent" not in df.columns:
                df[f"{series.name}_incoherent"] = False
            df.loc[keep_flag, f"{series.name}_incoherent"] = True
            stats["flagged_bounds"] += int(keep_flag.sum())
            # aplicar drops
            if drop_mask.any():
                df = df.loc[~drop_mask].copy()

        # payload_efficiency ∈ [0,1]
        if "payload_efficiency" in df.columns:
            _bound_check(df["payload_efficiency"], 0.0, 1.0, tol=0.05, name="payload_efficiency")
        # symmetries ∈ [-1,1]
        if "sym_bytes" in df.columns:
            _bound_check(df["sym_bytes"], -1.0, 1.0, tol=0.05, name="sym_bytes")
        if "sym_pkts" in df.columns:
            _bound_check(df["sym_pkts"], -1.0, 1.0, tol=0.05, name="sym_pkts")
        return stats

    # --- Outliers SOLO NORMAL ---
    def _cap_p99_normals(self, df: pd.DataFrame, cols: List[str], is_normal: pd.Series) -> Dict[str, Any]:
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
            info[c] = {"p99": p99, "n_capped": ncap}
        return info

    # --- Artefactos PCA (fit/apply) ---
    def _fit_artifacts(self, df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
        from sklearn.decomposition import PCA
        # Medianas
        medians = {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in cols}
        X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").fillna(medians[c]) for c in cols})
        # Log columns: no negativas y sesgadas
        log_cols: List[str] = []
        for c in cols:
            s = X[c]
            if s.min() >= 0:
                sk = float(s.skew())
                if np.isfinite(sk) and sk > self.SKEW_THRESHOLD:
                    log_cols.append(c)
        for c in log_cols:
            X[c] = np.log1p(X[c])
        # Z-score
        means = {c: float(X[c].mean()) for c in cols}
        stds  = {c: float(X[c].std(ddof=0)) for c in cols}
        Z = pd.DataFrame({c: ((X[c]-means[c]) / stds[c] if stds[c] and np.isfinite(stds[c]) and stds[c] > 0 else 0.0)
                          for c in cols})
        # PCA
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
            df[f"eff_pca_{i+1}"] = comps[:, i]
        df["eff_pca_var_expl_total"] = float(np.sum(pca.explained_variance_ratio_))
        return df, {"n_features_in": len(cols), "n_components": k,
                    "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_]}

    # --- RUN ---
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = df.copy()
        before_rows = len(df)
        report: Dict[str, Any] = {
            "rows_before": before_rows,
            "rows_after": None,
            "dropped": 0,
            "derived_cols": [],
            "bounds_check": {},
            "cap_normals": {},
            "artifacts_mode": "apply" if os.path.exists(self.ARTIFACTS_FILE) else "fit",
            "pca": {"n_components": 0, "explained_variance_ratio": []},
            "notes": [],
        }

        # 0) coerción numérica base
        base_present = [c for c in self.BASE_COLS if c in df.columns]
        self._to_numeric(df, base_present)

        # 1) derivar features
        derived = self._derive_features(df)
        report["derived_cols"] = list(derived)

        # 2) coherencia física (rangos)
        attack_col = self._attack_col(df)
        bounds_stats = self._physical_consistency(df, attack_col)
        report["bounds_check"] = bounds_stats

        # 3) outliers SOLO NORMAL (en variables no acotadas)
        is_normal = df[attack_col].astype(str).str.lower().eq("normal")
        cap_targets = [c for c in ["bytes_per_pkt_flow", "header_payload_ratio"] if c in df.columns]
        report["cap_normals"] = self._cap_p99_normals(df, cap_targets, is_normal)

        # 4) preparar columnas del subespacio eficiencia/balance para PCA
        #    - No incluimos 'sym_*' (acotadas) ni 'payload_ratio_log' si existe (se conservan fuera del PCA).
        eff_cols: List[str] = []
        for c in ["payload_efficiency", "bytes_per_pkt_flow", "header_payload_ratio", "rate_ratio_log"]:
            if c in df.columns:
                eff_cols.append(c)
        # si no hay columnas suficientes, salimos sólo con derivados y coherencias
        if len(eff_cols) < 2:
            report["notes"].append("Pocas columnas para PCA de eficiencia/balance; se salta PCA.")
            after = len(df)
            report["rows_after"] = after
            report["dropped"] = before_rows - after
            return df, report

        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)

        # 5) Fit/Apply artefactos PCA
        if not os.path.exists(self.ARTIFACTS_FILE):
            artifacts = self._fit_artifacts(df, eff_cols)
            with open(self.ARTIFACTS_FILE, "wb") as f:
                pickle.dump(artifacts, f)
            df, pinfo = self._apply_artifacts(df, artifacts)
            report["pca"] = pinfo
            report["notes"].append("Artefactos de eficiencia/balance entrenados y guardados.")
        else:
            with open(self.ARTIFACTS_FILE, "rb") as f:
                artifacts = pickle.load(f)
            missing = [c for c in artifacts["cols"] if c not in df.columns]
            if missing:
                report["notes"].append(f"Se agregaron columnas faltantes para aplicar PCA: {missing}")
                for c in missing: df[c] = np.nan
            df, pinfo = self._apply_artifacts(df, artifacts)
            report["pca"] = pinfo

        # 6) Reemplazo: dropear columnas del subespacio (salvo señales clave)
        drop_cols = eff_cols.copy()
        if self.KEEP_SIGNALS_OUTSIDE_PCA:
            # mantener señales interpretables:
            # - payload_ratio_log (creada en Grupo 3 si existe)
            # - simetrías bytes/paquetes (rangos acotados)
            for keep in ["payload_ratio_log", "sym_bytes", "sym_pkts"]:
                # se mantienen si existen
                pass
            # quitar del drop las que queremos conservar (no forman parte de eff_cols, pero aclaramos)
        df = df.drop(columns=drop_cols, errors="ignore")
        report["notes"].append(f"Columnas de eficiencia/balance reemplazadas por eff_pca_*: {drop_cols}")
        if self.KEEP_SIGNALS_OUTSIDE_PCA:
            report["notes"].append("Se conservan fuera del PCA: payload_ratio_log (si existe), sym_bytes, sym_pkts.")

        # 7) Final
        after_rows = len(df)
        report["rows_after"] = after_rows
        report["dropped"] = before_rows - after_rows
        report["notes"].extend([
            "Ratios/simetrías derivadas y validadas (rangos).",
            "Outliers en NORMAL cap P99 para variables no acotadas.",
            "Imputación por mediana + log1p (skew>1) + z-score (entrenamiento).",
            "PCA de eficiencia/balance con ≥80% var.exp. (máx 3).",
        ])
        return df, report

# CLEANING/cleaning_payload_features.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import os
import pickle
import numpy as np
import pandas as pd

class CleaningPayloadFeatures:
    """
    Grupo 3: Payload / Cabeceras — Preprocesamiento + PCA + coherencia direccional

    Pasos aplicados:
      - Control de coherencia direccional en paquetes (flow_pkts_payload.tot vs fwd+bwd),
        con umbral de 15%:
          * Si la fila es NORMAL y viola la coherencia → se descarta.
          * Si la fila es ATTACK/UNKNOWN → se conserva pero se marca payload_pkts_incoherent=True.
      - Conversión a numérico e imputación por mediana (aprendida en entrenamiento).
      - Transformación log1p en métricas no negativas y muy sesgadas.
      - Cap P99 en clase NORMAL (sólo durante el ajuste / fit de artefactos).
      - Estandarización z-score usando medias/std del entrenamiento.
      - PCA sobre el subespacio payload para obtener payload_pca_1..k,
        con k elegido para alcanzar >=80% de varianza explicada (tope 4).
      - Se conserva por fuera del PCA la métrica direccional payload_ratio_log.
      - Opcionalmente se descartan las columnas originales del bloque payload.

    Artefactos persistidos en:
      CLEANING/.artifacts/payload_prep.pkl
        {
          "cols": [...],
          "medians": {col: mediana},
          "log_cols": [...],
          "means": {col: media},
          "stds": {col: std},
          "pca": fitted_PCA_object,
          "var_expl_ratio": [...],
          "cap_info_normals_p99": {col: {...}}
        }

    Estos artefactos permiten modo FIT (primera corrida, típicamente sobre train)
    y modo APPLY (corridas posteriores: valid/test, o datasets nuevos).
    """

    name: str = "03_payload_features"
    depends_on: list[str] = ["01_comm_type", "02_flow_intensity"]

    ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), ".artifacts")
    ARTIFACTS_FILE = os.path.join(ARTIFACTS_DIR, "payload_prep.pkl")

    VAR_EXPL_TARGET = 0.80
    PCA_MAX_COMPONENTS = 4
    SKEW_THRESHOLD = 1.0
    KEEP_ORIGINAL = False  # si False, se dropean las columnas crudas luego de generar PCs

    # Columnas candidatas del bloque de payload / cabeceras.
    # Ajustalas si tu CSV tiene nombres ligeramente distintos.
    EXPLICIT_COLS = [
        # paquetes payload totales
        "fwd_pkts_payload.tot", "bwd_pkts_payload.tot", "flow_pkts_payload.tot",
        # bytes payload totales
        "fwd_bytes_payload.tot", "bwd_bytes_payload.tot", "flow_bytes_payload.tot",
        # estadísticos de payload por paquete
        "fwd_pkt_payload.avg_bytes", "bwd_pkt_payload.avg_bytes", "flow_pkt_payload.avg_bytes",
        "fwd_pkt_payload.std_bytes", "bwd_pkt_payload.std_bytes", "flow_pkt_payload.std_bytes",
        "fwd_pkt_payload.min_bytes", "bwd_pkt_payload.min_bytes", "flow_pkt_payload.min_bytes",
        "fwd_pkt_payload.max_bytes", "bwd_pkt_payload.max_bytes", "flow_pkt_payload.max_bytes",
        # tasa / ritmo
        "payload_bytes_per_second",
        # tamaños de header (opcionalmente informativos del bloque)
        "fwd_header_size_tot", "bwd_header_size_tot", "flow_header_size_tot",
    ]

    REL_ERR_THRESH = 0.15  # coherencia direccional paquetes

    # ---------- helpers internos ----------

    def _collect_payload_cols(self, df: pd.DataFrame) -> List[str]:
        cols = [c for c in self.EXPLICIT_COLS if c in df.columns]
        # sin duplicados, en orden
        seen = set()
        ordered = []
        for c in cols:
            if c not in seen:
                ordered.append(c)
                seen.add(c)
        return ordered

    def _to_numeric(self, df: pd.DataFrame, cols: List[str]) -> None:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    def _attack_col(self, df: pd.DataFrame) -> str:
        if "attack_type" in df.columns:
            return "attack_type"
        if "Attack_type" in df.columns:
            return "Attack_type"
        # si no existe, la creo como unknown (esto preserva consistencia con otros módulos)
        df["attack_type"] = "unknown"
        return "attack_type"

    # ---------- coherencia direccional en paquetes ----------

    def _directional_consistency_pkts(
        self, df: pd.DataFrame, attack_col: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rep = {
            "dropped_rel_error_normal_pkts": 0,
            "flagged_incoherent_attacks_pkts": 0,
        }

        need = {
            "fwd_pkts_payload.tot",
            "bwd_pkts_payload.tot",
            "flow_pkts_payload.tot",
        }
        if not need.issubset(df.columns):
            # no puedo chequear consistencia si faltan columnas
            return df, rep

        fwd = pd.to_numeric(df["fwd_pkts_payload.tot"], errors="coerce").fillna(0)
        bwd = pd.to_numeric(df["bwd_pkts_payload.tot"], errors="coerce").fillna(0)
        flow = pd.to_numeric(df["flow_pkts_payload.tot"], errors="coerce")

        expected = fwd + bwd
        denom = expected.replace(0, np.nan).abs()
        rel_err = (flow - expected).abs() / denom
        bad_rel = (rel_err > self.REL_ERR_THRESH).fillna(False)

        is_normal = df[attack_col].astype(str).str.lower().eq("normal")
        drop_mask = bad_rel & is_normal

        rep["dropped_rel_error_normal_pkts"] = int(drop_mask.sum())

        # dropeo filas incoherentes SOLO si son normales
        df_kept = df.loc[~drop_mask].copy()

        # recalculo incoherencia en el df_kept y marco payload_pkts_incoherent=True
        is_normal_kept = df_kept[attack_col].astype(str).str.lower().eq("normal")
        fwd2 = pd.to_numeric(df_kept.get("fwd_pkts_payload.tot"), errors="coerce").fillna(0)
        bwd2 = pd.to_numeric(df_kept.get("bwd_pkts_payload.tot"), errors="coerce").fillna(0)
        flow2 = pd.to_numeric(df_kept.get("flow_pkts_payload.tot"), errors="coerce")
        expected2 = fwd2 + bwd2
        denom2 = expected2.replace(0, np.nan).abs()
        rel_err2 = (flow2 - expected2).abs() / denom2
        incoh_att = (rel_err2 > self.REL_ERR_THRESH).fillna(False) & (~is_normal_kept)

        if "payload_pkts_incoherent" not in df_kept.columns:
            df_kept["payload_pkts_incoherent"] = False
        df_kept.loc[incoh_att, "payload_pkts_incoherent"] = True
        rep["flagged_incoherent_attacks_pkts"] = int(incoh_att.sum())

        return df_kept, rep

    # ---------- artefactos fit/apply ----------

    def _fit_artifacts(
        self,
        df: pd.DataFrame,
        cols: List[str],
        class_mask_normal: pd.Series,
    ) -> Dict[str, Any]:
        """
        Aprende:
          - medianas
          - qué columnas van con log1p
          - medias/std para z-score
          - PCA (k mínimo con var.exp acumulada >=80%, tope 4)
          - info de capping P99 en tráfico normal
        """
        from sklearn.decomposition import PCA

        # 1) Medianas por columna
        medians = {c: float(df[c].median(skipna=True)) for c in cols}

        # 2) Imputar temporalmente con la mediana para análisis de skew / zscore
        X = df[cols].copy()
        for c in cols:
            X[c] = X[c].fillna(medians[c])

        # 3) Elegir qué columnas van a log1p
        log_cols: List[str] = []
        for c in cols:
            # sólo log si nunca (o casi nunca) es negativa y está sesgada fuerte
            if X[c].min() >= 0:
                skew = float(X[c].skew(skipna=True))
                if np.isfinite(skew) and skew > self.SKEW_THRESHOLD:
                    log_cols.append(c)
        for c in log_cols:
            X[c] = np.log1p(X[c])

        # 4) Cap P99 dentro de NORMAL (solo en fit)
        X_cap = X.copy()
        cap_info: Dict[str, Dict[str, float | int]] = {}
        for c in cols:
            vals_norm = X_cap.loc[class_mask_normal, c]
            if vals_norm.notna().sum() > 0:
                p99 = float(np.nanpercentile(vals_norm.values, 99))
                over_mask = class_mask_normal & (X_cap[c] > p99)
                cap_info[c] = {
                    "p99": p99,
                    "n_capped_normals": int(over_mask.sum()),
                }
                X_cap.loc[over_mask, c] = p99
            else:
                cap_info[c] = {"p99": np.nan, "n_capped_normals": 0}

        # 5) Z-score con medias/std de todo X_cap
        means = {c: float(X_cap[c].mean()) for c in cols}
        stds = {c: float(X_cap[c].std(ddof=0)) for c in cols}
        Z = pd.DataFrame(index=X_cap.index)
        for c in cols:
            st = stds[c]
            if st and np.isfinite(st) and st > 0:
                Z[c] = (X_cap[c] - means[c]) / st
            else:
                Z[c] = 0.0

        # 6) PCA: determinar k
        if len(cols) < 2:
            # con 0/1 variable no vale la pena PCA de verdad
            from sklearn.decomposition import PCA as _PCA
            pca = _PCA(n_components=1)
            pca.fit(Z[cols].values)
            k = 1
        else:
            pca_tmp = PCA(n_components=min(len(cols), self.PCA_MAX_COMPONENTS))
            pca_tmp.fit(Z[cols].values)
            cum = np.cumsum(pca_tmp.explained_variance_ratio_)
            k = int(np.searchsorted(cum, self.VAR_EXPL_TARGET) + 1)
            k = max(1, min(k, self.PCA_MAX_COMPONENTS, len(cols)))

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
            "cap_info_normals_p99": cap_info,
        }
        return artifacts

    def _apply_artifacts(
        self,
        df: pd.DataFrame,
        art: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Aplica las transformaciones aprendidas:
          imputación mediana -> log1p -> cap P99 (NO: ojo, eso sólo en fit) -> zscore -> PCA
        Nota: en APPLY NO volvemos a hacer el cap P99, usamos directamente medias/std del fit.
        """
        cols = art["cols"]
        medians = art["medians"]
        log_cols = set(art["log_cols"])
        means = art["means"]
        stds = art["stds"]
        pca = art["pca"]

        # asegurar existencia de columnas
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        X = df[cols].copy()
        for c in cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(medians[c])

        # log1p donde corresponda
        for c in cols:
            if c in log_cols:
                X[c] = np.log1p(X[c])

        # z-score con medias/std del entrenamiento
        Z = pd.DataFrame(index=X.index)
        for c in cols:
            st = stds[c]
            if st and np.isfinite(st) and st > 0:
                Z[c] = (X[c] - means[c]) / st
            else:
                Z[c] = 0.0

        comps = pca.transform(Z[cols].values)
        k = pca.n_components_
        for i in range(k):
            df[f"payload_pca_{i+1}"] = comps[:, i]
        df["payload_pca_var_expl_total"] = float(np.sum(pca.explained_variance_ratio_))

        pinfo = {
            "n_features_in": len(cols),
            "n_components": k,
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        }
        return df, pinfo

    # ---------- run principal ----------

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = df.copy()
        before_rows = len(df)

        report: Dict[str, Any] = {
            "rows_before": before_rows,
            "rows_after": None,
            "dropped": 0,
            "directional_check": {},
            "payload_cols_used": [],
            "artifacts_mode": "apply" if os.path.exists(self.ARTIFACTS_FILE) else "fit",
            "log_cols": [],
            "pca": {"n_components": 0, "explained_variance_ratio": []},
            "notes": [],
        }

        # 0) coherencia direccional en paquetes con umbral 15%
        attack_col = self._attack_col(df)
        df, dir_rep = self._directional_consistency_pkts(df, attack_col)
        report["directional_check"] = dir_rep

        # 1) recolectar columnas del bloque payload
        payload_cols = self._collect_payload_cols(df)
        self._to_numeric(df, payload_cols)

        if not payload_cols:
            # No hay columnas de payload --> nada que hacer
            report["notes"].append("No se detectaron columnas payload; módulo 3 omitido.")
            report["rows_after"] = len(df)
            report["dropped"] = before_rows - len(df)
            return df, report

        report["payload_cols_used"] = list(payload_cols)

        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)

        # 2) FIT o APPLY de artefactos
        if not os.path.exists(self.ARTIFACTS_FILE):
            # ENTRENAMIENTO
            try:
                is_normal = df[attack_col].astype(str).str.lower().eq("normal")
                artifacts = self._fit_artifacts(df, payload_cols, is_normal)
            except Exception as e:
                report["notes"].append(f"Falló _fit_artifacts: {e}")
                # terminamos igual con df parcial
                after_rows = len(df)
                report["rows_after"] = after_rows
                report["dropped"] = before_rows - after_rows
                return df, report

            with open(self.ARTIFACTS_FILE, "wb") as f:
                pickle.dump(artifacts, f)

            df, pinfo = self._apply_artifacts(df, artifacts)
            report["pca"] = pinfo
            report["log_cols"] = artifacts["log_cols"]
            report["notes"].append("Artefactos de payload entrenados y guardados.")
        else:
            # INFERENCIA
            with open(self.ARTIFACTS_FILE, "rb") as f:
                artifacts = pickle.load(f)

            missing = [c for c in artifacts["cols"] if c not in df.columns]
            if missing:
                report["notes"].append(
                    f"Columnas faltantes para aplicar PCA, se crearon con NaN/mediana: {missing}"
                )

            df, pinfo = self._apply_artifacts(df, artifacts)
            report["pca"] = pinfo
            report["log_cols"] = artifacts["log_cols"]

        # 3) payload_ratio_log (fuera del PCA)
        #   Queremos log( (fwd_bytes+1) / (bwd_bytes+1) ), robusto incluso si faltan columnas.
        fwd_raw = df["fwd_bytes_payload.tot"] if "fwd_bytes_payload.tot" in df.columns else pd.Series(np.nan, index=df.index)
        bwd_raw = df["bwd_bytes_payload.tot"] if "bwd_bytes_payload.tot" in df.columns else pd.Series(np.nan, index=df.index)

        fbytes = pd.to_numeric(fwd_raw, errors="coerce").fillna(0.0)
        bbytes = pd.to_numeric(bwd_raw, errors="coerce").fillna(0.0)

        # ratio seguro
        ratio = (fbytes + 1.0) / (bbytes + 1.0)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        df["payload_ratio_log"] = np.log(ratio)

        # 4) Dropear columnas crudas del bloque payload si no las queremos más
        if not self.KEEP_ORIGINAL:
            df = df.drop(columns=[c for c in payload_cols if c in df.columns], errors="ignore")
            report["notes"].append(
                "Columnas originales de payload reemplazadas por payload_pca_* y payload_ratio_log."
            )

        # 5) Cierre
        after_rows = len(df)
        report["rows_after"] = after_rows
        report["dropped"] = before_rows - after_rows
        report["notes"].extend([
            "Imputación por mediana (fit) y reuso en apply.",
            "log1p sólo en columnas no negativas y muy sesgadas.",
            "Cap P99 en tráfico NORMAL sólo en el ajuste (no en apply).",
            "Z-score con medias y std del ajuste.",
            "PCA reteniendo >=80% var.expl (hasta 4 componentes).",
            "payload_ratio_log conservada como feature independiente.",
        ])

        return df, report

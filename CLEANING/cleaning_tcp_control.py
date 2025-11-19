# CLEANING/cleaning_tcp_control.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Iterable
import os, pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class CleaningTCPControl:
    """
    Grupo 5: Control TCP (flags y ventanas) con FIT/APPLY + PCA persistente

    Depende de:
      - 01_comm_type

    Pipeline (train/FIT):
      1) Validación semántica:
         - Flags enteros no negativos
         - Ventanas no negativas
         - Drops determinísticos SOLO en NORMAL (incoherencias TCP)
           y flag 'tcp_semantics_incoherent' en ATTACK/UNKNOWN.
      2) Clipping por clase (Normal vs Attack/Unknown) en flags y ventanas:
         - Aprende P1–P99 por clase y los guarda (percentiles).
      3) Robust-scaling de ventanas (mediana/IQR) → crea columnas *_r (se conservan).
      4) PCA sobre subespacio TCP:
         - Features = ventanas robustas *_r + flags (estandarizados)
         - Imputación por mediana → log1p (no negativas con skew>1) → z-score → PCA
         - Elige k mínimo con ≥80% var. explicada (máx 4)
         - Agrega tcp_pca_1..k y tcp_pca_var_expl_total
      5) Drop de ventanas crudas (se mantienen *_r) y se mantienen flags.

    En APPLY (valid/test):
      - Reaplica percentiles, robust-scaling, imputación/log/zscore y PCA con artefactos guardados.

    Artefactos (.pkl):
      CLEANING/.artifacts/tcp_control_prep.pkl
    """

    name: str = "05_tcp_control"
    depends_on: list[str] = ["01_comm_type"]

    ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), ".artifacts")
    ARTIFACTS_FILE = os.path.join(ARTIFACTS_DIR, "tcp_control_prep.pkl")

    VAR_EXPL_TARGET = 0.80
    PCA_MAX_COMPONENTS = 4
    SKEW_THRESHOLD = 1.0

    FLAG_COLS = [
        "flow_SYN_flag_count", "flow_ACK_flag_count", "flow_FIN_flag_count",
        "flow_RST_flag_count", "flow_PSH_flag_count", "flow_URG_flag_count",
        "fwd_PSH_flag_count", "bwd_PSH_flag_count"  # si existen direccionales
    ]
    WINDOW_COLS = [
        "fwd_init_window_size", "bwd_init_window_size",
        "fwd_last_window_size", "bwd_last_window_size"
    ]
    PKT_TOT_COLS = ["fwd_pkts_tot", "bwd_pkts_tot"]
    LABEL_CANDS = ["attack_type", "Attack_type", "label", "Label"]

    # ----------------- utils -----------------
    def _attack_col(self, df: pd.DataFrame) -> str:
        for c in self.LABEL_CANDS:
            if c in df.columns:
                return c
        df["attack_type"] = "unknown"
        return "attack_type"

    def _to_numeric(self, df: pd.DataFrame, cols: Iterable[str]) -> None:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    @staticmethod
    def _set_non_negative(df: pd.DataFrame, cols: Iterable[str]) -> int:
        n = 0
        for c in cols:
            if c in df.columns:
                mask = df[c] < 0
                if mask.any():
                    df.loc[mask, c] = np.nan
                    n += int(mask.sum())
        return n

    @staticmethod
    def _round_flags_to_int(df: pd.DataFrame, cols: Iterable[str]) -> int:
        changed = 0
        for c in cols:
            if c in df.columns:
                before = df[c].copy()
                df[c] = pd.to_numeric(df[c], errors="coerce").round()
                df.loc[df[c] < 0, c] = 0
                df[c] = df[c].fillna(0).astype("Int64")
                changed += int((before.astype(float) != df[c].astype(float)).sum(skipna=True))
        return changed

    # -------- incoherencias TCP (drops sólo NORMAL) --------
    def _drop_non_tcp_with_flags(self, df: pd.DataFrame, is_normal: pd.Series) -> Tuple[pd.DataFrame, int, int]:
        cols = [c for c in self.FLAG_COLS if c in df.columns]
        if not cols:
            return df, 0, 0
        any_flag = df[cols].fillna(0).sum(axis=1) > 0
        proto = df.get("proto", "unknown")
        proto = proto.astype(str).str.lower()
        incoh = any_flag & (proto != "tcp")
        drop = incoh & is_normal
        kept = incoh & (~is_normal)
        d = int(drop.sum())
        df = df.loc[~drop].copy()
        kept = kept.loc[df.index]
        if "tcp_semantics_incoherent" not in df.columns:
            df["tcp_semantics_incoherent"] = False
        df.loc[kept, "tcp_semantics_incoherent"] = True
        return df, d, int(kept.sum())

    def _drop_syn_and_rst(self, df: pd.DataFrame, is_normal: pd.Series) -> Tuple[pd.DataFrame, int, int]:
        syn = df.get("flow_SYN_flag_count"); rst = df.get("flow_RST_flag_count")
        if syn is None or rst is None:
            return df, 0, 0
        pattern = (syn.fillna(0) > 0) & (rst.fillna(0) > 0)
        drop = pattern & is_normal
        kept = pattern & (~is_normal)
        d = int(drop.sum())
        df = df.loc[~drop].copy()
        kept = kept.loc[df.index]
        if "tcp_semantics_incoherent" not in df.columns:
            df["tcp_semantics_incoherent"] = False
        df.loc[kept, "tcp_semantics_incoherent"] = True
        return df, d, int(kept.sum())

    def _drop_rst_and_ack_without_syn(self, df: pd.DataFrame, is_normal: pd.Series) -> Tuple[pd.DataFrame, int, int]:
        rst = df.get("flow_RST_flag_count"); ack = df.get("flow_ACK_flag_count"); syn = df.get("flow_SYN_flag_count")
        if rst is None or ack is None or syn is None:
            return df, 0, 0
        pattern = (rst.fillna(0) > 0) & (ack.fillna(0) > 0) & (syn.fillna(0) == 0)
        drop = pattern & is_normal
        kept = pattern & (~is_normal)
        d = int(drop.sum())
        df = df.loc[~drop].copy()
        kept = kept.loc[df.index]
        if "tcp_semantics_incoherent" not in df.columns:
            df["tcp_semantics_incoherent"] = False
        df.loc[kept, "tcp_semantics_incoherent"] = True
        return df, d, int(kept.sum())

    def _drop_flags_gt_packets(self, df: pd.DataFrame, is_normal: pd.Series) -> Tuple[pd.DataFrame, int, int]:
        if not {"fwd_pkts_tot", "bwd_pkts_tot"}.issubset(df.columns):
            return df, 0, 0
        pkt_tot = df["fwd_pkts_tot"].fillna(0) + df["bwd_pkts_tot"].fillna(0)
        cols = [c for c in self.FLAG_COLS if c in df.columns]
        if not cols:
            return df, 0, 0
        flag_sum = df[cols].fillna(0).sum(axis=1)
        incoh = flag_sum > pkt_tot
        drop = incoh & is_normal
        kept = incoh & (~is_normal)
        d = int(drop.sum())
        df = df.loc[~drop].copy()
        kept = kept.loc[df.index]
        if "tcp_semantics_incoherent" not in df.columns:
            df["tcp_semantics_incoherent"] = False
        df.loc[kept, "tcp_semantics_incoherent"] = True
        return df, d, int(kept.sum())

    # -------- percentiles por clase (fit/apply) --------
    def _fit_percentiles(self, df: pd.DataFrame, cols: List[str], label_col: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"normal": {}, "attack_unknown": {}}
        is_normal = df[label_col].astype(str).str.lower().eq("normal")
        is_attack_or_unknown = ~is_normal
        for c in cols:
            s_norm = pd.to_numeric(df.loc[is_normal, c], errors="coerce")
            s_attk = pd.to_numeric(df.loc[is_attack_or_unknown, c], errors="coerce")
            if s_norm.notna().sum() > 0:
                out["normal"][c] = {
                    "p1": float(np.nanpercentile(s_norm, 1)),
                    "p99": float(np.nanpercentile(s_norm, 99)),
                }
            if s_attk.notna().sum() > 0:
                out["attack_unknown"][c] = {
                    "p1": float(np.nanpercentile(s_attk, 1)),
                    "p99": float(np.nanpercentile(s_attk, 99)),
                }
        return out

    def _apply_percentiles(self, df: pd.DataFrame, specs: Dict[str, Any], label_col: str) -> None:
        is_normal = df[label_col].astype(str).str.lower().eq("normal")
        is_attack_or_unknown = ~is_normal
        for c in set(list(specs.get("normal", {}).keys()) + list(specs.get("attack_unknown", {}).keys())):
            if c not in df.columns:
                continue
            x = pd.to_numeric(df[c], errors="coerce")
            if c in specs.get("normal", {}):
                p1 = specs["normal"][c]["p1"]; p99 = specs["normal"][c]["p99"]
                df.loc[is_normal & x.notna() & (x < p1), c] = p1
                df.loc[is_normal & x.notna() & (x > p99), c] = p99
            if c in specs.get("attack_unknown", {}):
                p1 = specs["attack_unknown"][c]["p1"]; p99 = specs["attack_unknown"][c]["p99"]
                df.loc[is_attack_or_unknown & x.notna() & (x < p1), c] = p1
                df.loc[is_attack_or_unknown & x.notna() & (x > p99), c] = p99

    # -------- robust-scaling de ventanas (fit/apply) --------
    def _fit_window_stats(self, df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for c in cols:
            if c in df.columns:
                med = float(pd.to_numeric(df[c], errors="coerce").median())
                q75 = float(pd.to_numeric(df[c], errors="coerce").quantile(0.75))
                q25 = float(pd.to_numeric(df[c], errors="coerce").quantile(0.25))
                stats[c] = {"median": med, "iqr": max(q75 - q25, 0.0)}
        return stats

    def _apply_window_stats(self, df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> List[str]:
        created = []
        for c, st in stats.items():
            if c not in df.columns:
                continue
            x = pd.to_numeric(df[c], errors="coerce")
            med, iqr = st["median"], st["iqr"]
            if iqr and np.isfinite(iqr) and iqr > 0:
                df[f"{c}_r"] = (x - med) / iqr
            else:
                df[f"{c}_r"] = np.nan
            created.append(f"{c}_r")
        return created

    # -------- PCA fit/apply --------
    def _fit_pca_artifacts(self, df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
        # 1) imputaciones por mediana
        medians = {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in cols}
        X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").fillna(medians[c]) for c in cols})

        # 2) log1p en no-negativas con skew fuerte
        log_cols: List[str] = []
        for c in cols:
            s = X[c]
            if s.min() >= 0:
                sk = float(s.skew())
                if np.isfinite(sk) and sk > self.SKEW_THRESHOLD:
                    X[c] = np.log1p(s)
                    log_cols.append(c)

        # 3) z-score
        means = {c: float(X[c].mean()) for c in cols}
        stds  = {c: float(X[c].std(ddof=0)) for c in cols}
        Z = pd.DataFrame({c: ((X[c]-means[c]) / stds[c] if stds[c] and np.isfinite(stds[c]) and stds[c] > 0 else 0.0)
                          for c in cols})

        # 4) PCA k mínimo ≥80% (máx 4)
        pca_tmp = PCA(n_components=min(len(cols), self.PCA_MAX_COMPONENTS))
        pca_tmp.fit(Z[cols].values)
        cum = np.cumsum(pca_tmp.explained_variance_ratio_)
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

    def _apply_pca_artifacts(self, df: pd.DataFrame, art: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
            df[f"tcp_pca_{i+1}"] = comps[:, i]
        df["tcp_pca_var_expl_total"] = float(np.sum(pca.explained_variance_ratio_))

        info = {"n_components": k, "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_]}
        return df, info

    # ----------------- RUN -----------------
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = df.copy()
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        before = len(df)

        report: Dict[str, Any] = {
            "rows_before": before,
            "rows_after": None,
            "dropped": 0,
            "fixed_negatives_to_nan": 0,
            "flag_counts_rounded_to_int": 0,
            "drops": {
                "non_tcp_with_flags_dropped": 0,
                "non_tcp_with_flags_flagged": 0,
                "syn_and_rst_dropped": 0,
                "syn_and_rst_flagged": 0,
                "rst_and_ack_without_syn_dropped": 0,
                "rst_and_ack_without_syn_flagged": 0,
                "flags_gt_packets_dropped": 0,
                "flags_gt_packets_flagged": 0,
            },
            "percentiles_by_class": {},
            "window_robust_stats": {},
            "pca": {},
            "notes": []
        }

        # 0) tipado y saneo básico
        numeric_cols = list(set(self.FLAG_COLS + self.WINDOW_COLS + self.PKT_TOT_COLS))
        self._to_numeric(df, numeric_cols)
        report["fixed_negatives_to_nan"] = self._set_non_negative(df, numeric_cols)
        report["flag_counts_rounded_to_int"] = self._round_flags_to_int(df, self.FLAG_COLS)

        # 1) contexto de clase
        label_col = self._attack_col(df)
        is_normal = df[label_col].astype(str).str.lower().eq("normal")
        if "proto" not in df.columns:
            df["proto"] = "unknown"

        # 2) incoherencias TCP (drop SOLO en NORMAL)
        df, d, f = self._drop_non_tcp_with_flags(df, is_normal)
        report["drops"]["non_tcp_with_flags_dropped"] = d
        report["drops"]["non_tcp_with_flags_flagged"] = f

        # recalcular máscara luego de drops
        is_normal = df[label_col].astype(str).str.lower().eq("normal")
        df, d, f = self._drop_syn_and_rst(df, is_normal)
        report["drops"]["syn_and_rst_dropped"] = d
        report["drops"]["syn_and_rst_flagged"] = f

        is_normal = df[label_col].astype(str).str.lower().eq("normal")
        df, d, f = self._drop_rst_and_ack_without_syn(df, is_normal)
        report["drops"]["rst_and_ack_without_syn_dropped"] = d
        report["drops"]["rst_and_ack_without_syn_flagged"] = f

        is_normal = df[label_col].astype(str).str.lower().eq("normal")
        df, d, f = self._drop_flags_gt_packets(df, is_normal)
        report["drops"]["flags_gt_packets_dropped"] = d
        report["drops"]["flags_gt_packets_flagged"] = f

        # 3) percentiles por clase (fit/apply) para ventanas y flags
        clip_cols = [c for c in (self.WINDOW_COLS + self.FLAG_COLS) if c in df.columns]

        if not os.path.exists(self.ARTIFACTS_FILE):
            pct = self._fit_percentiles(df, clip_cols, label_col)
            self._apply_percentiles(df, pct, label_col)
        else:
            with open(self.ARTIFACTS_FILE, "rb") as f:
                art_loaded = pickle.load(f)
            pct = art_loaded.get("percentiles_by_class", {})
            if pct:
                self._apply_percentiles(df, pct, label_col)
            else:
                # fallback seguro si .pkl viejo no lo tiene
                pct = self._fit_percentiles(df, clip_cols, label_col)
                self._apply_percentiles(df, pct, label_col)
        report["percentiles_by_class"] = pct

        # 4) robust-scaling de ventanas (mediana/IQR) → *_r (fit/apply)
        present_win = [c for c in self.WINDOW_COLS if c in df.columns]
        if not os.path.exists(self.ARTIFACTS_FILE):
            win_stats = self._fit_window_stats(df, present_win)
            created_r = self._apply_window_stats(df, win_stats)
        else:
            with open(self.ARTIFACTS_FILE, "rb") as f:
                art_loaded = pickle.load(f)
            win_stats = art_loaded.get("window_robust_stats", self._fit_window_stats(df, present_win))
            created_r = self._apply_window_stats(df, win_stats)
        report["window_robust_stats"] = win_stats

        # 5) preparar features para PCA = ventanas robustas *_r + flags
        pca_features: List[str] = []
        pca_features += [f"{c}_r" for c in present_win if f"{c}_r" in df.columns]
        pca_features += [c for c in self.FLAG_COLS if c in df.columns]

        # 6) PCA (fit/apply con imputación/log/zscore persistentes)
        if not os.path.exists(self.ARTIFACTS_FILE):
            pca_art = self._fit_pca_artifacts(df, pca_features)
            with open(self.ARTIFACTS_FILE, "wb") as f:
                pickle.dump({
                    "percentiles_by_class": pct,
                    "window_robust_stats": win_stats,
                    **pca_art
                }, f)
            df, pinfo = self._apply_pca_artifacts(df, pca_art)
            report["pca"] = pinfo
            report["notes"].append("Artefactos TCP (percentiles, robust, PCA) entrenados y guardados.")
        else:
            with open(self.ARTIFACTS_FILE, "rb") as f:
                art_loaded = pickle.load(f)
            # asegurar que existan features declaradas en art
            for c in art_loaded.get("cols", []):
                if c not in df.columns:
                    df[c] = np.nan
            df, pinfo = self._apply_pca_artifacts(df, art_loaded)
            report["pca"] = pinfo
            report["notes"].append("Artefactos TCP aplicados (sin reentrenar).")

        # 7) columnas finales: mantener flags + ventanas robustas; dropear ventanas crudas
        drop_raw = [c for c in present_win if c in df.columns]
        df.drop(columns=drop_raw, inplace=True, errors="ignore")
        report["notes"].append(f"Ventanas crudas removidas: {drop_raw} (se conservan *_r y flags).")

        after = len(df)
        report["rows_after"] = after
        report["dropped"] = before - after
        return df, report

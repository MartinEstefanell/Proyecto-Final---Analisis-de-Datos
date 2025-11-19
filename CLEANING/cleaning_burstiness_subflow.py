# CLEANING/cleaning_final_assembly.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Set
import os
import pickle
import numpy as np
import pandas as pd

class CleaningFinalAssembly:
    """
    Grupo 7: Ensamblado Final y QA (+ PCA opcional)

    Depende de:
      - 01_comm_type
      - 02_flow_intensity
      - 03_payload_features
      - 04_traffic_rate
      - 05_tcp_control
      - 06_efficiency_balance

    Objetivos:
      1) Normalizar columna de etiqueta y derivar binaria is_attack (0/1).
      2) Codificar categóricas clave (proto, service) con mapeos persistentes (entreno -> .artifacts).
      3) Remover columnas:
         - Identificadores/leakage (timestamps, IPs, puertos, IDs) si existen.
         - Casi-constantes (>=99.5% mismo valor).
         - Duplicadas exactas.
         - Colineales perfectas (|corr| >= 0.999) entre numéricas.
      4) Garantizar numerics: no NaN/Inf (imputación conservadora por mediana).
      5) Ordenar columnas (label primero, luego numéricas y dummies) y reportar metadatos.
      6) (Opcional) PCA sobre las numéricas resultantes:
         - Exporta varianza, loadings, scree y proyección 2D si se provee pca_out_dir.

    Artefactos:
      CLEANING/.artifacts/final_assembly.pkl
        {
          "categorical_maps": {"proto": {...}, "service": {...}},
          "drop_id_cols": [...],
          "constant_cols": [...],
          "duplicate_cols": [...],
          "corr_dropped_pairs": [(col_keep, col_drop), ...],
          "numeric_medians": {col: med, ...},
          "columns_order": [..., ...]
        }

    Nota: Este módulo NO guarda CSV. El guardado de PCA es opcional (XLSX/PNG) y
    solo ocurre si se pasa pca_out_dir a run(...).
    """

    name: str = "07_burstiness_subflow"
    depends_on: list[str] = [
        "01_comm_type",
        "02_flow_intensity",
        "03_payload_features",
        "04_traffic_rate",
        "05_tcp_control",
        "06_efficiency_balance",
    ]

    ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), ".artifacts")
    ARTIFACTS_FILE = os.path.join(ARTIFACTS_DIR, "final_assembly.pkl")

    # ---------------- Config ----------------
    TOP_N_CATEGORIES = 30
    CONST_THRESH = 0.995
    CORR_THRESH = 0.999
    PCA_MAX_COMPONENTS = 7
    MAX_POINTS_JITTER = 1500  # (solo relevante si se agregan otras figuras)
    ID_GUESS_PATTERNS = [
        "flow_id", "id", "uuid", "session", "timestamp", "ts", "time",
        "src_ip", "dst_ip", "source_ip", "destination_ip",
        "src_port", "sport", "dst_port", "dport",
        "mac", "eth", "vlan"
    ]
    LABEL_CANDIDATES = ["attack_type", "Attack_type", "label", "Label", "class"]
    CATEGORICAL_CANDIDATES = ["proto", "service"]

    # ---------------- Utils ----------------
    def _ensure_artifacts_dir(self):
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)

    def _detect_label_col(self, df: pd.DataFrame) -> str:
        for c in self.LABEL_CANDIDATES:
            if c in df.columns:
                return c
        # Si no existe, crear una etiqueta desconocida
        df["attack_type"] = "unknown"
        return "attack_type"

    def _normalize_label(self, df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rep = {"label_col": label_col, "unique_labels_before": [], "unique_labels_after": []}
        df[label_col] = df[label_col].astype(str)
        rep["unique_labels_before"] = sorted(df[label_col].unique().tolist())
        df[label_col] = df[label_col].str.strip()
        is_normal = df[label_col].str.lower().eq("normal")
        df.loc[is_normal, label_col] = "normal"
        rep["unique_labels_after"] = sorted(df[label_col].unique().tolist())
        df["is_attack"] = (~df[label_col].eq("normal")).astype("int8")
        return df, rep

    def _guess_id_cols(self, df: pd.DataFrame) -> List[str]:
        drops = []
        names = [str(c).lower() for c in df.columns]
        for idx, name in enumerate(names):
            for pat in self.ID_GUESS_PATTERNS:
                if pat in name:
                    drops.append(df.columns[idx])
                    break
        for c in df.columns:
            if df[c].nunique(dropna=True) >= 0.99 * len(df):
                if c not in self.LABEL_CANDIDATES and c not in ["is_attack"]:
                    drops.append(c)
        seen = set(); ordered=[]
        for c in drops:
            if c not in seen:
                seen.add(c); ordered.append(c)
        return ordered

    def _one_hot_fit(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        s = df[col].astype(str).fillna("__missing__")
        top = s.value_counts().head(self.TOP_N_CATEGORIES).index.tolist()
        mapping = {v: v for v in top}
        mapping["__other__"] = "__other__"
        return {"col": col, "top_values": top, "mapping": mapping}

    def _one_hot_apply(self, df: pd.DataFrame, spec: Dict[str, Any]) -> List[str]:
        col = spec["col"]
        top = set(spec["top_values"])
        out_cols = []
        s = df[col].astype(str).fillna("__missing__")
        s = s.where(s.isin(top), "__other__")
        for v in list(top) + ["__other__"]:
            newc = f"{col}__{v}"
            df[newc] = (s == v).astype("int8")
            out_cols.append(newc)
        return out_cols

    def _remove_constant_cols(self, df: pd.DataFrame) -> List[str]:
        drops = []
        for c in df.columns:
            vc = df[c].value_counts(dropna=False)
            if not vc.empty and (vc.iloc[0] / len(df)) >= self.CONST_THRESH:
                if c not in self.LABEL_CANDIDATES and c not in ["is_attack"]:
                    drops.append(c)
        if drops:
            df.drop(columns=drops, inplace=True, errors="ignore")
        return drops

    def _remove_duplicate_cols(self, df: pd.DataFrame) -> List[str]:
        drops = []
        hashes = {}
        for c in df.columns:
            s = df[c]
            key = (tuple(pd.isna(s)), tuple(s.fillna("__nan__").astype(str)))
            if key in hashes:
                drops.append(c)
            else:
                hashes[key] = c
        if drops:
            df.drop(columns=drops, inplace=True, errors="ignore")
        return drops

    def _remove_collinear(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        num_cols = [c for c in df.columns if c not in self.LABEL_CANDIDATES + ["is_attack"] and pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 2:
            return []
        corr = df[num_cols].corr(method="pearson").abs()
        to_drop: Set[str] = set()
        pairs: List[Tuple[str, str]] = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                if c1 in to_drop or c2 in to_drop:
                    continue
                val = corr.iloc[i, j]
                if np.isfinite(val) and val >= self.CORR_THRESH:
                    to_drop.add(c2)
                    pairs.append((c1, c2))
        if to_drop:
            df.drop(columns=list(to_drop), inplace=True, errors="ignore")
        return pairs

    def _numeric_medians_fit(self, df: pd.DataFrame) -> Dict[str, float]:
        meds = {}
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and c not in ["is_attack"]:
                meds[c] = float(pd.to_numeric(df[c], errors="coerce").median())
        return meds

    def _numeric_medians_apply(self, df: pd.DataFrame, meds: Dict[str, float]) -> None:
        for c, m in meds.items():
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                s = s.replace([np.inf, -np.inf], np.nan).fillna(m)
                df[c] = s

    # ---------------- Helpers para exportar (PCA) ----------------
    @staticmethod
    def _ensure_outdir(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _find_converter() -> str | None:
        import sys
        here = os.path.dirname(__file__)
        candidates = [
            os.path.join(here, "csv_to_xlsx_format.py"),
            os.path.join(os.path.dirname(here), "csv_to_xlsx_format.py"),
            os.path.join(os.getcwd(), "csv_to_xlsx_format.py"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        # si está en PATH como script/entrypoint
        return "csv_to_xlsx_format"

    @classmethod
    def _save_via_converter(cls, df: pd.DataFrame, out_dir: str, base_name: str, sheet_name: str = "Sheet1") -> str:
        import sys, tempfile, subprocess
        cls._ensure_outdir(out_dir)
        converter = cls._find_converter()
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_csv = os.path.join(tmpd, f"{base_name}.csv")
            df.to_csv(tmp_csv, index=False)
            xlsx_out = os.path.join(out_dir, f"{base_name}.xlsx")
            cmd = [sys.executable, converter, tmp_csv, xlsx_out, "--sheet-name", sheet_name, "--outdir", out_dir]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(f"[XLSX] {xlsx_out}")
            return xlsx_out

    @classmethod
    def _save_plot(cls, fig, out_dir: str, name: str):
        import matplotlib.pyplot as plt
        cls._ensure_outdir(out_dir)
        fig.tight_layout(pad=1.0)
        out = os.path.join(out_dir, name)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[PNG] {out}")

    # ---------------- PCA block ----------------
    def _pca_block(self, df: pd.DataFrame, label_col: str, out_dir: str | None, report: Dict[str, Any]) -> None:
        """
        Ejecuta PCA sobre todas las columnas numéricas finales (post-ensamblado).
        - Guarda varianza, loadings, scree y proyección 2D si out_dir no es None.
        - Actualiza el report con un resumen.
        """
        # variables numéricas elegibles (excluye la binaria y la etiqueta si quedó numérica)
        vars_for_pca = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])
                        and c not in ("is_attack",)]
        # Si la etiqueta quedó numérica (raro), excluirla por nombre:
        vars_for_pca = [c for c in vars_for_pca if c not in self.LABEL_CANDIDATES]

        # Dataset para PCA
        pca_df = df[vars_for_pca + ["is_attack"]].copy() if vars_for_pca else pd.DataFrame()
        pca_df = pca_df.replace([np.inf, -np.inf], np.nan).dropna()
        n_rows = len(pca_df)
        n_vars = len(vars_for_pca)

        # Resumen previo
        report.setdefault("pca", {})
        report["pca"].update({"n_rows": int(n_rows), "n_vars": int(n_vars), "notes": []})

        if n_rows < 50 or n_vars < 2:
            report["pca"]["notes"].append("PCA omitido (insuficientes filas o variables).")
            if out_dir:
                # dejar constancia en XLSX
                self._save_via_converter(pd.DataFrame([{"note": "PCA skipped (insufficient rows/vars)"}]),
                                         out_dir, "final_pca_info", "pca_info")
            return

        # Standardize + PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        X = pca_df[vars_for_pca].values
        Xs = StandardScaler().fit_transform(X)
        k = min(self.PCA_MAX_COMPONENTS, n_vars)
        pca = PCA(n_components=k, random_state=42)
        Xp = pca.fit_transform(Xs)

        var_exp = (pca.explained_variance_ratio_ * 100.0).round(2)
        cum_exp = np.cumsum(pca.explained_variance_ratio_) * 100.0
        cum_exp = np.round(cum_exp, 2)

        # Actualizar resumen en report
        report["pca"].update({
            "explained_variance_%": var_exp.tolist(),
            "cumulative_%": cum_exp.tolist(),
            "top3_cumulative_%": float(cum_exp[min(2, k-1)]) if k >= 1 else 0.0
        })

        # Si no hay directorio de salida, no exportamos archivos
        if not out_dir:
            report["pca"]["notes"].append("PCA ejecutado sin exportar archivos (pca_out_dir=None).")
            return

        # ----- Exportar XLSX -----
        var_df = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(k)],
            "Explained_Variance_%": var_exp,
            "Cumulative_%": cum_exp
        })
        self._save_via_converter(var_df, out_dir, "final_pca_variance", "pca_variance")

        loadings = pd.DataFrame(pca.components_.T, index=vars_for_pca, columns=[f"PC{i+1}" for i in range(k)])
        loadings = loadings.reset_index().rename(columns={"index": "Variable"})
        self._save_via_converter(loadings, out_dir, "final_pca_loadings", "pca_loadings")

        # ----- Figuras -----
        import matplotlib.pyplot as plt

        # Scree
        fig = plt.figure(figsize=(7.2, 5.2))
        plt.plot(np.arange(1, k + 1), var_exp.values if hasattr(var_exp, "values") else var_exp, marker="o")
        plt.xlabel("PC"); plt.ylabel("Explained Var (%)")
        plt.title("Final Assembly — PCA Scree", fontsize=11)
        plt.grid(alpha=0.35)
        self._save_plot(fig, out_dir, "final_pca_scree.png")

        # Proyección 2D PC1–PC2 coloreada por is_attack (0/1)
        proj = pd.DataFrame(Xp[:, :2], columns=["PC1", "PC2"])
        proj["is_attack"] = pca_df["is_attack"].values
        fig = plt.figure(figsize=(7.6, 5.6))
        for val, marker, label, alpha in [(0, "o", "Normal", 0.55), (1, "x", "Attack", 0.6)]:
            s = proj[proj["is_attack"] == val]
            if s.empty:
                continue
            # Colores simples, sin estilos específicos
            plt.scatter(s["PC1"], s["PC2"], s=12, marker=marker, alpha=alpha, label=label)
        plt.title("Final Assembly — PCA 2D", fontsize=11)
        plt.grid(alpha=0.35); plt.legend()
        self._save_plot(fig, out_dir, "final_pca_2d.png")

        report["pca"]["notes"].append(f"PCA exportado en: {os.path.abspath(out_dir)}")

    # ---------------- Artefactos ----------------
    def _fit_artifacts(self, df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {"categorical_maps": {}, "drop_id_cols": [], "constant_cols": [],
                                     "duplicate_cols": [], "corr_dropped_pairs": [], "numeric_medians": {},
                                     "columns_order": []}

        id_cols = self._guess_id_cols(df)
        artifacts["drop_id_cols"] = id_cols
        df1 = df.drop(columns=id_cols, errors="ignore").copy()

        for col in self.CATEGORICAL_CANDIDATES:
            if col in df1.columns:
                artifacts["categorical_maps"][col] = self._one_hot_fit(df1, col)

        df2 = df1.copy()
        created = []
        for col, spec in artifacts["categorical_maps"].items():
            created += self._one_hot_apply(df2, spec)
        df2.drop(columns=[c for c in self.CATEGORICAL_CANDIDATES if c in df2.columns], inplace=True, errors="ignore")

        consts = self._remove_constant_cols(df2)
        artifacts["constant_cols"] = consts

        dups = self._remove_duplicate_cols(df2)
        artifacts["duplicate_cols"] = dups

        pairs = self._remove_collinear(df2)
        artifacts["corr_dropped_pairs"] = pairs

        meds = self._numeric_medians_fit(df2)
        artifacts["numeric_medians"] = meds

        cols = list(df2.columns)
        ordered = []
        if label_col in df.columns:
            ordered.append(label_col)
        if "is_attack" in df.columns and "is_attack" not in ordered:
            ordered.append("is_attack")
        ordered += [c for c in cols if c not in ordered]
        artifacts["columns_order"] = ordered
        return artifacts

    def _apply_artifacts(self, df: pd.DataFrame, label_col: str, art: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rep = {"dropped_ids": [], "created_dummies": [], "dropped_constants": [], "dropped_duplicates": [],
               "dropped_correlated": [], "imputed_numerics": [], "final_columns": []}

        drop_ids = [c for c in art.get("drop_id_cols", []) if c in df.columns]
        if drop_ids:
            df = df.drop(columns=drop_ids, errors="ignore")
        rep["dropped_ids"] = drop_ids

        dummy_cols = []
        for col, spec in art.get("categorical_maps", {}).items():
            if col in df.columns:
                dummy_cols += self._one_hot_apply(df, spec)
        for col in art.get("categorical_maps", {}).keys():
            if col in df.columns:
                df.drop(columns=[col], inplace=True, errors="ignore")
        rep["created_dummies"] = dummy_cols

        consts = [c for c in art.get("constant_cols", []) if c in df.columns]
        if consts:
            df.drop(columns=consts, inplace=True, errors="ignore")
        rep["dropped_constants"] = consts

        dups = [c for c in art.get("duplicate_cols", []) if c in df.columns]
        if dups:
            df.drop(columns=dups, inplace=True, errors="ignore")
        rep["dropped_duplicates"] = dups

        dropped_corr = []
        for keep, drop in art.get("corr_dropped_pairs", []):
            if drop in df.columns:
                df.drop(columns=[drop], inplace=True, errors="ignore")
                dropped_corr.append((keep, drop))
        rep["dropped_correlated"] = dropped_corr

        self._numeric_medians_apply(df, art.get("numeric_medians", {}))
        rep["imputed_numerics"] = sorted(list(art.get("numeric_medians", {}).keys()))

        final_cols = [c for c in art.get("columns_order", []) if c in df.columns]
        final_cols += [c for c in df.columns if c not in final_cols]
        df = df[final_cols]
        rep["final_columns"] = final_cols
        return df, rep

    # ---------------- API ----------------
    def run(self, df: pd.DataFrame, pca_out_dir: str | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ejecuta el ensamblado final. Si pca_out_dir es provisto, también corre PCA y exporta XLSX/PNG.
        """
        self._ensure_artifacts_dir()
        df = df.copy()
        before_rows = len(df)

        report: Dict[str, Any] = {
            "rows_before": before_rows,
            "rows_after": None,
            "dropped": 0,
            "label_normalization": {},
            "artifacts_mode": "apply" if os.path.exists(self.ARTIFACTS_FILE) else "fit",
            "assembly": {},
            "notes": [],
        }

        # 0) detectar/normalizar etiqueta y derivar binaria
        label_col = self._detect_label_col(df)
        df, lab_rep = self._normalize_label(df, label_col)
        report["label_normalization"] = lab_rep

        # 1) FIT o APPLY
        if not os.path.exists(self.ARTIFACTS_FILE):
            artifacts = self._fit_artifacts(df, label_col)
            with open(self.ARTIFACTS_FILE, "wb") as f:
                pickle.dump(artifacts, f)
            df2, ass = self._apply_artifacts(df, label_col, artifacts)
            report["assembly"] = ass
            report["notes"].append("Artefactos de ensamblado final entrenados y guardados.")
            df = df2
        else:
            with open(self.ARTIFACTS_FILE, "rb") as f:
                artifacts = pickle.load(f)
            df2, ass = self._apply_artifacts(df, label_col, artifacts)
            report["assembly"] = ass
            df = df2

        # 2) QA final: asegurar que no queden NaN/Inf en numéricas
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        for c in num_cols:
            s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if s.isna().any():
                med = float(s.median())
                df[c] = s.fillna(med)

        # 3) PCA opcional (estilo burst/intraflow)
        try:
            if pca_out_dir:
                self._ensure_outdir(pca_out_dir)
            self._pca_block(df, label_col, pca_out_dir, report)
        except Exception as e:
            report.setdefault("pca", {})
            report["pca"]["error"] = f"{type(e).__name__}: {e}"

        # 4) Final
        after_rows = len(df)
        report["rows_after"] = after_rows
        report["dropped"] = before_rows - after_rows
        report["notes"].extend([
            "Categóricas codificadas de forma estable (top-N + '__other__').",
            "IDs/leakage removidos; constantes/duplicadas/colineales depuradas.",
            "Imputación por mediana aplicada a numéricas restantes.",
            "Columnas ordenadas y QA final sin NaN/Inf.",
            "PCA disponible (si pca_out_dir fue provisto).",
        ])
        return df, report
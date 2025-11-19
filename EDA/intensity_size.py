# -*- coding: utf-8 -*-
"""
EDA - Intensidad y Tamaño de Flujo (RT-IoT2022)
------------------------------------------------
Analiza el volumen, ritmo, regularidad y estabilidad del tráfico IoT.
Incluye PCA para reducción exploratoria de dimensionalidad.
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------
# CONFIG
# --------------------------------------------

NUM_VARS = [
    "flow_pkts_payload.tot", "fwd_pkts_tot", "bwd_pkts_tot", "flow_duration",
    "payload_bytes_per_second", "flow_pkts_per_sec", "flow_pkts_payload.avg",
    "down_up_ratio", "flow_iat.avg", "flow_iat.std",
    "active.avg", "active.std", "idle.avg", "idle.std"
]

NORMAL_HINTS = {"normal", "benign", "mqtt_publish"}

# --------------------------------------------
# UTILS
# --------------------------------------------

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def norm_cat(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().replace({"": np.nan})

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def is_normal_label(x: str) -> bool:
    if not isinstance(x, str): return False
    return any(h in x.lower().strip() for h in NORMAL_HINTS)

def _find_converter() -> str | None:
    names = ["csv_to_xlsx_format.py", "csv_to_xlsx_format"]
    here = os.path.dirname(__file__)
    for c in [
        os.path.join(here, "csv_to_xlsx_format.py"),
        os.path.join(os.path.dirname(here), "csv_to_xlsx_format.py"),
        os.path.join(os.getcwd(), "csv_to_xlsx_format.py"),
    ]:
        if os.path.exists(c):
            return c
    return "csv_to_xlsx_format"

def _save_via_converter(df: pd.DataFrame, out_dir: str, base_name: str, sheet_name="Sheet1"):
    ensure_outdir(out_dir)
    converter = _find_converter()
    if converter is None:
        raise RuntimeError("No se encontró csv_to_xlsx_format.py")

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_csv = os.path.join(tmpd, f"{base_name}.csv")
        df.to_csv(tmp_csv, index=False)
        xlsx_out = os.path.join(out_dir, f"{base_name}.xlsx")
        cmd = [sys.executable, converter, tmp_csv, xlsx_out, "--sheet-name", sheet_name, "--outdir", out_dir]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"[XLSX] {xlsx_out}")
        return xlsx_out

def _save_plot(fig, out_dir: str, name: str):
    ensure_outdir(out_dir)
    out = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[PNG] {out}")

# --------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------

def main(df: pd.DataFrame, out_dir: str | None = None):
    out_dir = out_dir or "./EDA/Outputs_intensity_size"
    ensure_outdir(out_dir)
    print(f"\n📊 Analizando Intensidad y Tamaño de Flujo...")

    attack_col = "Attack_type" if "Attack_type" in df.columns else None
    if attack_col is None:
        raise KeyError("Columna Attack_type no encontrada.")
    att_norm = norm_cat(df[attack_col])
    df["traffic_class"] = att_norm.apply(lambda x: "normal" if is_normal_label(x) else ("attack" if isinstance(x, str) else "unknown"))

    # Convertir a numérico y filtrar columnas válidas
    vars_ok = [v for v in NUM_VARS if v in df.columns]
    if not vars_ok:
        raise KeyError("Ninguna variable esperada encontrada en el dataset.")

    df_num = df[vars_ok + ["traffic_class"]].copy()
    for v in vars_ok:
        df_num[v] = to_numeric_safe(df_num[v])

    # --------------------------
    # 1️⃣ Estadísticas descriptivas
    # --------------------------
    desc = df_num.groupby("traffic_class")[vars_ok].agg(["count", "mean", "median", "std", "min", "max"])
    desc.columns = ['_'.join(col).strip() for col in desc.columns.values]
    _save_via_converter(desc.reset_index(), out_dir, "eda_intensity_descriptivas", "intensity_descriptivas")

    # --------------------------
    # 2️⃣ Coeficiente de variación
    # --------------------------
    cv = df_num.groupby("traffic_class")[vars_ok].std() / df_num.groupby("traffic_class")[vars_ok].mean()
    cv = cv.T.rename_axis("variable").reset_index()
    _save_via_converter(cv, out_dir, "eda_intensity_cv", "coef_variacion")

    # --------------------------
    # 3️⃣ Correlaciones
    # --------------------------
    corr = df_num[vars_ok].corr(method="pearson")
    _save_via_converter(corr.reset_index(), out_dir, "eda_intensity_correlaciones", "correlaciones")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Figure 1. Correlation between flow intensity and size variables.")
    _save_plot(fig, out_dir, "eda_intensity_corr_heatmap.png")

    # --------------------------
    # 4️⃣ Outliers (IQR)
    # --------------------------
    outlier_summary = []
    for v in vars_ok:
        for cls in df_num["traffic_class"].unique():
            subset = df_num[df_num["traffic_class"] == cls][v].dropna()
            if len(subset) < 5: continue
            Q1, Q3 = subset.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outlier_pct = ((subset < lower) | (subset > upper)).mean() * 100
            outlier_summary.append({"variable": v, "class": cls, "outlier_%": round(outlier_pct, 2)})
    outlier_df = pd.DataFrame(outlier_summary)
    _save_via_converter(outlier_df, out_dir, "eda_intensity_outliers", "outliers")

    # --------------------------
    # 5️⃣ PCA (reducción exploratoria)
    # --------------------------
    print("⚙️  Ejecutando PCA para reducción de dimensionalidad...")

    df_pca = df_num.dropna(subset=vars_ok).copy()
    X = df_pca[vars_ok].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(len(vars_ok), 10), random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Varianza explicada
    var_exp = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(pca.n_components_)],
        "Varianza_Explicada": np.round(pca.explained_variance_ratio_ * 100, 2),
        "Varianza_Acumulada": np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 2)
    })
    _save_via_converter(var_exp, out_dir, "eda_intensity_pca_varianza", "pca_varianza")

    # Cargas (loadings)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=vars_ok
    ).reset_index().rename(columns={"index": "Variable"})
    _save_via_converter(loadings, out_dir, "eda_intensity_pca_loadings", "pca_loadings")

    # Scree plot
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(var_exp) + 1), var_exp["Varianza_Explicada"], marker="o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Figure 4. PCA Scree Plot – Explained Variance by Component")
    _save_plot(fig, out_dir, "eda_intensity_pca_scree.png")

    # Proyección 2D (PC1 vs PC2)
    pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
    pca_df["traffic_class"] = df_pca["traffic_class"].values

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=pca_df.sample(n=min(5000, len(pca_df)), random_state=42),
        x="PC1", y="PC2", hue="traffic_class", alpha=0.6,
        palette={"normal": "#4CAF50", "attack": "#F44336"}
    )
    ax.set_title("Figure 5. PCA 2D Projection (PC1 vs PC2) by traffic class")
    _save_plot(fig, out_dir, "eda_intensity_pca_2d.png")

    print(f"\nintensity_size listo. Resultados guardados en: {os.path.abspath(out_dir)}")

# EOF

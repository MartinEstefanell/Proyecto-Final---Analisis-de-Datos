# -*- coding: utf-8 -*-
"""
EDA — Burst / Intraflow Balance (RT-IoT2022)
--------------------------------------------
- Recibe: df y out_dir (no carga archivos).
- Deriva: subflow_balance_bytes, bulk_balance_bytes, bulk_rate_balance (log10 ratios).
- Exporta: descriptivas, CV, outliers (IQR) a XLSX; PCA (varianza y loadings) a XLSX.
- Figuras: SOLO boxplots (full y zoom P99) y PCA (scree + proyección 2D).
  → NO genera histogramas.

Requisitos:
- csv_to_xlsx_format.py disponible en la carpeta o en PATH (para convertir a .xlsx).
"""

import os, sys, tempfile, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------- Config --------------------
NORMAL_HINTS = {"normal", "benign", "mqtt_publish"}
MAX_POINTS_JITTER = 1500   # puntos máximos por clase para el overlay con jitter
PCA_MAX_COMPONENTS = 7

BOX_STYLE = dict(
    notch=True, showmeans=True, meanline=True, widths=0.55,
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    medianprops=dict(linewidth=2.2, color="black"),
    meanprops=dict(linewidth=1.5, linestyle="--", color="#F44336"),
    flierprops=dict(marker="o", markersize=2, alpha=0.15),
)

# -------------------- Utilidades --------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def norm_cat(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().replace({"": np.nan})

def is_normal_label(x: str) -> bool:
    if not isinstance(x, str):
        return False
    x = x.strip().lower()
    return any(h in x for h in NORMAL_HINTS)

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _find_converter() -> str | None:
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "csv_to_xlsx_format.py"),
        os.path.join(os.path.dirname(here), "csv_to_xlsx_format.py"),
        os.path.join(os.getcwd(), "csv_to_xlsx_format.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "csv_to_xlsx_format"  # en PATH

def _save_via_converter(df: pd.DataFrame, out_dir: str, base_name: str, sheet_name: str = "Sheet1") -> str:
    ensure_outdir(out_dir)
    converter = _find_converter()
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
    fig.tight_layout(pad=1.0)
    out = os.path.join(out_dir, name)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] {out}")

# -------------------- Derivación de variables --------------------
def _safe_get(df: pd.DataFrame, names: list[str]) -> pd.Series:
    """Devuelve la primera columna existente en 'names' convertida a numérico; si ninguna existe, retorna NaNs."""
    for n in names:
        if n in df.columns:
            return to_num(df[n])
    return pd.Series(np.nan, index=df.index)

def _build_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    subflow_balance_bytes = log10( (subflow_fwd_bytes + 1) / (subflow_bwd_bytes + 1) )
    bulk_balance_bytes    = log10( (fwd_avg_bytes_per_bulk + eps) / (bwd_avg_bytes_per_bulk + eps) )
    bulk_rate_balance     = log10( (fwd_avg_bulk_rate + eps)     / (bwd_avg_bulk_rate + eps) )
    """
    d = df.copy()
    eps = 1e-9

    # Subflow bytes
    sub_fwd = _safe_get(d, ["subflow_fwd_bytes", "Subflow Fwd Bytes", "subflow.fwd.bytes"])
    sub_bwd = _safe_get(d, ["subflow_bwd_bytes", "Subflow Bwd Bytes", "subflow.bwd.bytes"])
    d["subflow_balance_bytes"] = np.log10((sub_fwd.fillna(0) + 1.0) / (sub_bwd.fillna(0) + 1.0))

    # Bulk bytes promedio por ráfaga
    fwd_bulk_bytes = _safe_get(d, ["fwd_avg_bytes_per_bulk", "Fwd Avg Bytes/Bulk", "fwd_avg_bytes_bulk"])
    bwd_bulk_bytes = _safe_get(d, ["bwd_avg_bytes_per_bulk", "Bwd Avg Bytes/Bulk", "bwd_avg_bytes_bulk"])
    d["bulk_balance_bytes"] = np.log10((fwd_bulk_bytes.fillna(0) + eps) / (bwd_bulk_bytes.fillna(0) + eps))

    # Bulk rate promedio (ráfagas por segundo o similar)
    fwd_bulk_rate = _safe_get(d, ["fwd_avg_bulk_rate", "Fwd Avg Bulks Rate", "fwd_bulk_rate"])
    bwd_bulk_rate = _safe_get(d, ["bwd_avg_bulk_rate", "Bwd Avg Bulks Rate", "bwd_bulk_rate"])
    d["bulk_rate_balance"] = np.log10((fwd_bulk_rate.fillna(0) + eps) / (bwd_bulk_rate.fillna(0) + eps))

    # Clase: normal/attack/unknown
    d["Attack_type"] = norm_cat(d["Attack_type"])
    d["traffic_class"] = d["Attack_type"].apply(lambda x: "normal" if is_normal_label(x)
                                                else ("attack" if isinstance(x, str) else "unknown"))

    # Limpiar infinitos
    for c in ["subflow_balance_bytes","bulk_balance_bytes","bulk_rate_balance"]:
        d[c] = d[c].replace([np.inf, -np.inf], np.nan)

    return d

# -------------------- Figuras (boxplots + PCA) --------------------
def _box_balance_by_class(
    df: pd.DataFrame,
    var: str,
    class_col: str,
    out_dir: str,
    base_name: str,
    zoom_p99: bool = True,
    symmetric_zero: bool = True,
):
    """
    Boxplot por clase con:
      - baseline 0 (si symmetric_zero=True)
      - medianas anotadas
      - overlay de puntos (jitter) con muestreo
      - versión full y versión zoom P99
    """
    sub = df[[var, class_col]].dropna()
    if sub.empty:
        return

    def _draw_box(ax, data, labels):
        bp = ax.boxplot(data, labels=labels, **BOX_STYLE)
        # Baseline en 0 si corresponde (ratios log10)
        if symmetric_zero:
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.8)
        # Grid
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_ylabel(var)
        # Determinar el título según la variable
        if var == "bulk_rate_balance":
            ax.set_title("Figure P2. bulk_rate_balance by class", fontsize=11)
        else:
            ax.set_title(f"{var} by class", fontsize=11)

        # Anotar medianas y overlay de puntos con jitter
        for ix, s in enumerate(data, start=1):
            if len(s) == 0:
                continue
            med = float(np.median(s))
            ax.text(ix - 0.18, med, f"median≈{med:.3f}", color="#F44336", fontsize=9, va="bottom")

            # Overlay (jitter + muestreo para no sobrecargar)
            show = s
            if len(show) > MAX_POINTS_JITTER:
                show = np.random.default_rng(42).choice(show, size=MAX_POINTS_JITTER, replace=False)
            x_j = np.random.default_rng(123).normal(loc=ix, scale=0.04, size=len(show))
            ax.scatter(x_j, show, s=6, alpha=0.20, edgecolors="none")

    # Datos por clase
    data, labels = [], []
    for cls, label in [("normal","Normal"), ("attack","Attack"), ("unknown","Unknown")]:
        vals = sub[sub[class_col]==cls][var].to_numpy()
        if len(vals) == 0: 
            continue
        data.append(vals); labels.append(label)

    if not data:
        return

    # ---- FULL ----
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    _draw_box(ax, data, labels)

    # Límites: simétricos alrededor de 0 si aplica
    all_vals = np.concatenate(data)
    if symmetric_zero:
        mx = np.nanpercentile(np.abs(all_vals), 99)
        ax.set_ylim(-mx*1.05, mx*1.05)
    else:
        lo, hi = np.nanpercentile(all_vals, [1, 99])
        pad = (hi - lo) * 0.08 + 1e-12
        ax.set_ylim(lo - pad, hi + pad)

    _save_plot(fig, out_dir, f"{base_name}.png")

    # ---- ZOOM P99 ----
    if zoom_p99:
        fig, ax = plt.subplots(figsize=(7.5, 5.6))
        _draw_box(ax, data, labels)

        if symmetric_zero:
            mx = np.nanpercentile(np.abs(all_vals), 99)
            ax.set_ylim(-mx*1.05, mx*1.05)
        else:
            lo, hi = np.nanpercentile(all_vals, [1, 99])
            pad = (hi - lo) * 0.05 + 1e-12
            ax.set_ylim(lo - pad, hi + pad)

        _save_plot(fig, out_dir, f"{base_name}_(zoom_P99).png")

def _pca_block(df: pd.DataFrame, vars_for_pca: list[str], out_dir: str):
    pca_df = df[vars_for_pca + ["traffic_class"]].dropna()
    if len(pca_df) < 50 or len(vars_for_pca) < 2:
        _save_via_converter(pd.DataFrame([{"note":"PCA skipped (insufficient rows/vars)"}]),
                            out_dir, "burst_pca_info", "pca_info")
        return

    X = pca_df[vars_for_pca].values
    Xs = StandardScaler().fit_transform(X)
    k = min(PCA_MAX_COMPONENTS, len(vars_for_pca))
    pca = PCA(n_components=k, random_state=42)
    Xp = pca.fit_transform(Xs)

    # Varianza explicada
    var_exp = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(k)],
        "Explained_Variance_%": np.round(pca.explained_variance_ratio_*100, 2),
        "Cumulative_%": np.round(np.cumsum(pca.explained_variance_ratio_)*100, 2)
    })
    _save_via_converter(var_exp, out_dir, "burst_pca_variance", "pca_variance")

    # Loadings
    loadings = pd.DataFrame(pca.components_.T, index=vars_for_pca, columns=[f"PC{i+1}" for i in range(k)])
    loadings = loadings.reset_index().rename(columns={"index":"Variable"})
    _save_via_converter(loadings, out_dir, "burst_pca_loadings", "pca_loadings")

    # Scree
    fig = plt.figure(figsize=(7.2, 5.2))
    plt.plot(np.arange(1, k+1), var_exp["Explained_Variance_%"].values, marker="o")
    plt.xlabel("PC"); plt.ylabel("Explained Var (%)")
    plt.title("Figure P1. Burst/Intraflow PCA — Scree", fontsize=11)
    plt.grid(alpha=0.35)
    _save_plot(fig, out_dir, "burst_pca_scree.png")

    # Proyección 2D
    proj = pd.DataFrame(Xp[:, :2], columns=["PC1","PC2"])
    proj["traffic_class"] = pca_df["traffic_class"].values
    fig = plt.figure(figsize=(7.6, 5.6))
    color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
    for cls, marker, label, alpha in [("normal","o","Normal",0.55),
                                      ("attack","x","Attack",0.6),
                                      ("unknown",".","Unknown",0.35)]:
        s = proj[proj["traffic_class"]==cls]
        if s.empty: continue
        plt.scatter(s["PC1"], s["PC2"], s=12, marker=marker, alpha=alpha, label=label, color=color_map.get(cls))
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Burst/Intraflow PCA — 2D", fontsize=11)
    plt.grid(alpha=0.35); plt.legend()
    _save_plot(fig, out_dir, "burst_pca_2d.png")

# -------------------- Métricas tabulares --------------------
def _export_descriptives_cv_outliers(df: pd.DataFrame, vars_list: list[str], out_dir: str):
    # Descriptivas por clase
    desc = df.groupby("traffic_class")[vars_list].agg(["count","mean","median","std","min","max"])
    desc.columns = ["_".join(col).strip() for col in desc.columns.values]
    _save_via_converter(desc.reset_index(), out_dir, "burst_descriptivas", "descriptivas")

    # CV y outliers (IQR) por clase
    cv_rows, out_rows = [], []
    for v in vars_list:
        for cls in df["traffic_class"].unique():
            s = df.loc[df["traffic_class"]==cls, v].replace([np.inf,-np.inf], np.nan).dropna()
            if len(s) >= 5:
                cv = float(np.std(s) / (np.mean(s) + 1e-12))
                cv_rows.append({"variable": v, "class": cls, "n": int(len(s)), "cv": round(cv, 4)})
                q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
                low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
                pct = ((s < low) | (s > high)).mean() * 100.0
                out_rows.append({"variable": v, "class": cls, "n": int(len(s)), "iqr_outliers_%": round(pct, 2)})
            else:
                cv_rows.append({"variable": v, "class": cls, "n": int(len(s)), "cv": np.nan})
                out_rows.append({"variable": v, "class": cls, "n": int(len(s)), "iqr_outliers_%": np.nan})
    _save_via_converter(pd.DataFrame(cv_rows), out_dir, "burst_cv", "cv")
    _save_via_converter(pd.DataFrame(out_rows), out_dir, "burst_outliers", "outliers")

# -------------------- MAIN --------------------
def main(df: pd.DataFrame, out_dir: str | None = None):
    out_dir = out_dir or "Outputs_burst_balance"
    ensure_outdir(out_dir)
    print("\n📦 Analizando Burst / Intraflow balance…")

    # Derivar
    d = _build_derived(df)

    # Variables efectivas (las que realmente existen)
    vars_ok = [v for v in ["subflow_balance_bytes","bulk_balance_bytes","bulk_rate_balance"] if v in d.columns and d[v].notna().any()]
    if not vars_ok:
        raise RuntimeError("No se pudieron derivar variables de burst/balance (faltan columnas base).")

    # Exportar métricas tabulares
    _export_descriptives_cv_outliers(d, vars_ok, out_dir)

    # Boxplots (full + zoom) — baseline 0 y estilo mejorado

    zero_centered = {"subflow_balance_bytes", "bulk_balance_bytes", "bulk_rate_balance"}
    for v in vars_ok:
        _box_balance_by_class(
            d, v, "traffic_class", out_dir,
            base_name=f"burst_box_{v}",
            zoom_p99=True,
            symmetric_zero=(v in zero_centered)
        )

    # PCA
    _pca_block(d, vars_ok, out_dir)

    print(f"\n burst/intraflow listo. Salidas en: {os.path.abspath(out_dir)}")


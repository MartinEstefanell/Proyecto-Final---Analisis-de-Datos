# -*- coding: utf-8 -*-
"""
EDA - Burst/Intraflow (variables originales)
--------------------------------------------
- NO deriva nuevas variables.
- Trabaja con pares originales fwd/bwd (bytes y rates) para 'bulk' y 'subflow'.
- Produce: descriptivas, CV, outliers, correlaciones, boxplots (full + zoom P99),
           scatter fwd vs bwd (log-log si aplica), y PCA (PC1, PC2, ...).

Uso:
    from burstiness_originals import main
    main(df, out_dir="Outputs_burstiness_originals")
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------- Config --------------------
NORMAL_HINTS = {"normal", "benign", "mqtt_publish"}
MAX_POINTS_SCATTER = 12000
TOP_CORR_LABEL = "Originals correlations"

ORIG_PAIRS = [
    ("bulk_fwd_bytes", "bulk_bwd_bytes"),
    ("bulk_fwd_rate",  "bulk_bwd_rate"),
    ("subflow_fwd_bytes", "subflow_bwd_bytes"),
    ("subflow_fwd_rate",  "subflow_bwd_rate")
]

# -------------------- Utils --------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def norm_cat(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().replace({"": np.nan})

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def is_normal_label(x: str) -> bool:
    if not isinstance(x, str):
        return False
    x = x.strip().lower()
    return any(h in x for h in NORMAL_HINTS)

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
    return None  # intentaremos fallback a pandas.ExcelWriter

def _save_via_converter(df: pd.DataFrame, out_dir: str, base_name: str, sheet_name: str = "Sheet1") -> str:
    """Guarda a .xlsx con el conversor si existe; si no, usa pandas ExcelWriter."""
    ensure_outdir(out_dir)
    xlsx_out = os.path.join(out_dir, f"{base_name}.xlsx")
    converter = _find_converter()
    if converter:
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_csv = os.path.join(tmpd, f"{base_name}.csv")
            df.to_csv(tmp_csv, index=False)
            cmd = [sys.executable, converter, tmp_csv, xlsx_out, "--sheet-name", sheet_name, "--outdir", out_dir]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        # Fallback
        with pd.ExcelWriter(xlsx_out, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    print(f"[XLSX] {xlsx_out}")
    return xlsx_out

def _save_plot(fig, out_dir: str, name: str):
    ensure_outdir(out_dir)
    out = os.path.join(out_dir, name)
    fig.tight_layout(pad=1.0)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] {out}")

def _nice_y_limits(values: np.ndarray) -> tuple[float, float]:
    """Evita 'singular transform' cuando todo es (casi) cero."""
    if values.size == 0:
        return (-1.0, 1.0)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (-1.0, 1.0)
    if np.isclose(vmin, vmax):
        eps = 1e-3 if vmax == 0 else abs(vmax) * 1e-2
        return (vmin - eps, vmax + eps)
    pad = 0.05 * (vmax - vmin)
    return (vmin - pad, vmax + pad)

# -------------------- Plots --------------------
def _box_by_class(df, var, class_col, out_dir, base_name, zoom_p99=False):
    sub = df[[var, class_col]].dropna()
    if sub.empty:
        return

    def _plot(data_full: pd.DataFrame, suffix: str):
        fig = plt.figure(figsize=(8.5, 6.2))
        data, labels = [], []
        for cls, label in [("normal","Normal"), ("attack","Attack"), ("unknown","Unknown")]:
            s = data_full[data_full[class_col]==cls][var].to_numpy()
            if len(s) == 0: 
                continue
            data.append(s); labels.append(label)

        if not data:
            plt.close(fig); return

        bp = plt.boxplot(
            data, labels=labels, showmeans=True, meanline=True, widths=0.55, notch=False,
            flierprops={"markersize":3, "alpha":0.25}
        )

        # Medianas como texto
        for i, s in enumerate(data, 1):
            if len(s) == 0: continue
            med = np.nanmedian(s)
            plt.text(i, med, f"median≈{med:.3f}", ha="center", va="bottom", fontsize=9, color="#F44336")

        # Eje log si todo es positivo y hay cola; si no, lineal con límites robustos
        all_pos = all((np.nanmin(s) >= 0 for s in data))
        spread_ok = any((np.nanmax(s) / (np.nanmin(s) + 1e-12) > 50 for s in data if np.nanmin(s) > 0))
        if all_pos and spread_ok:
            plt.yscale("symlog", linthresh=1e-3)
        yvals = np.concatenate([np.asarray(s) for s in data])
        lo, hi = _nice_y_limits(yvals)
        plt.ylim(lo, hi)

        plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        plt.ylabel(var)
        title = f"{var} by class" if not suffix else f"{var} by class {suffix}"
        plt.title(title, fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        _save_plot(fig, out_dir, f"{base_name}{suffix.replace(' ','_')}.png")

    # full
    _plot(sub, suffix="")

    # zoom P99 por clase
    if zoom_p99:
        zrows = []
        for cls in sub[class_col].unique():
            s = sub[sub[class_col]==cls][var].to_numpy()
            if len(s) == 0: 
                continue
            lo, hi = np.percentile(s, [1, 99])
            z = np.clip(s, lo, hi)
            zrows.extend([(v, cls) for v in z])
        zdf = pd.DataFrame(zrows, columns=[var, class_col]) if zrows else pd.DataFrame(columns=[var, class_col])
        if not zdf.empty:
            _plot(zdf, suffix="(zoom P99)")

def _scatter_fwd_vs_bwd(df, xvar, yvar, class_col, out_dir, base_name):
    sub = df[[xvar, yvar, class_col]].dropna()
    if sub.empty:
        return
    if len(sub) > MAX_POINTS_SCATTER:
        sub = sub.sample(MAX_POINTS_SCATTER, random_state=42)

    fig = plt.figure(figsize=(7.8, 6.0))
    for cls, marker, label, alpha in [
        ("normal","o","Normal",0.50),
        ("attack","x","Attack",0.65),
        ("unknown",".","Unknown",0.35),
    ]:
        s = sub[sub[class_col]==cls]
        if s.empty: 
            continue
        plt.scatter(s[xvar], s[yvar], s=10, marker=marker, alpha=alpha, label=label)

    # Diagonal
    all_vals = sub[[xvar, yvar]].values.flatten()
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        lo = min(lo, 1e-9)
        xs = np.linspace(lo, hi, 100)
        plt.plot(xs, xs, linestyle="--", color="gray", alpha=0.6, linewidth=1)

    # Eje log si procede
    minx, miny = sub[xvar].min(), sub[yvar].min()
    if minx > 0 and miny > 0:
        plt.xscale("log"); plt.yscale("log")

    plt.xlabel(xvar); plt.ylabel(yvar)
    plt.title(f"{xvar} vs {yvar}", fontsize=12)
    plt.grid(alpha=0.3); plt.legend()
    _save_plot(fig, out_dir, f"{base_name}.png")

def _corr_heatmap(mat: pd.DataFrame, title: str, out_dir: str, fname: str):
    if mat.empty: return
    fig_w = max(6, min(1 + 0.35 * mat.shape[1], 18))
    fig_h = max(6, min(1 + 0.35 * mat.shape[0], 18))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(mat.shape[1])); ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(mat.columns.astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mat.index.astype(str), fontsize=8)
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(im, ax=ax); cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    # anotar
    if mat.shape[0] <= 30 and mat.shape[1] <= 30:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    _save_plot(fig, out_dir, fname)

# -------------------- MAIN --------------------
def main(df: pd.DataFrame, out_dir: str | None = None):
    out_dir = out_dir or "Outputs_burstiness_originals"
    ensure_outdir(out_dir)
    print("\n📦 Analizando Burst/Intraflow (variables originales)…")

    # Detectar columnas disponibles
    colset = set(df.columns)
    pairs = [(x, y) for (x, y) in ORIG_PAIRS if x in colset and y in colset]
    if not pairs:
        print("⚠️  No se encontraron columnas originales esperadas.")
        return

    # Casting seguro
    cast_cols = sorted({c for p in pairs for c in p})
    d0 = df.copy()
    for c in cast_cols:
        d0[c] = to_numeric_safe(d0[c])

    # Clase normal/attack/unknown
    if "Attack_type" in d0.columns:
        d0["Attack_type"] = norm_cat(d0["Attack_type"])
        d0["traffic_class"] = d0["Attack_type"].apply(lambda x: "normal" if is_normal_label(x) else ("attack" if isinstance(x, str) else "unknown"))
    else:
        d0["traffic_class"] = "unknown"

    # Calidad básica
    qrows = []
    for c in cast_cols:
        qrows.append({"metric": f"nulls_in_{c}", "value": int(d0[c].isna().sum())})
    _save_via_converter(pd.DataFrame(qrows), out_dir, "burst_orig_quality_checks", "quality_checks")

    # Descriptivas / CV / Outliers por clase
    desc_df_rows, cv_rows, out_rows = [], [], []
    classes = d0["traffic_class"].unique()

    for cls in classes:
        sub = d0[d0["traffic_class"] == cls]
        if sub.empty: 
            continue
        desc = sub[cast_cols].agg(["count","mean","median","std","min","max"]).T.reset_index().rename(columns={"index":"variable"})
        desc.insert(0, "class", cls)
        desc_df_rows.append(desc)

        for v in cast_cols:
            s = sub[v].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) >= 5:
                cv = float(np.std(s) / (np.mean(s) + 1e-12))
                q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
                lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
                pct = ((s < lo) | (s > hi)).mean() * 100.0
            else:
                cv, pct = np.nan, np.nan
            cv_rows.append({"variable": v, "class": cls, "n": int(len(s)), "cv": round(cv, 4)})
            out_rows.append({"variable": v, "class": cls, "n": int(len(s)), "iqr_outliers_%": (None if np.isnan(pct) else round(pct, 2))})

    if desc_df_rows:
        _save_via_converter(pd.concat(desc_df_rows, ignore_index=True), out_dir, "burst_orig_descriptivas", "descriptivas")
    if cv_rows:
        _save_via_converter(pd.DataFrame(cv_rows), out_dir, "burst_orig_cv", "cv")
    if out_rows:
        _save_via_converter(pd.DataFrame(out_rows), out_dir, "burst_orig_outliers", "outliers")

    # Boxplots por clase (full + zoom P99)
    for v in cast_cols:
        _box_by_class(d0, v, "traffic_class", out_dir, f"burst_orig_box_{v}", zoom_p99=True)

    # Scatters fwd vs bwd
    for xvar, yvar in pairs:
        _scatter_fwd_vs_bwd(d0, xvar, yvar, "traffic_class", out_dir, f"burst_orig_scatter_{xvar}_vs_{yvar}")

    # Correlaciones (todas las originales disponibles)
    corr_mat = d0[cast_cols].corr(numeric_only=True)
    _save_via_converter(corr_mat.reset_index(), out_dir, "burst_orig_correlaciones", "correlaciones")
    _corr_heatmap(corr_mat, TOP_CORR_LABEL, out_dir, "burst_orig_corr_heatmap.png")

    # PCA sobre originales disponibles (requiere >=3 variables y >=50 filas válidas)
    pca_vars = cast_cols.copy()
    pca_df = d0[pca_vars + ["traffic_class"]].dropna()
    if len(pca_df) >= 50 and len(pca_vars) >= 3:
        X = pca_df[pca_vars].values
        Xs = StandardScaler().fit_transform(X)
        k = min(len(pca_vars), 8)
        pca = PCA(n_components=k, random_state=42)
        Xp = pca.fit_transform(Xs)

        var_exp = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(k)],
            "Explained_Variance_%": np.round(pca.explained_variance_ratio_*100, 2),
            "Cumulative_%": np.round(np.cumsum(pca.explained_variance_ratio_)*100, 2),
        })
        _save_via_converter(var_exp, out_dir, "burst_orig_pca_variance", "pca_variance")

        loadings = pd.DataFrame(pca.components_.T, index=pca_vars, columns=[f"PC{i+1}" for i in range(k)]).reset_index().rename(columns={"index":"Variable"})
        _save_via_converter(loadings, out_dir, "burst_orig_pca_loadings", "pca_loadings")

        # Scree
        fig = plt.figure(figsize=(7.2, 5.2))
        plt.plot(np.arange(1, k+1), var_exp["Explained_Variance_%"].values, marker="o")
        plt.xlabel("PC"); plt.ylabel("Explained Var (%)")
        plt.title("Burst/Intraflow Originals — Scree", fontsize=12)
        plt.grid(alpha=0.3)
        _save_plot(fig, out_dir, "burst_orig_pca_scree.png")

        # 2D (PC1 vs PC2)
        proj = pd.DataFrame(Xp[:, :2], columns=["PC1","PC2"])
        proj["traffic_class"] = pca_df["traffic_class"].values
        if len(proj) > MAX_POINTS_SCATTER:
            proj = proj.sample(MAX_POINTS_SCATTER, random_state=42)
        fig = plt.figure(figsize=(7.8, 6.0))
        color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
        for cls, marker, label, alpha in [("normal","o","Normal",0.55),
                                          ("attack","x","Attack",0.6),
                                          ("unknown",".","Unknown",0.35)]:
            s = proj[proj["traffic_class"]==cls]
            if s.empty: continue
            plt.scatter(s["PC1"], s["PC2"], s=12, marker=marker, alpha=alpha, label=label, color=color_map.get(cls))
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title("Burst/Intraflow Originals — PCA 2D (PC1 vs PC2)", fontsize=12)
        plt.grid(alpha=0.3); plt.legend()
        _save_plot(fig, out_dir, "burst_orig_pca_2d.png")
    else:
        _save_via_converter(pd.DataFrame([{"note": "PCA skipped (insufficient rows/vars)"}]), out_dir, "burst_orig_pca_info", "pca_info")

    print(f"\n burst/originals listo. Salidas en: {os.path.abspath(out_dir)}")

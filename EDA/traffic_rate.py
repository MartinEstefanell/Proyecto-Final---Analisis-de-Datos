# -*- coding: utf-8 -*-
"""
EDA - Ritmo de Tráfico (RT-IoT2022)
-----------------------------------
Analiza la dinámica temporal del flujo:
- Tasas: flow_byts_s, flow_pkts_s (+ opcional payload_bytes_per_second)
- Intervalos: flow_iat.min/max/tot, fwd_iat.avg, bwd_iat.avg
- Períodos: active.min/max, idle.min/max
Exporta XLSX (vía csv_to_xlsx_format.py) y gráficos PNG (matplotlib).
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------- Config --------------------
NORMAL_HINTS = {"normal", "benign", "mqtt_publish"}

RATE_VARS = ["flow_byts_s", "flow_pkts_s", "payload_bytes_per_second"]  # la última es opcional
IAT_VARS  = ["flow_iat.min", "flow_iat.max", "flow_iat.tot", "fwd_iat.avg", "bwd_iat.avg"]
PERIOD_VARS = ["active.min", "active.max", "idle.min", "idle.max"]

ALL_VARS = RATE_VARS + IAT_VARS + PERIOD_VARS

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
    return "csv_to_xlsx_format"  # en PATH

def _save_via_converter(df: pd.DataFrame, out_dir: str, base_name: str, sheet_name: str = "Sheet1") -> str:
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
    fig.tight_layout(pad=1.0)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] {out}")

# -------------------- Plots (matplotlib puro) --------------------
def _hist_log_by_class(df, var, class_col, title, fname, bins=60):
    sub = df[[var, class_col]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[sub[var] > 0]
    if sub.empty: return
    color_map = {"normal": "#43A047", "attack": "#F44336"}
    fig = plt.figure(figsize=(7.5, 5.5))
    for lab, name, alpha in [("normal", "Normal", 0.6), ("attack", "Attack", 0.6)]:
        vals = sub.loc[sub[class_col] == lab, var]
        vals = np.log10(vals.replace(0, np.nan).dropna())
        if not vals.empty:
            plt.hist(vals, bins=bins, alpha=alpha, label=name, color=color_map.get(lab))
    plt.xlabel(f"log10({var})")
    plt.ylabel("Count")
    plt.title(title, fontsize=11)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    _save_plot(fig, out_dir=fname.rsplit(os.sep, 1)[0], name=fname.rsplit(os.sep, 1)[1])

def _box_log_by_class(df, var, class_col, title, fname, clip_p99=False):
    sub = df[[var, class_col]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[sub[var] > 0]
    if sub.empty: return

    def _plot(d, suffix, zoom=False):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        data, labels = [], []
        for lab, name in [("normal", "Normal"), ("attack", "Attack")]:
            s = d[d[class_col] == lab][var].to_numpy()
            if len(s) == 0: 
                continue
            if zoom:
                p99 = np.percentile(s, 99)
                s = np.clip(s, a_min=np.min(s), a_max=p99)
            data.append(s); labels.append(name)

        if not data: 
            plt.close(fig); return

        ax.boxplot(data, labels=labels, showfliers=True,
                   flierprops={"markersize":2, "alpha":0.25},
                   notch=True, widths=0.55, showmeans=True, meanline=True)
        ax.set_yscale("log")
        ax.set_ylabel(f"{var} (log scale)")
        ttl = title + (" (zoom P99)" if zoom else "")
        ax.set_title(ttl, fontsize=11)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        ax.grid(axis="y", which="both", linestyle="--", alpha=0.35)
        _save_plot(fig, out_dir=fname.rsplit(os.sep, 1)[0], name=fname.rsplit(os.sep, 1)[1].replace(".png", "_p99.png" if zoom else ".png"))

    _plot(sub, "full", zoom=False)
    if clip_p99:
        _plot(sub, "p99", zoom=True)

def _scatter_loglog(df, xvar, yvar, class_col, title, fname):
    sub = df[[xvar, yvar, class_col]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[(sub[xvar] > 0) & (sub[yvar] > 0)]
    if sub.empty: return
    if len(sub) > 8000:
        sub = sub.sample(8000, random_state=42)

    color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
    fig = plt.figure(figsize=(7.5, 5.5))
    for cls, marker, name, alpha in [("normal", "o", "Normal", 0.5),
                                     ("attack", "x", "Attack", 0.6),
                                     ("unknown", ".", "Unknown", 0.3)]:
        s = sub[sub[class_col] == cls]
        if not s.empty:
            plt.scatter(s[xvar], s[yvar], s=10, marker=marker, alpha=alpha, label=name, color=color_map.get(cls))
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(xvar); plt.ylabel(yvar)
    plt.title(title, fontsize=11)
    plt.legend()
    plt.grid(which="both", linestyle="--", alpha=0.3)
    _save_plot(fig, out_dir=fname.rsplit(os.sep, 1)[0], name=fname.rsplit(os.sep, 1)[1])

def _corr_heatmap(mat: pd.DataFrame, title: str, out_path: str):
    if mat.empty: return
    fig_w = max(6, min(1 + 0.35 * mat.shape[1], 18))
    fig_h = max(6, min(1 + 0.35 * mat.shape[0], 18))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(mat.shape[1])); ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(mat.columns.astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mat.index.astype(str), fontsize=8)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax); cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    if mat.shape[0] <= 30 and mat.shape[1] <= 30:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    out_dir, name = out_path.rsplit(os.sep, 1)
    _save_plot(fig, out_dir, name)

def _bar_outliers_pct(out_df: pd.DataFrame, out_dir: str, fname: str, top=12):
    """Bar chart de % outliers por variable y clase (top por media de %)."""
    dfp = out_df.dropna(subset=["iqr_outliers_%"]).copy()
    if dfp.empty: return
    rank = dfp.groupby("variable")["iqr_outliers_%"].mean().sort_values(ascending=False).head(top).index
    piv = dfp[dfp["variable"].isin(rank)].pivot(index="variable", columns="class", values="iqr_outliers_%").fillna(0.0)
    piv = piv.loc[rank]  # mantener orden
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    x = np.arange(len(piv.index))
    w = 0.35
    nrm = piv["normal"].values if "normal" in piv.columns else np.zeros(len(x))
    atk = piv["attack"].values if "attack" in piv.columns else np.zeros(len(x))
    # Colors: normal -> green, attack -> red
    ax.bar(x - w/2, nrm, width=w, label="Normal", color="#43A047")
    ax.bar(x + w/2, atk, width=w, label="Attack", color="#F44336")
    ax.set_xticks(x); ax.set_xticklabels(piv.index.astype(str), rotation=45, ha="right")
    ax.set_ylabel("% IQR Outliers")
    ax.set_title("Figure R9. Top variables by outlier rate (IQR) – Normal vs Attack", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    _save_plot(fig, out_dir, fname)

# -------------------- MAIN --------------------
def main(df: pd.DataFrame, out_dir: str | None = None):
    out_dir = out_dir or "Outputs_traffic_rate"
    ensure_outdir(out_dir)
    print("\n📊 Analizando Ritmo de Tráfico...")

    # Etiqueta de clase
    if "Attack_type" not in df.columns:
        raise KeyError("Columna Attack_type no encontrada.")
    df = df.copy()
    df["Attack_type"] = norm_cat(df["Attack_type"])
    df["traffic_class"] = df["Attack_type"].apply(lambda x: "normal" if is_normal_label(x) else ("attack" if isinstance(x, str) else "unknown"))

    # Filtrar columnas existentes
    vars_ok = [v for v in ALL_VARS if v in df.columns]
    if not vars_ok:
        raise KeyError("No se encontraron variables de ritmo esperadas en el dataset.")

    # Casting numérico
    for v in vars_ok:
        df[v] = to_numeric_safe(df[v])

    # ---------------- 1) Descriptivas por clase ----------------
    desc = df.groupby("traffic_class")[vars_ok].agg(["count", "mean", "median", "std", "min", "max"])
    desc.columns = ["_".join(col).strip() for col in desc.columns.values]
    _save_via_converter(desc.reset_index(), out_dir, "eda_rate_descriptivas", "rate_descriptivas")

    # ---------------- 2) CV por clase ----------------
    gstd = df.groupby("traffic_class")[vars_ok].std()
    gmean = df.groupby("traffic_class")[vars_ok].mean().replace(0, np.nan)
    cv = (gstd / gmean).T.rename_axis("variable").reset_index()
    _save_via_converter(cv, out_dir, "eda_rate_cv", "coef_variacion")

    # ---------------- 3) Outliers (IQR) por clase ----------------
    out_rows = []
    for v in vars_ok:
        for cls in df["traffic_class"].unique():
            s = df.loc[df["traffic_class"] == cls, v].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 5:
                out_rows.append({"variable": v, "class": cls, "n": int(len(s)), "iqr_outliers_%": np.nan})
                continue
            q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            pct = ((s < low) | (s > high)).mean() * 100.0
            out_rows.append({"variable": v, "class": cls, "n": int(len(s)), "iqr_outliers_%": round(pct, 2)})
    out_df = pd.DataFrame(out_rows)
    _save_via_converter(out_df, out_dir, "eda_rate_outliers", "outliers")

    # ----- Gráfico de barras (Top variables por % outliers) -----
    _bar_outliers_pct(out_df, out_dir, "eda_rate_outliers_bar.png", top=12)

    # ---------------- 4) Correlaciones ----------------
    corr_vars = [v for v in vars_ok if v != "payload_bytes_per_second" or "payload_bytes_per_second" in df.columns]
    corr_mat = df[corr_vars].corr(numeric_only=True)
    _save_via_converter(corr_mat.reset_index(), out_dir, "eda_rate_correlaciones", "correlaciones")
    _corr_heatmap(corr_mat, "Figure P2. Correlation matrix – Traffic rate & temporal metrics",
                  os.path.join(out_dir, "eda_rate_corr_heatmap.png"))

    # ---------------- 5) Gráficos clave ----------------
    # 1.a/1.b: Histogramas + Boxplots log de tasas
    if "flow_byts_s" in df.columns:
        _hist_log_by_class(df, "flow_byts_s", "traffic_class",
                           "Figure R1. Distribution of flow_byts_s (log10)",
                           os.path.join(out_dir, "eda_rate_hist_flow_byts_s.png"))
        _box_log_by_class(df, "flow_byts_s", "traffic_class",
                          "Figure R2. flow_byts_s by class (log scale)",
                          os.path.join(out_dir, "eda_rate_box_flow_byts_s.png"), clip_p99=True)

    if "flow_pkts_s" in df.columns:
        _hist_log_by_class(df, "flow_pkts_s", "traffic_class",
                           "Figure R3. Distribution of flow_pkts_s (log10)",
                           os.path.join(out_dir, "eda_rate_hist_flow_pkts_s.png"))
        _box_log_by_class(df, "flow_pkts_s", "traffic_class",
                          "Figure R5. flow_pkts_s by class (log scale)",
                          os.path.join(out_dir, "eda_rate_box_flow_pkts_s.png"), clip_p99=True)

    # 3.a: Ritmo direccional (intervalos promedio por dirección)
    if "fwd_iat.avg" in df.columns and "bwd_iat.avg" in df.columns:
        _scatter_loglog(df, "fwd_iat.avg", "bwd_iat.avg", "traffic_class",
                        "Figure P3. fwd_iat.avg vs bwd_iat.avg (log-log)",
                        os.path.join(out_dir, "eda_rate_scatter_fwd_vs_bwd_iat_avg.png"))
        # Boxplots complementarios fwd vs bwd por clase (con P99)
        # Normal
        for cls_lab in ["normal", "attack"]:
            sub = df[df["traffic_class"] == cls_lab][["fwd_iat.avg", "bwd_iat.avg"]].replace([np.inf, -np.inf], np.nan).dropna()
            sub = sub[(sub["fwd_iat.avg"] > 0) & (sub["bwd_iat.avg"] > 0)]
            if sub.empty: 
                continue
            # Full
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            ax.boxplot([sub["fwd_iat.avg"].values, sub["bwd_iat.avg"].values],
                       labels=["fwd_iat.avg", "bwd_iat.avg"],
                       showfliers=True, flierprops={"markersize":2, "alpha":0.25},
                       notch=False, widths=0.55, showmeans=True, meanline=True)
            ax.set_yscale("log"); ax.set_ylabel("Seconds (log scale)")
            ax.set_title(f"Figure P4. fwd_iat.avg vs bwd_iat.avg — {cls_lab.capitalize()}", fontsize=11)
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1))
            ax.yaxis.set_major_formatter(LogFormatterSciNotation())
            ax.grid(axis="y", which="both", linestyle="--", alpha=0.35)
            _save_plot(fig, out_dir, f"eda_rate_box_iat_avg_{cls_lab}.png")
            # Zoom P99
            p99_f, p99_b = np.percentile(sub["fwd_iat.avg"], 99), np.percentile(sub["bwd_iat.avg"], 99)
            f_clip = np.clip(sub["fwd_iat.avg"].values, a_min=sub["fwd_iat.avg"].min(), a_max=p99_f)
            b_clip = np.clip(sub["bwd_iat.avg"].values, a_min=sub["bwd_iat.avg"].min(), a_max=p99_b)
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            ax.boxplot([f_clip, b_clip],
                       labels=["fwd_iat.avg", "bwd_iat.avg"],
                       showfliers=True, flierprops={"markersize":2, "alpha":0.25},
                       notch=False, widths=0.55, showmeans=True, meanline=True)
            ax.set_yscale("log"); ax.set_ylabel("Seconds (log scale)")
            ax.set_title(f"Figure R6c. fwd_iat.avg vs bwd_iat.avg — {cls_lab.capitalize()} (zoom P99)", fontsize=11)
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1))
            ax.yaxis.set_major_formatter(LogFormatterSciNotation())
            ax.grid(axis="y", which="both", linestyle="--", alpha=0.35)
            _save_plot(fig, out_dir, f"eda_rate_box_iat_avg_{cls_lab}_p99.png")

    # ---------------- 6) PCA de variables de ritmo ----------------
    print("⚙️  Ejecutando PCA en variables de ritmo...")
    pca_vars = [v for v in vars_ok]  # todas las de ritmo disponibles
    pca_df = df[pca_vars + ["traffic_class"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pca_df) >= 50 and len(pca_vars) >= 3:
        X = pca_df[pca_vars].values
        Xs = StandardScaler().fit_transform(X)
        k = min(len(pca_vars), 10)
        pca = PCA(n_components=k, random_state=42)
        Xp = pca.fit_transform(Xs)

        var_exp = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(k)],
            "Explained_Variance_%": np.round(pca.explained_variance_ratio_ * 100, 2),
            "Cumulative_%": np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 2)
        })
        _save_via_converter(var_exp, out_dir, "eda_rate_pca_variance", "pca_variance")

        loadings = pd.DataFrame(
            pca.components_.T, index=pca_vars, columns=[f"PC{i+1}" for i in range(k)]
        ).reset_index().rename(columns={"index": "Variable"})
        _save_via_converter(loadings, out_dir, "eda_rate_pca_loadings", "pca_loadings")

        # Scree plot
        fig = plt.figure(figsize=(7.5, 5.5))
        plt.plot(np.arange(1, k + 1), var_exp["Explained_Variance_%"].values, marker="o")
        plt.xlabel("Principal Component"); plt.ylabel("Explained Variance (%)")
        plt.title("Figure P1. PCA Scree Plot – Traffic rate metrics", fontsize=11)
        plt.grid(alpha=0.3)
        _save_plot(fig, out_dir, "eda_rate_pca_scree.png")

        # Proyección 2D
        proj = pd.DataFrame(Xp[:, :2], columns=["PC1", "PC2"])
        proj["traffic_class"] = pca_df["traffic_class"].values
        if len(proj) > 8000:
            proj = proj.sample(8000, random_state=42)
        fig = plt.figure(figsize=(7.5, 5.5))
        color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
        for cls, marker, name, alpha in [("normal", "o", "Normal", 0.5),
                                         ("attack", "x", "Attack", 0.6),
                                         ("unknown", ".", "Unknown", 0.3)]:
            s = proj[proj["traffic_class"] == cls]
            if not s.empty:
                plt.scatter(s["PC1"], s["PC2"], s=10, marker=marker, alpha=alpha, label=name, color=color_map.get(cls))
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title("Figure R8. PCA 2D Projection – Traffic rate", fontsize=11)
        plt.legend(); plt.grid(alpha=0.3)
        _save_plot(fig, out_dir, "eda_rate_pca_2d.png")
    else:
        note = pd.DataFrame([{"note": "PCA skipped (insufficient rows or variables)"}])
        _save_via_converter(note, out_dir, "eda_rate_pca_info", "pca_info")

    print(f"\n traffic_rate listo. Salidas en: {os.path.abspath(out_dir)}")

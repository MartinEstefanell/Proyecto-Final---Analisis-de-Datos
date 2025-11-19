# -*- coding: utf-8 -*-
"""
EDA - Tamaño de los Datos Transmitidos (RT-IoT2022)
---------------------------------------------------
Analiza payload por dirección (fwd/bwd) y total:
- Descriptivas por clase (normal/attack/unknown)
- Razón direccional (fwd/bwd)
- CV por clase, outliers (IQR)
- Correlaciones (heatmap)
- Coherencia interna (checks)
- PCA focalizado en payloads
Exporta XLSX (csv_to_xlsx_format.py) y PNG (matplotlib).
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

# ---------- Config ----------
NORMAL_HINTS = {"normal", "benign", "mqtt_publish"}

# Variables exactas del dataset (payload y headers)
FWD_PAYLOAD = ["fwd_pkts_payload.min", "fwd_pkts_payload.max", "fwd_pkts_payload.tot",
               "fwd_pkts_payload.avg", "fwd_pkts_payload.std"]
BWD_PAYLOAD = ["bwd_pkts_payload.min", "bwd_pkts_payload.max", "bwd_pkts_payload.tot",
               "bwd_pkts_payload.avg", "bwd_pkts_payload.std"]
FLOW_PAYLOAD = ["flow_pkts_payload.min", "flow_pkts_payload.max", "flow_pkts_payload.tot",
                "flow_pkts_payload.avg", "flow_pkts_payload.std"]
HEADERS = ["fwd_header_size_tot", "bwd_header_size_tot"]

ALL_NUM_VARS = FWD_PAYLOAD + BWD_PAYLOAD + FLOW_PAYLOAD + HEADERS

# ---------- Utils ----------
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

# ---------- Gráficos auxiliares (matplotlib puro) ----------
def plot_hist_log_ratio(df, ratio_col, class_col, out_dir):
    sub = df[[ratio_col, class_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty: 
        return
    fig = plt.figure(figsize=(7.5, 5.5))
    color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
    for lab, name in [("normal", "Normal"), ("attack", "Attack"), ("unknown", "Unknown")]:
        vals = sub.loc[sub[class_col] == lab, ratio_col]
        vals = np.log10(vals.replace(0, np.nan).dropna())
        if not vals.empty:
            plt.hist(vals, bins=60, alpha=0.6, label=name, color=color_map.get(lab))
    plt.xlabel("log10(payload_ratio = fwd_tot / (bwd_tot+1))")
    plt.ylabel("Count")
    plt.title("Figure P2. Distribution of payload direction ratio (log10)", fontsize=11)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    _save_plot(fig, out_dir, "eda_payload_ratio_hist_log.png")

def plot_box_fwd_bwd_totals_pretty(df, fwd_col, bwd_col, class_col, out_dir):
    sub = df[[fwd_col, bwd_col, class_col]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[(sub[fwd_col] > 0) & (sub[bwd_col] > 0)]
    if sub.empty:
        return

    color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
    def _one_boxplot(d, cls_label, suffix, clip_p99=False):
        fwd = d[fwd_col].to_numpy()
        bwd = d[bwd_col].to_numpy()
        if clip_p99:
            p99_f, p99_b = np.percentile(fwd, 99), np.percentile(bwd, 99)
            fwd = np.clip(fwd, a_min=np.min(fwd), a_max=p99_f)
            bwd = np.clip(bwd, a_min=np.min(bwd), a_max=p99_b)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        box_colors = [color_map.get(suffix, "#9E9E9E"), color_map.get(suffix, "#9E9E9E")]
        bp = ax.boxplot(
            [fwd, bwd],
            labels=["Forward total", "Backward total"],
            showfliers=True,
            flierprops={"markersize": 2, "alpha": 0.25},
            notch=True,
            widths=0.55,
            showmeans=True,
            meanline=True,
            patch_artist=True
        )
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
        ax.set_yscale("log")
        ax.set_ylabel("Bytes (log scale)")
        title = f"Figure P1. Forward vs Backward payload totals — {cls_label}"
        if clip_p99:
            title += " (zoom P99)"
        ax.set_title(title, fontsize=11)

        # Ejes log bien marcados
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        ax.grid(axis="y", which="both", linestyle="--", alpha=0.35)

        # Anotar medianas
        med_f, med_b = np.median(fwd), np.median(bwd)
        ax.text(1, med_f, f"median≈{int(med_f):,}", va="bottom", ha="center", fontsize=8)
        ax.text(2, med_b, f"median≈{int(med_b):,}", va="bottom", ha="center", fontsize=8)

        fname = f"eda_payload_fwd_vs_bwd_box_{suffix}.png" if not clip_p99 else f"eda_payload_fwd_vs_bwd_box_{suffix}_p99.png"
        _save_plot(fig, out_dir, fname)

    # Normal
    d_norm = sub[sub[class_col] == "normal"]
    if not d_norm.empty:
        _one_boxplot(d_norm, "Normal", "normal", clip_p99=False)
        _one_boxplot(d_norm, "Normal", "normal", clip_p99=True)

    # Attack
    d_att = sub[sub[class_col] == "attack"]
    if not d_att.empty:
        _one_boxplot(d_att, "Attack", "attack", clip_p99=False)
        _one_boxplot(d_att, "Attack", "attack", clip_p99=True)

def plot_scatter_fwdavg_bwdavg(df, fwd_avg, bwd_avg, class_col, out_dir):
    sub = df[[fwd_avg, bwd_avg, class_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty: 
        return
    if len(sub) > 8000:
        sub = sub.sample(8000, random_state=42)
    fig = plt.figure(figsize=(7.5, 5.5))
    color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
    for cls, marker, name, alpha in [("normal", "o", "Normal", 0.5),
                                     ("attack", "x", "Attack", 0.6),
                                     ("unknown", ".", "Unknown", 0.3)]:
        s = sub[sub[class_col] == cls]
        if not s.empty:
            plt.scatter(s[fwd_avg], s[bwd_avg], s=10, marker=marker, alpha=alpha, label=name, color=color_map.get(cls))
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("fwd_pkts_payload.avg (bytes)")
    plt.ylabel("bwd_pkts_payload.avg (bytes)")
    plt.title("Figure P3. Forward vs Backward average payload size (log-log)", fontsize=11)
    plt.legend()
    plt.grid(which="both", linestyle="--", alpha=0.3)
    _save_plot(fig, out_dir, "eda_payload_fwdavg_vs_bwdavg_scatter.png")

def plot_corr_heatmap(mat: pd.DataFrame, out_dir: str, title: str, fname: str):
    if mat.empty: 
        return
    fig_w = max(6, min(1 + 0.35 * mat.shape[1], 18))
    fig_h = max(6, min(1 + 0.35 * mat.shape[0], 18))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(mat.shape[1])); ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(mat.columns.astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mat.index.astype(str), fontsize=8)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    # Anotar valores si no es enorme
    if mat.shape[0] <= 30 and mat.shape[1] <= 30:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    _save_plot(fig, out_dir, fname)

# ---------- Main ----------
def main(df: pd.DataFrame, out_dir: str | None = None):
    out_dir = out_dir or "Outputs_payload_size"
    ensure_outdir(out_dir)
    print("\n📊 Analizando Tamaño de los Datos Transmitidos...")

    # Etiqueta de clase
    if "Attack_type" not in df.columns:
        raise KeyError("Columna Attack_type no encontrada.")
    df = df.copy()
    df["Attack_type"] = norm_cat(df["Attack_type"])
    df["traffic_class"] = df["Attack_type"].apply(
        lambda x: "normal" if is_normal_label(x) else ("attack" if isinstance(x, str) else "unknown")
    )

    # Mantener solo columnas disponibles
    vars_ok = [v for v in ALL_NUM_VARS if v in df.columns]
    if not vars_ok:
        raise KeyError("No se encontraron columnas de payload esperadas en el dataset.")

    # Casting numérico
    for v in vars_ok:
        df[v] = to_numeric_safe(df[v])

    # Derivados clave
    fwd_tot = "fwd_pkts_payload.tot" if "fwd_pkts_payload.tot" in df.columns else None
    bwd_tot = "bwd_pkts_payload.tot" if "bwd_pkts_payload.tot" in df.columns else None
    flow_tot = "flow_pkts_payload.tot" if "flow_pkts_payload.tot" in df.columns else None
    fwd_avg = "fwd_pkts_payload.avg" if "fwd_pkts_payload.avg" in df.columns else None
    bwd_avg = "bwd_pkts_payload.avg" if "bwd_pkts_payload.avg" in df.columns else None

    # Razón direccional
    if fwd_tot and bwd_tot:
        df["payload_ratio"] = df[fwd_tot] / (df[bwd_tot] + 1.0)

    # ---------------- 1) Descriptivas por clase ----------------
    desc_vars = vars_ok.copy()
    if "payload_ratio" in df.columns:
        desc_vars = desc_vars + ["payload_ratio"]

    desc = df.groupby("traffic_class")[desc_vars].agg(["count", "mean", "median", "std", "min", "max"])
    desc.columns = ["_".join(col).strip() for col in desc.columns.values]
    _save_via_converter(desc.reset_index(), out_dir, "eda_payload_descriptivas", "payload_descriptivas")

    # ---------------- 2) CV por clase ----------------
    gstd = df.groupby("traffic_class")[desc_vars].std()
    gmean = df.groupby("traffic_class")[desc_vars].mean().replace(0, np.nan)
    cv = (gstd / gmean).T.rename_axis("variable").reset_index()
    _save_via_converter(cv, out_dir, "eda_payload_cv", "coef_variacion")

    # ---------------- 3) Outliers (IQR) por clase ----------------
    out_rows = []
    for v in desc_vars:
        for cls in df["traffic_class"].unique():
            s = df.loc[df["traffic_class"] == cls, v].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 5:
                out_rows.append({"variable": v, "class": cls, "n": int(len(s)), "iqr_outliers_%": np.nan})
                continue
            q1, q3 = np.percentile(s, [25, 75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            pct = ((s < low) | (s > high)).mean() * 100.0
            out_rows.append({"variable": v, "class": cls, "n": int(len(s)), "iqr_outliers_%": round(pct, 2)})
    out_df = pd.DataFrame(out_rows)
    _save_via_converter(out_df, out_dir, "eda_payload_outliers", "outliers")

    # ---------------- 4) Correlaciones (payload vars) ----------------
    corr_vars = [v for v in (FWD_PAYLOAD + BWD_PAYLOAD + FLOW_PAYLOAD) if v in df.columns]
    corr_mat = df[corr_vars].corr(numeric_only=True)
    _save_via_converter(corr_mat.reset_index(), out_dir, "eda_payload_correlaciones", "correlaciones")
    plot_corr_heatmap(corr_mat, out_dir, "Figure P4. Correlation matrix for payload metrics", "eda_payload_corr_heatmap.png")

    # ---------------- 5) Gráficos clave ----------------
    if "payload_ratio" in df.columns:
        plot_hist_log_ratio(df, "payload_ratio", "traffic_class", out_dir)

    if fwd_tot and bwd_tot:
        plot_box_fwd_bwd_totals_pretty(df, fwd_tot, bwd_tot, "traffic_class", out_dir)

    if fwd_avg and bwd_avg:
        plot_scatter_fwdavg_bwdavg(df, fwd_avg, bwd_avg, "traffic_class", out_dir)

    # ---------------- 6) Checks de coherencia interna ----------------
    checks = []
    # Coherencia: flow_tot ≈ fwd_tot + bwd_tot (cuando existen las 3 columnas)
    if flow_tot and fwd_tot and bwd_tot:
        trip = df[[flow_tot, fwd_tot, bwd_tot]].replace([np.inf, -np.inf], np.nan).dropna()
        if not trip.empty:
            eps = 1e-9
            rel_err = (trip[flow_tot] - (trip[fwd_tot] + trip[bwd_tot])).abs() / (trip[flow_tot].abs() + eps)
            bad_pct = (rel_err > 0.15).mean() * 100.0
            checks.append({"check": "flow_tot_vs_fwd_plus_bwd_relerr>15%", "value_%": round(bad_pct, 2)})

    checks_df = pd.DataFrame(checks) if checks else pd.DataFrame(
        [{"check": "no_coherence_issues_computed", "value_%": np.nan}]
    )
    _save_via_converter(checks_df, out_dir, "eda_payload_coherence_checks", "coherence_checks")

    # ---------------- 7) PCA focalizado en payloads ----------------
    print("⚙️  Ejecutando PCA focalizado en payloads...")
    pca_vars = corr_vars.copy()
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
        _save_via_converter(var_exp, out_dir, "eda_payload_pca_variance", "pca_variance")

        loadings = pd.DataFrame(
            pca.components_.T, index=pca_vars, columns=[f"PC{i+1}" for i in range(k)]
        ).reset_index().rename(columns={"index": "Variable"})
        _save_via_converter(loadings, out_dir, "eda_payload_pca_loadings", "pca_loadings")

        # Scree plot
        fig = plt.figure(figsize=(7.5, 5.5))
        plt.plot(np.arange(1, k + 1), var_exp["Explained_Variance_%"].values, marker="o")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance (%)")
        plt.title("Figure P5. PCA Scree Plot – Payload metrics", fontsize=11)
        plt.grid(alpha=0.3)
        _save_plot(fig, out_dir, "eda_payload_pca_scree.png")

        # Proyección 2D (PC1 vs PC2)
        proj = pd.DataFrame(Xp[:, :2], columns=["PC1", "PC2"])
        proj["traffic_class"] = pca_df["traffic_class"].values
        if len(proj) > 8000:
            proj = proj.sample(8000, random_state=42)
        color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
        fig = plt.figure(figsize=(7.5, 5.5))
        for cls, marker, name, alpha in [("normal", "o", "Normal", 0.5),
                                         ("attack", "x", "Attack", 0.6),
                                         ("unknown", ".", "Unknown", 0.3)]:
            s = proj[proj["traffic_class"] == cls]
            if not s.empty:
                plt.scatter(s["PC1"], s["PC2"], s=10, marker=marker, alpha=alpha, label=name, color=color_map.get(cls))
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title("Figure P6. PCA 2D Projection (payload-only) by class", fontsize=11)
        plt.legend()
        plt.grid(alpha=0.3)
        _save_plot(fig, out_dir, "eda_payload_pca_2d.png")
    else:
        note = pd.DataFrame([{"note": "PCA skipped (insufficient rows or variables)"}])
        _save_via_converter(note, out_dir, "eda_payload_pca_info", "pca_info")

    print(f"\n payload_size listo. Salidas en: {os.path.abspath(out_dir)}")

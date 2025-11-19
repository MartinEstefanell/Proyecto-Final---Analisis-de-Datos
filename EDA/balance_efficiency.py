# -*- coding: utf-8 -*-
"""
EDA - Eficiencia y Balance (RT-IoT2022)
---------------------------------------
- Recibe df y out_dir (no carga archivos).
- Deriva: pkt_balance, byte_balance, rate_balance (y |.|),
         fwd_eff, bwd_eff, flow_eff, overhead_ratio (opcional),
         y usa down_up_ratio nativa.
- Produce: descriptivas, hist balance (full + zoom P99), boxplots de eficiencia
           (incluye versiones zoom P99), scatter byte_balance vs pkt_balance,
           análisis por servicio, correlaciones, CV, outliers (IQR) y PCA.

Requisitos:
- csv_to_xlsx_format.py disponible en la carpeta o en PATH.
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
TOP_SERVICES = 10  # para vistas por servicio
MAX_POINTS_SCATTER = 8000  # muestreo para scatter

DERIVED_BALANCE_VARS = [
    "pkt_balance","byte_balance","rate_balance",
    "abs_pkt_balance","abs_byte_balance","abs_rate_balance",
    "down_up_ratio"
]
DERIVED_EFF_VARS = ["fwd_eff","bwd_eff","flow_eff","overhead_ratio"]

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

# -------------------- Feature engineering --------------------
def _build_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Casting seguro de bases
    for c in [
        "fwd_pkts_tot","bwd_pkts_tot",
        "fwd_pkts_payload.tot","bwd_pkts_payload.tot","flow_pkts_payload.tot",
        "fwd_pkts_per_sec","bwd_pkts_per_sec","flow_pkts_per_sec",
        "fwd_header_size_tot","bwd_header_size_tot",
        "down_up_ratio"
    ]:
        if c in d.columns:
            d[c] = to_numeric_safe(d[c])

    eps = 1e-9

    # Balances (log10 ratio con +1 para evitar /0)
    fwd_pkts = d.get("fwd_pkts_tot", pd.Series(np.nan, index=d.index))
    bwd_pkts = d.get("bwd_pkts_tot", pd.Series(np.nan, index=d.index))
    d["pkt_balance"] = np.log10((fwd_pkts.fillna(0) + 1.0) / (bwd_pkts.fillna(0) + 1.0))

    fwd_bytes = d.get("fwd_pkts_payload.tot", pd.Series(np.nan, index=d.index))
    bwd_bytes = d.get("bwd_pkts_payload.tot", pd.Series(np.nan, index=d.index))
    d["byte_balance"] = np.log10((fwd_bytes.fillna(0) + 1.0) / (bwd_bytes.fillna(0) + 1.0))

    fwd_rate = d.get("fwd_pkts_per_sec", pd.Series(np.nan, index=d.index))
    bwd_rate = d.get("bwd_pkts_per_sec", pd.Series(np.nan, index=d.index))
    d["rate_balance"] = np.log10((fwd_rate.fillna(0) + eps) / (bwd_rate.fillna(0) + eps))

    d["abs_pkt_balance"]  = d["pkt_balance"].abs()
    d["abs_byte_balance"] = d["byte_balance"].abs()
    d["abs_rate_balance"] = d["rate_balance"].abs()

    # Eficiencias (payload / (payload + headers))
    fwd_hdr = d.get("fwd_header_size_tot", pd.Series(np.nan, index=d.index)).fillna(0)
    bwd_hdr = d.get("bwd_header_size_tot", pd.Series(np.nan, index=d.index)).fillna(0)
    d["fwd_eff"] = (fwd_bytes.fillna(0)) / (fwd_bytes.fillna(0) + fwd_hdr + eps)
    d["bwd_eff"] = (bwd_bytes.fillna(0)) / (bwd_bytes.fillna(0) + bwd_hdr + eps)
    flow_bytes = d.get("flow_pkts_payload.tot", pd.Series(np.nan, index=d.index)).fillna(0)
    d["flow_eff"] = flow_bytes / (flow_bytes + fwd_hdr + bwd_hdr + eps)

    # Overhead global (opcional, útil para correlación con flow_eff)
    d["overhead_ratio"] = (fwd_hdr + bwd_hdr) / (flow_bytes + 1.0)

    # Limpiar infinitos
    for c in DERIVED_BALANCE_VARS + DERIVED_EFF_VARS:
        if c in d.columns:
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)

    return d

# -------------------- Plots --------------------
def _hist_by_class(df, var, class_col, out_dir, base_name, bins=60, zoom_p99=True):
    sub = df[[var, class_col]].dropna()
    if sub.empty: return

    # Full
    color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
    fig = plt.figure(figsize=(7.5, 5.5))
    for cls, label, alpha in [("normal","Normal",0.6), ("attack","Attack",0.6), ("unknown","Unknown",0.4)]:
        s = sub[sub[class_col]==cls][var].to_numpy()
        if len(s)==0: continue
        plt.hist(s, bins=bins, alpha=alpha, label=label, density=False, color=color_map.get(cls))
    plt.xlabel(var); plt.ylabel("Count")
    # Determinar el título según la variable
    if var == "rate_balance":
        plt.title("Figure P3. rate_balance - histogram", fontsize=11)
    else:
        plt.title(f"{var} - histogram", fontsize=11)
    plt.legend(); plt.grid(alpha=0.3)
    _save_plot(fig, out_dir, f"{base_name}.png")

    # Zoom P99 (percentiles por clase)
    if zoom_p99:
        fig = plt.figure(figsize=(7.5, 5.5))
        for cls, label, alpha in [("normal","Normal",0.6), ("attack","Attack",0.6), ("unknown","Unknown",0.4)]:
            s = sub[sub[class_col]==cls][var].to_numpy()
            if len(s)==0: continue
            lo, hi = np.percentile(s, [1, 99])
            s = np.clip(s, lo, hi)
            plt.hist(s, bins=bins, alpha=alpha, label=label, density=False, color=color_map.get(cls))
        plt.xlabel(var + " (zoom P1-P99)"); plt.ylabel("Count")
        # Determinar el título según la variable
        if var == "pkt_balance":
            plt.title("Figure P1. pkt_balance - histogram (zoom)", fontsize=11)
        elif var == "byte_balance":
            plt.title("Figure P2. byte_balance - histogram (zoom)", fontsize=11)
        elif var == "rate_balance":
            plt.title(f"{var} - histogram (zoom)", fontsize=11)
        else:
            plt.title(f"{var} - histogram (zoom)", fontsize=11)
        plt.legend(); plt.grid(alpha=0.3)
        _save_plot(fig, out_dir, f"{base_name}_zoom.png")

def _box_eff_by_class(df, var, class_col, out_dir, base_name):
    sub = df[[var, class_col]].dropna()
    if sub.empty: return
    fig = plt.figure(figsize=(7, 5.2))
    data, labels = [], []
    for cls, label in [("normal","Normal"), ("attack","Attack"), ("unknown","Unknown")]:
        s = sub[sub[class_col]==cls][var].to_numpy()
        if len(s)==0: continue
        data.append(s); labels.append(label)
    if not data:
        plt.close(fig); return
    plt.boxplot(data, labels=labels, showmeans=True, meanline=True, widths=0.55,
                flierprops={"markersize":2, "alpha":0.25}, notch=True)
    plt.ylim(0,1)
    plt.ylabel(var)
    plt.title(f"{var} by class", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    _save_plot(fig, out_dir, f"{base_name}.png")

def _box_eff_by_class_zoom_p99(df, var, class_col, out_dir, out_name):
    """
    Boxplot por clase con rango común P1–P99 (normal+attack) + media en línea verde
    y anotación de mediana.
    """
    sub = df[[var, class_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty: return

    s_norm = sub[sub[class_col] == "normal"][var].to_numpy()
    s_attk = sub[sub[class_col] == "attack"][var].to_numpy()

    if len(s_norm) == 0 and len(s_attk) == 0:
        return

    # P1–P99 global usando los datos disponibles
    stack = s_norm if len(s_attk) == 0 else (s_attk if len(s_norm) == 0 else np.hstack([s_norm, s_attk]))
    p1, p99 = np.percentile(stack, [1, 99])

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    data, labels = [], []
    if len(s_norm):  data.append(s_norm);  labels.append("Normal")
    if len(s_attk): data.append(s_attk);  labels.append("Attack")

    bp = ax.boxplot(
        data, labels=labels, showfliers=True, widths=0.55, notch=True
    )
    ax.set_ylim(p1, p99)
    ax.set_ylabel(var)
    ax.set_title(f"{var} by class (zoom P99)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Medias (línea verde discontinua por caja)
    if len(s_norm):
        ax.hlines(np.mean(s_norm), 0.8, 1.2, linestyles="--", colors="green", linewidth=1.4)
    if len(s_attk):
        x_low = 1.8 if len(s_norm) else 0.8
        x_high = 2.2 if len(s_norm) else 1.2
        ax.hlines(np.mean(s_attk), x_low, x_high, linestyles="--", colors="green", linewidth=1.4)

    # Etiquetas de mediana (usar color rojo para indicar attack/median)
    for i, line in enumerate(bp["medians"], start=1):
        y = float(line.get_ydata()[0])
        ax.text(i, y, f"median≈{y:.3f}", ha="center", va="bottom", fontsize=9, color="#F44336")

    _save_plot(fig, out_dir, out_name)

def _scatter_balance(df, xvar, yvar, class_col, out_dir, base_name):
    sub = df[[xvar, yvar, class_col]].dropna()
    if sub.empty: return
    if len(sub) > MAX_POINTS_SCATTER:
        sub = sub.sample(MAX_POINTS_SCATTER, random_state=42)
    color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
    fig = plt.figure(figsize=(7.5, 5.5))
    for cls, marker, label, alpha in [("normal","o","Normal",0.5),
                                      ("attack","x","Attack",0.6),
                                      ("unknown",".","Unknown",0.3)]:
        s = sub[sub[class_col]==cls]
        if s.empty: continue
        plt.scatter(s[xvar], s[yvar], s=10, marker=marker, alpha=alpha, label=label, color=color_map.get(cls))
    plt.xlabel(xvar); plt.ylabel(yvar)
    # Determinar el título según las variables
    if xvar == "byte_balance" and yvar == "pkt_balance":
        plt.title("Figure P4. byte_balance vs pkt_balance", fontsize=11)
    else:
        plt.title(f"{xvar} vs {yvar}", fontsize=11)
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
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax); cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")
    if mat.shape[0] <= 30 and mat.shape[1] <= 30:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    _save_plot(fig, out_dir, fname)

# -------------------- MAIN --------------------
def main(df: pd.DataFrame, out_dir: str | None = None):
    out_dir = out_dir or "Outputs_balance_efficiency"
    ensure_outdir(out_dir)
    print("\n📊 Analizando Eficiencia y Balance...")

    # Requisitos mínimos
    required_any = [
        "fwd_pkts_tot","bwd_pkts_tot",
        "fwd_pkts_payload.tot","bwd_pkts_payload.tot","flow_pkts_payload.tot",
        "fwd_pkts_per_sec","bwd_pkts_per_sec",
        "fwd_header_size_tot","bwd_header_size_tot",
        "down_up_ratio","Attack_type"
    ]
    missing = [c for c in required_any if c not in df.columns]
    if missing:
        print(f"⚠️  Columnas ausentes (se continúa si hay suficientes): {missing}")

    # Clase (normal/attack/unknown)
    d0 = df.copy()
    d0["Attack_type"] = norm_cat(d0["Attack_type"])
    d0["traffic_class"] = d0["Attack_type"].apply(lambda x: "normal" if is_normal_label(x) else ("attack" if isinstance(x, str) else "unknown"))

    # Derivar features
    d = _build_derived(d0)

    # Calidad básica de derivadas
    qrows = []
    for c in DERIVED_BALANCE_VARS + DERIVED_EFF_VARS:
        if c in d.columns:
            qrows.append({"metric": f"nulls_in_{c}", "value": int(d[c].isna().sum())})
    quality_df = pd.DataFrame(qrows) if qrows else pd.DataFrame([{"metric":"note","value":"no checks"}])
    _save_via_converter(quality_df, out_dir, "balance_quality_checks", "quality_checks")

    # Descriptivas por clase
    desc_vars = [c for c in DERIVED_BALANCE_VARS + DERIVED_EFF_VARS if c in d.columns]
    if desc_vars:
        desc = d.groupby("traffic_class")[desc_vars].agg(["count","mean","median","std","min","max"])
        desc.columns = ["_".join(col).strip() for col in desc.columns.values]
        _save_via_converter(desc.reset_index(), out_dir, "balance_descriptivas", "descriptivas")

    # 1) Histogramas de balance (full + zoom)
    for v in ["pkt_balance","byte_balance","rate_balance"]:
        if v in d.columns:
            _hist_by_class(d, v, "traffic_class", out_dir, f"balance_hist_{v}", bins=60, zoom_p99=True)

    # 2) Boxplots de eficiencia por clase (normal y versión zoom P99)
    for v in ["fwd_eff","bwd_eff","flow_eff"]:
        if v in d.columns:
            _box_eff_by_class(d, v, "traffic_class", out_dir, f"eff_box_{v}")
            _box_eff_by_class_zoom_p99(d, v, "traffic_class", out_dir, f"eff_box_{v}_(zoom_P99).png")

    # 3) Scatter/hexbin de relaciones entre balances
    if all(v in d.columns for v in ["byte_balance","pkt_balance"]):
        _scatter_balance(d, "byte_balance", "pkt_balance", "traffic_class", out_dir, "balance_scatter_byte_vs_pkt")

    # 4) Análisis por servicio (top N por frecuencia)
    if "service" in d.columns:
        svc = norm_cat(d["service"])
        d["service_norm"] = svc
        top_services = svc.value_counts(dropna=True).head(TOP_SERVICES).index.tolist()
        svc_df = d[d["service_norm"].isin(top_services)].copy()
        cols_for_service = [c for c in ["byte_balance","pkt_balance","rate_balance","flow_eff"] if c in d.columns]
        if cols_for_service and not svc_df.empty:
            svc_desc = svc_df.groupby(["service_norm","traffic_class"])[cols_for_service].median().reset_index()
            _save_via_converter(svc_desc, out_dir, "balance_by_service_median", "by_service_median")

    # 5) Correlación del sub-bloque (balances + eficiencias + down_up_ratio)
    corr_vars = [c for c in DERIVED_BALANCE_VARS + DERIVED_EFF_VARS if c in d.columns]
    corr_mat = d[corr_vars].corr(numeric_only=True) if corr_vars else pd.DataFrame()
    if not corr_mat.empty:
        _save_via_converter(corr_mat.reset_index(), out_dir, "balance_correlaciones", "correlaciones")
        _corr_heatmap(corr_mat, "Figure P5. Balance/Efficiency Correlations", out_dir, "balance_corr_heatmap.png")

    # 6) CV y outliers (IQR) por clase
    cv_rows, out_rows = [], []
    for v in desc_vars:
        for cls in d["traffic_class"].unique():
            s = d.loc[d["traffic_class"]==cls, v].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) >= 5:
                cv = (np.std(s) / (np.mean(s) + 1e-12))
                cv_rows.append({"variable": v, "class": cls, "n": int(len(s)), "cv": round(cv, 4)})
                q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
                low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
                pct = ((s < low) | (s > high)).mean() * 100.0
                out_rows.append({"variable": v, "class": cls, "n": int(len(s)), "iqr_outliers_%": round(pct, 2)})
            else:
                cv_rows.append({"variable": v, "class": cls, "n": int(len(s)), "cv": np.nan})
                out_rows.append({"variable": v, "class": cls, "n": int(len(s)), "iqr_outliers_%": np.nan})
    if cv_rows:
        _save_via_converter(pd.DataFrame(cv_rows), out_dir, "balance_cv", "cv")
    if out_rows:
        _save_via_converter(pd.DataFrame(out_rows), out_dir, "balance_outliers", "outliers")

    # 7) PCA del sub-bloque
    pca_vars = [c for c in ["pkt_balance","byte_balance","rate_balance","fwd_eff","bwd_eff","flow_eff","down_up_ratio"] if c in d.columns]
    pca_df = d[pca_vars + ["traffic_class"]].dropna()
    if len(pca_df) >= 50 and len(pca_vars) >= 3:
        X = pca_df[pca_vars].values
        Xs = StandardScaler().fit_transform(X)
        k = min(len(pca_vars), 8)
        pca = PCA(n_components=k, random_state=42)
        Xp = pca.fit_transform(Xs)

        var_exp = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(k)],
            "Explained_Variance_%": np.round(pca.explained_variance_ratio_ * 100, 2),
            "Cumulative_%": np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 2)
        })
        _save_via_converter(var_exp, out_dir, "balance_pca_variance", "pca_variance")

        loadings = pd.DataFrame(pca.components_.T, index=pca_vars, columns=[f"PC{i+1}" for i in range(k)])
        loadings = loadings.reset_index().rename(columns={"index":"Variable"})
        _save_via_converter(loadings, out_dir, "balance_pca_loadings", "pca_loadings")

        # Scree
        fig = plt.figure(figsize=(7.2, 5.2))
        plt.plot(np.arange(1, k+1), var_exp["Explained_Variance_%"].values, marker="o")
        plt.xlabel("PC"); plt.ylabel("Explained Var (%)")
        plt.title("Figure P7. Balance/Efficiency PCA - Scree", fontsize=11)
        plt.grid(alpha=0.3)
        _save_plot(fig, out_dir, "balance_pca_scree.png")

        # 2D
        proj = pd.DataFrame(Xp[:, :2], columns=["PC1","PC2"])
        proj["traffic_class"] = pca_df["traffic_class"].values
        if len(proj) > MAX_POINTS_SCATTER:
            proj = proj.sample(MAX_POINTS_SCATTER, random_state=42)
        fig = plt.figure(figsize=(7.5, 5.5))
        color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
        for cls, marker, label, alpha in [("normal","o","Normal",0.5),
                                          ("attack","x","Attack",0.6),
                                          ("unknown",".","Unknown",0.3)]:
            s = proj[proj["traffic_class"]==cls]
            if s.empty: continue
            plt.scatter(s["PC1"], s["PC2"], s=10, marker=marker, alpha=alpha, label=label, color=color_map.get(cls))
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title("Figure P6. Balance/Efficiency PCA - 2D", fontsize=11)
        plt.grid(alpha=0.3); plt.legend()
        _save_plot(fig, out_dir, "balance_pca_2d.png")
    else:
        _save_via_converter(pd.DataFrame([{"note":"PCA skipped (insufficient rows/vars)"}]), out_dir, "balance_pca_info", "pca_info")

    print(f"\n balance_efficiency listo. Salidas en: {os.path.abspath(out_dir)}")

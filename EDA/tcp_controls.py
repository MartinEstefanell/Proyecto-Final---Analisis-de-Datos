# -*- coding: utf-8 -*-
"""
EDA - Control de conexión TCP (RT-IoT2022)
------------------------------------------
Bloque 5: TCP control plane (banderas y ventanas).
- Filtra proto = tcp
- Descriptivas de banderas (conteos) y tasas por segundo
- Direccionalidad PSH/URG
- Taxonomía de handshake/cierre
- ECN (ECE/CWR) prevalencia
- Ventanas (init/last) y colapso de ventana
- Correlaciones del bloque
- PCA del bloque TCP (banderas/ventanas/tasas)
- Outliers IQR por clase
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

FLAG_VARS = [
    "flow_SYN_flag_count","flow_ACK_flag_count","flow_FIN_flag_count","flow_RST_flag_count",
    "flow_ECE_flag_count","flow_CWR_flag_count",
    "fwd_PSH_flag_count","bwd_PSH_flag_count","fwd_URG_flag_count","bwd_URG_flag_count"
]

WINDOW_VARS = [
    "fwd_init_window_size","bwd_init_window_size","fwd_last_window_size"  # agrega bwd_last_window_size si existe
]

# tasas por segundo (se crean)
RATE_VARS = [
    "syn_rate","ack_rate","fin_rate","rst_rate","psh_fwd_rate","psh_bwd_rate",
    "urg_fwd_rate","urg_bwd_rate","ece_rate","cwr_rate"
]

ALL_TCP_VARS = FLAG_VARS + WINDOW_VARS  # base para casting numérico/chequeos

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

# -------------------- Helpers de análisis --------------------
def _add_rates(df_tcp: pd.DataFrame) -> pd.DataFrame:
    """Crea tasas por segundo para flags (evita sesgo por duración)."""
    d = df_tcp.copy()
    dur = to_numeric_safe(d.get("flow_duration", pd.Series(np.nan, index=d.index)))
    dur = dur.where(dur > 0, np.nan)

    def rate(col):
        v = to_numeric_safe(d.get(col, pd.Series(0, index=d.index)))
        return v / dur

    d["syn_rate"] = rate("flow_SYN_flag_count")
    d["ack_rate"] = rate("flow_ACK_flag_count")
    d["fin_rate"] = rate("flow_FIN_flag_count")
    d["rst_rate"] = rate("flow_RST_flag_count")
    d["psh_fwd_rate"] = rate("fwd_PSH_flag_count")
    d["psh_bwd_rate"] = rate("bwd_PSH_flag_count")
    d["urg_fwd_rate"] = rate("fwd_URG_flag_count")
    d["urg_bwd_rate"] = rate("bwd_URG_flag_count")
    d["ece_rate"] = rate("flow_ECE_flag_count")
    d["cwr_rate"] = rate("flow_CWR_flag_count")
    return d

def _handshake_taxonomy(row) -> str:
    syn = (row.get("flow_SYN_flag_count", 0) or 0) > 0
    ack = (row.get("flow_ACK_flag_count", 0) or 0) > 0
    fin = (row.get("flow_FIN_flag_count", 0) or 0) > 0
    rst = (row.get("flow_RST_flag_count", 0) or 0) > 0
    if rst:
        return "reset"
    if syn and not ack and not fin:
        return "half_open"
    if ack and fin and not rst:
        return "established_closed"
    if ack and not fin and not rst:
        return "no_close"
    return "other"

def _collapse_flag(init_v: float, last_v: float, thr: float = 0.25) -> int:
    try:
        if pd.isna(init_v) or pd.isna(last_v) or init_v <= 0:
            return 0
        return int(last_v < thr * init_v)
    except Exception:
        return 0

# -------------------- Plots --------------------
def _bar_counts_by_class(df, cols, class_col, title, fname):
    color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
    sub = df[[class_col] + cols].copy()
    agg = sub.groupby(class_col)[cols].sum().T
    fig, ax = plt.subplots(figsize=(9.5, 6))
    x = np.arange(len(agg.index))
    width = 0.8 / max(1, len(agg.columns))
    for i, cls in enumerate(agg.columns):
        color = color_map.get(cls, "#9E9E9E")
        ax.bar(x + i*width, agg[cls].values, width=width, label=str(cls).capitalize(), color=color)
    ax.set_xticks(x); ax.set_xticklabels(agg.index.astype(str), rotation=45, ha="right")
    ax.set_ylabel("Total count")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    _save_plot(fig, out_dir=fname.rsplit(os.sep,1)[0], name=fname.rsplit(os.sep,1)[1])

def _box_log_by_class(df, var, class_col, title, fname, clip_p99=False):
    sub = df[[var, class_col]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[sub[var] >= 0]
    if sub.empty: return

    def _plot(vals, label_suffix):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        data, labels = [], []
        for lab, name in [("normal", "Normal"), ("attack", "Attack"), ("unknown", "Unknown")]:
            s = vals[vals[class_col] == lab][var].to_numpy()
            if len(s) == 0: 
                continue
            if clip_p99:
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
        ax.set_title(title + label_suffix, fontsize=11)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        ax.grid(axis="y", which="both", linestyle="--", alpha=0.35)
        _save_plot(fig, out_dir=fname.rsplit(os.sep,1)[0], name=fname.rsplit(os.sep,1)[1].replace(".png", f"{label_suffix.replace(' ','_')}.png"))

    _plot(sub, "")
    if clip_p99:
        _plot(sub, " (zoom P99)")

def _scatter_loglog(df, xvar, yvar, class_col, title, fname):
    color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
    sub = df[[xvar, yvar, class_col]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[(sub[xvar] > 0) & (sub[yvar] > 0)]
    if sub.empty: return
    if len(sub) > 8000:
        sub = sub.sample(8000, random_state=42)

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
    plt.legend(); plt.grid(which="both", linestyle="--", alpha=0.3)
    _save_plot(fig, out_dir=fname.rsplit(os.sep,1)[0], name=fname.rsplit(os.sep,1)[1])

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

# -------------------- MAIN --------------------
def main(df: pd.DataFrame, out_dir: str | None = None):
    out_dir = out_dir or "Outputs_tcp_control"
    ensure_outdir(out_dir)
    print("\n📊 Analizando Control de Conexión TCP...")

    # Chequeo de columnas mínimas
    required = ["proto","flow_duration","Attack_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}")

    # Clase
    df = df.copy()
    df["Attack_type"] = norm_cat(df["Attack_type"])
    df["traffic_class"] = df["Attack_type"].apply(lambda x: "normal" if is_normal_label(x) else ("attack" if isinstance(x, str) else "unknown"))

    # Filtrar TCP
    proto_norm = norm_cat(df["proto"])
    df_tcp = df[proto_norm == "tcp"].copy()
    if df_tcp.empty:
        raise ValueError("No hay registros TCP para analizar.")

    # Casting numérico en variables TCP
    for v in ALL_TCP_VARS + ["flow_duration"]:
        if v in df_tcp.columns:
            df_tcp[v] = to_numeric_safe(df_tcp[v])

    # Crear tasas por segundo
    df_tcp = _add_rates(df_tcp)

    # ---- 1) Alcance y calidad (resumen básico) ----
    qual_rows = []
    for c in FLAG_VARS + WINDOW_VARS + ["flow_duration"]:
        if c in df_tcp.columns:
            qual_rows.append({"metric": f"nulls_in_{c}", "value": int(df_tcp[c].isna().sum())})
    quality_df = pd.DataFrame(qual_rows) if qual_rows else pd.DataFrame([{"metric":"note","value":"no checks"}])
    _save_via_converter(quality_df, out_dir, "tcp_quality_checks", "quality_checks")

    # ---- 2) Distribuciones de banderas: descriptivas por clase ----
    flag_cols_present = [c for c in FLAG_VARS if c in df_tcp.columns]
    if flag_cols_present:
        desc_flags = df_tcp.groupby("traffic_class")[flag_cols_present].agg(["count","mean","median","std","min","max"])
        desc_flags.columns = ["_".join(col).strip() for col in desc_flags.columns.values]
        _save_via_converter(desc_flags.reset_index(), out_dir, "tcp_flags_descriptivas", "flags_descriptivas")

        # barras totales por clase (SYN/ACK/FIN/RST/ECE/CWR)
        base_flags = [c for c in ["flow_SYN_flag_count","flow_ACK_flag_count","flow_FIN_flag_count","flow_RST_flag_count","flow_ECE_flag_count","flow_CWR_flag_count"] if c in flag_cols_present]
        if base_flags:
            _bar_counts_by_class(df_tcp, base_flags, "traffic_class",
                                 "Figure P2. TCP Flags Totals by class", os.path.join(out_dir, "tcp_flags_totals_by_class.png"))

    # ---- 3) Tasas por segundo: boxplots por clase ----
    rate_cols_present = [c for c in RATE_VARS if c in df_tcp.columns]
    desc_rates = pd.DataFrame()
    if rate_cols_present:
        desc_rates = df_tcp.groupby("traffic_class")[rate_cols_present].agg(["count","mean","median","std","min","max"])
        desc_rates.columns = ["_".join(col).strip() for col in desc_rates.columns.values]
        _save_via_converter(desc_rates.reset_index(), out_dir, "tcp_rates_descriptivas", "rates_descriptivas")

        # ack_rate
        if "ack_rate" in rate_cols_present:
            _box_log_by_class(df_tcp, "ack_rate", "traffic_class", "Figure P3. ack_rate by class", os.path.join(out_dir, "tcp_box_ack_rate.png"), clip_p99=True)
        # syn_rate
        if "syn_rate" in rate_cols_present:
            _box_log_by_class(df_tcp, "syn_rate", "traffic_class", "Figure P4. syn_rate by class", os.path.join(out_dir, "tcp_box_syn_rate.png"), clip_p99=True)
        # rst_rate
        if "rst_rate" in rate_cols_present:
            _box_log_by_class(df_tcp, "rst_rate", "traffic_class", "Figure P5. rst_rate by class", os.path.join(out_dir, "tcp_box_rst_rate.png"), clip_p99=True)

    # ---- 4) Direccionalidad PSH/URG ----
    if all(c in df_tcp.columns for c in ["fwd_PSH_flag_count","bwd_PSH_flag_count"]):
        _scatter_loglog(df_tcp, "fwd_PSH_flag_count","bwd_PSH_flag_count","traffic_class",
                        "Figure P6. fwd_PSH vs bwd_PSH (log-log)", os.path.join(out_dir, "tcp_scatter_psh_fwd_vs_bwd.png"))
        _box_log_by_class(df_tcp, "fwd_PSH_flag_count", "traffic_class", "Figure P7. fwd_PSH by class", os.path.join(out_dir, "tcp_box_fwd_psh.png"), clip_p99=True)
        _box_log_by_class(df_tcp, "bwd_PSH_flag_count", "traffic_class", "Figure P8. bwd_PSH by class", os.path.join(out_dir, "tcp_box_bwd_psh.png"), clip_p99=True)

    if all(c in df_tcp.columns for c in ["fwd_URG_flag_count","bwd_URG_flag_count"]):
        _scatter_loglog(df_tcp, "fwd_URG_flag_count","bwd_URG_flag_count","traffic_class",
                        "Figure P5. fwd_URG vs bwd_URG (log-log)", os.path.join(out_dir, "tcp_scatter_urg_fwd_vs_bwd.png"))
        _box_log_by_class(df_tcp, "fwd_URG_flag_count", "traffic_class", "fwd_URG by class", os.path.join(out_dir, "tcp_box_fwd_urg.png"), clip_p99=True)
        _box_log_by_class(df_tcp, "bwd_URG_flag_count", "traffic_class", "bwd_URG by class", os.path.join(out_dir, "tcp_box_bwd_urg.png"), clip_p99=True)

    # ---- 5) Taxonomía handshake/cierre ----
    df_tcp["tcp_pattern"] = df_tcp.apply(_handshake_taxonomy, axis=1)
    tax = pd.crosstab(df_tcp["tcp_pattern"], df_tcp["traffic_class"]).reset_index()
    _save_via_converter(tax, out_dir, "tcp_handshake_taxonomy", "handshake_taxonomy")

    # gráfico de categorías por clase
    color_map = {"normal": "#43A047", "attack": "#F44336", "unknown": "#9E9E9E"}
    tax_plot = pd.crosstab(df_tcp["tcp_pattern"], df_tcp["traffic_class"])
    if not tax_plot.empty:
        # Ordenar columnas para que normal/attack/unknown siempre estén en el mismo orden visual
        ordered_cols = [c for c in ["normal", "attack", "unknown"] if c in tax_plot.columns]
        tax_plot = tax_plot[ordered_cols]
        colors = [color_map.get(c, "#9E9E9E") for c in tax_plot.columns]
        fig, ax = plt.subplots(figsize=(9.5, 6))
        tax_plot.plot(kind="bar", ax=ax, color=colors)
        ax.set_title("Figure P1. TCP Handshake Taxonomy by class", fontsize=11)
        ax.set_xlabel("Taxonomy"); ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        ax.legend([c.capitalize() for c in tax_plot.columns])
        _save_plot(fig, out_dir, "tcp_handshake_taxonomy_bar.png")

    # ---- 6) ECN (ECE/CWR) ----
    if "flow_ECE_flag_count" in df_tcp.columns or "flow_CWR_flag_count" in df_tcp.columns:
        color_map = {"normal": "#1976D2", "attack": "#F44336", "unknown": "#9E9E9E"}
        ece = to_numeric_safe(df_tcp.get("flow_ECE_flag_count", pd.Series(0, index=df_tcp.index))).fillna(0)
        cwr = to_numeric_safe(df_tcp.get("flow_CWR_flag_count", pd.Series(0, index=df_tcp.index))).fillna(0)
        df_tcp["ecn_present"] = ((ece > 0) | (cwr > 0)).astype(int)
        ecn_summary = df_tcp.groupby("traffic_class")["ecn_present"].mean().reset_index()
        ecn_summary["ecn_present_%"] = (ecn_summary["ecn_present"] * 100).round(2)
        ecn_summary.drop(columns=["ecn_present"], inplace=True)
        _save_via_converter(ecn_summary, out_dir, "tcp_ecn_summary", "ecn_summary")

        # barras
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        bars = ax.bar(ecn_summary["traffic_class"], ecn_summary["ecn_present_%"],
                     color=[color_map.get(cls, "#9E9E9E") for cls in ecn_summary["traffic_class"]])
        ax.set_ylabel("% flows with ECN (ECE/CWR>0)")
        ax.set_title("Figure P3. ECN presence by class", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        _save_plot(fig, out_dir, "tcp_ecn_presence_bar.png")

    # ---- 7) Ventanas y colapso ----
    win_cols = [c for c in ["fwd_init_window_size","bwd_init_window_size","fwd_last_window_size"] if c in df_tcp.columns]
    for w in win_cols:
        df_tcp[w] = to_numeric_safe(df_tcp[w])
    if "fwd_init_window_size" in df_tcp.columns and "fwd_last_window_size" in df_tcp.columns:
        color_map = {"normal": "#1976D2", "attack": "#F44336", "unknown": "#9E9E9E"}
        df_tcp["fwd_window_collapse"] = [
            _collapse_flag(i, l, thr=0.25) for i, l in zip(df_tcp["fwd_init_window_size"], df_tcp["fwd_last_window_size"])
        ]
        collapse = df_tcp.groupby("traffic_class")["fwd_window_collapse"].mean().reset_index()
        collapse["collapse_%"] = (collapse["fwd_window_collapse"] * 100).round(2)
        collapse.drop(columns=["fwd_window_collapse"], inplace=True)
        _save_via_converter(collapse, out_dir, "tcp_window_collapse", "window_collapse")

        # bar colapso
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        bars = ax.bar(collapse["traffic_class"], collapse["collapse_%"],
                     color=[color_map.get(cls, "#9E9E9E") for cls in collapse["traffic_class"]])
        ax.set_ylabel("% with window collapse (last < 0.25*init)")
        ax.set_title("Figure P4. TCP window collapse (forward) by class", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        _save_plot(fig, out_dir, "tcp_window_collapse_bar.png")

    # boxplots ventanas (log)
    for w in ["fwd_init_window_size","bwd_init_window_size","fwd_last_window_size"]:
        if w in df_tcp.columns:
            _box_log_by_class(df_tcp, w, "traffic_class", f"{w} by class", os.path.join(out_dir, f"tcp_box_{w}.png"), clip_p99=True)

    # ---- 8) Correlaciones (bloque TCP) ----
    corr_vars = []
    for c in FLAG_VARS + RATE_VARS + WINDOW_VARS:
        if c in df_tcp.columns:
            corr_vars.append(c)
    corr_mat = df_tcp[corr_vars].corr(numeric_only=True) if corr_vars else pd.DataFrame()
    if not corr_mat.empty:
        _save_via_converter(corr_mat.reset_index(), out_dir, "tcp_correlaciones", "correlaciones")
        _corr_heatmap(corr_mat, "Figure P7. TCP Control Block Correlations", os.path.join(out_dir, "tcp_corr_heatmap.png"))

    # ---- 9) Outliers (IQR) por clase en tasas/ventanas ----
    out_rows = []
    out_vars = [v for v in RATE_VARS + WINDOW_VARS if v in df_tcp.columns]
    for v in out_vars:
        for cls in df_tcp["traffic_class"].unique():
            s = df_tcp.loc[df_tcp["traffic_class"] == cls, v].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 5:
                out_rows.append({"variable": v, "class": cls, "n": int(len(s)), "iqr_outliers_%": np.nan})
                continue
            q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            pct = ((s < low) | (s > high)).mean() * 100.0
            out_rows.append({"variable": v, "class": cls, "n": int(len(s)), "iqr_outliers_%": round(pct, 2)})
    out_df = pd.DataFrame(out_rows)
    _save_via_converter(out_df, out_dir, "tcp_outliers", "outliers")

    # ---- 10) PCA del bloque TCP ----
    print("⚙️  Ejecutando PCA en bloque TCP...")
    pca_vars = [v for v in (FLAG_VARS + RATE_VARS + WINDOW_VARS) if v in df_tcp.columns]
    pca_df = df_tcp[pca_vars + ["traffic_class"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pca_df) >= 50 and len(pca_vars) >= 3:
        X = pca_df[pca_vars].values
        # log1p a tasas y ventanas no negativas (por seguridad si hay colas)
        # (banderas ya son conteos; tasas pueden tener muchos ceros)
        log_mask = np.array([1 if ("rate" in v or "window" in v) else 0 for v in pca_vars], dtype=bool)
        X_log = X.copy()
        X_log[:, log_mask] = np.log1p(np.clip(X[:, log_mask], a_min=0, a_max=None))
        Xs = StandardScaler().fit_transform(X_log)

        k = min(len(pca_vars), 10)
        pca = PCA(n_components=k, random_state=42)
        Xp = pca.fit_transform(Xs)

        var_exp = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(k)],
            "Explained_Variance_%": np.round(pca.explained_variance_ratio_ * 100, 2),
            "Cumulative_%": np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 2)
        })
        _save_via_converter(var_exp, out_dir, "tcp_pca_variance", "pca_variance")

        loadings = pd.DataFrame(pca.components_.T, index=pca_vars, columns=[f"PC{i+1}" for i in range(k)])
        loadings = loadings.reset_index().rename(columns={"index":"Variable"})
        _save_via_converter(loadings, out_dir, "tcp_pca_loadings", "pca_loadings")

        # Scree plot
        fig = plt.figure(figsize=(7.5, 5.5))
        plt.plot(np.arange(1, k + 1), var_exp["Explained_Variance_%"].values, marker="o")
        plt.xlabel("Principal Component"); plt.ylabel("Explained Variance (%)")
        plt.title("Figure P9. TCP control PCA - Scree", fontsize=11)
        plt.grid(alpha=0.3)
        _save_plot(fig, out_dir, "tcp_pca_scree.png")

        # Proyección 2D
        color_map = {"normal": "#1976D2", "attack": "#F44336", "unknown": "#9E9E9E"}
        proj = pd.DataFrame(Xp[:, :2], columns=["PC1", "PC2"])
        proj["traffic_class"] = pca_df["traffic_class"].values
        if len(proj) > 8000:
            proj = proj.sample(8000, random_state=42)
        fig = plt.figure(figsize=(7.5, 5.5))
        for cls, marker, name, alpha in [("normal", "o", "Normal", 0.5),
                                         ("attack", "x", "Attack", 0.6),
                                         ("unknown", ".", "Unknown", 0.3)]:
            s = proj[proj["traffic_class"] == cls]
            if not s.empty:
                plt.scatter(s["PC1"], s["PC2"], s=10, marker=marker, alpha=alpha, label=name, color=color_map.get(cls))
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title("Figure P10. TCP control PCA - 2D projection", fontsize=11)
        plt.legend(); plt.grid(alpha=0.3)
        _save_plot(fig, out_dir, "tcp_pca_2d.png")
    else:
        note = pd.DataFrame([{"note": "PCA skipped (insufficient rows or variables)"}])
        _save_via_converter(note, out_dir, "tcp_pca_info", "pca_info")

    print(f"\n tcp_control listo. Salidas en: {os.path.abspath(out_dir)}")

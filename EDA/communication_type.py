# -*- coding: utf-8 -*-
"""
EDA - Tipo de Comunicación (RT-IoT2022)
---------------------------------------
Versión con títulos abreviados en las figuras.
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TOP_PROTOCOLS = 10
TOP_SERVICES = 15
NORMAL_HINTS = {"normal", "benign", "mqtt_publish"}

# -------------------- Utils --------------------

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def norm_cat(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    return s.astype(str).str.strip().str.lower().replace({"": np.nan})

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def is_normal_label(x: str) -> bool:
    if not isinstance(x, str):
        return False
    x = x.strip().lower()
    return any(h in x for h in NORMAL_HINTS)

def _find_converter() -> str | None:
    names = ["csv_to_xlsx_format.py", "csv_to_xlsx_format"]
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "csv_to_xlsx_format.py"),
        os.path.join(os.path.dirname(here), "csv_to_xlsx_format.py"),
        os.path.join(os.getcwd(), "csv_to_xlsx_format.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "csv_to_xlsx_format"

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
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Fallo el conversor para {base_name}: {e.stderr or e.stdout}") from e

        print(f"[XLSX] {xlsx_out}")
        return xlsx_out

def _save_plot(fig, out_dir: str, name: str):
    ensure_outdir(out_dir)
    out = os.path.join(out_dir, name)
    fig.tight_layout(pad=1.0)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[PNG] {out}")
    plt.close(fig)

# -------------------- Plots --------------------

def _plot_top_protocols(proto_freq: pd.DataFrame, out_dir: str):
    if proto_freq is None or proto_freq.empty:
        return
    topk = proto_freq.head(TOP_PROTOCOLS)
    fig = plt.figure()
    plt.bar(topk["proto"].astype(str), topk["count"])
    plt.xlabel("Protocol")
    plt.ylabel("Count")
    plt.title("Figure 1. IoT Protocol Distribution", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    _save_plot(fig, out_dir, "eda_top_protocols.png")

def _plot_top_services(service_freq: pd.DataFrame, out_dir: str):
    if service_freq is None or service_freq.empty:
        return
    topk = service_freq.head(TOP_SERVICES)
    fig = plt.figure()
    plt.bar(topk["service"].astype(str), topk["count"])
    plt.xlabel("Service")
    plt.ylabel("Count")
    plt.title("Figure 2. IoT Service Frequency", fontsize=11)
    plt.xticks(rotation=60, ha="right")
    _save_plot(fig, out_dir, "eda_top_services.png")

def _plot_services_stacked(service_by_attack: pd.DataFrame, out_dir: str):
    if service_by_attack is None or service_by_attack.empty:
        return
    svc_tot = service_by_attack.sum(axis=1).sort_values(ascending=False).head(12)
    svc_sel = service_by_attack.loc[svc_tot.index]
    x = np.arange(len(svc_sel.index))
    normals = svc_sel.get("normal", pd.Series([0]*len(x), index=svc_sel.index)).values
    attacks = svc_sel.get("attack", pd.Series([0]*len(x), index=svc_sel.index)).values

    fig = plt.figure()
    # Colors: normal -> green, attack -> red
    plt.bar(x, normals, 0.6, label="Normal", color="#43A047")
    plt.bar(x, attacks, 0.6, bottom=normals, label="Attack", color="#F44336")
    plt.xticks(x, svc_sel.index.astype(str), rotation=60, ha="right")
    plt.xlabel("Service (Top 12)")
    plt.ylabel("Count")
    plt.title("Figure 3. Services: Normal vs Attack", fontsize=11)
    plt.legend()
    _save_plot(fig, out_dir, "eda_services_normal_vs_attack_stacked.png")

def _plot_full_heatmap(df: pd.DataFrame, out_dir: str, proto_col: str, service_col: str):
    sub = df.dropna(subset=[service_col, proto_col])
    if sub.empty:
        return
    svc = norm_cat(sub[service_col])
    prt = norm_cat(sub[proto_col])
    contingency = pd.crosstab(svc, prt)

    fig_w = max(6, min(1 + 0.35 * contingency.shape[1], 20))
    fig_h = max(6, min(1 + 0.35 * contingency.shape[0], 20))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(contingency.values, aspect="auto")
    ax.set_xticks(np.arange(contingency.shape[1]))
    ax.set_xticklabels(contingency.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(np.arange(contingency.shape[0]))
    ax.set_yticklabels(contingency.index.astype(str))
    ax.set_xlabel("Protocol")
    ax.set_ylabel("Service")
    ax.set_title("Figure 4. Service vs Protocol", fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

    # Se eliminan los números en las celdas del heatmap

    _save_plot(fig, out_dir, "eda_heatmap_servicio_vs_proto_full.png")

# -------------------- MAIN --------------------

def main(df: pd.DataFrame, out_dir: str | None = None):
    out_dir = out_dir or r"C:\Users\agust\Escritorio\Estudio\Semestres\6to Semestre\Análisis de Datos\Proyecto Final\Proyecto-Final---Analisis-de-Datos\EDA\output_communication_type"
    ensure_outdir(out_dir)

    proto_col = "proto" if "proto" in df.columns else None
    service_col = "service" if "service" in df.columns else None
    attack_col = "Attack_type" if "Attack_type" in df.columns else ("attack_type" if "attack_type" in df.columns else None)

    proto_freq = pd.DataFrame()
    service_freq = pd.DataFrame()
    proto_by_attack = pd.DataFrame()
    service_by_attack = pd.DataFrame()
    services_only_attack = pd.DataFrame(columns=["service"])
    quality_df = pd.DataFrame()

    if proto_col:
        pf = df[proto_col].value_counts(dropna=False).rename_axis("proto").reset_index(name="count")
        proto_freq = pf
        _save_via_converter(proto_freq, out_dir, "eda_proto_frecuencias", sheet_name="proto_frecuencias")

    if service_col:
        sf = df[service_col].value_counts(dropna=False).rename_axis("service").reset_index(name="count")
        service_freq = sf
        _save_via_converter(service_freq, out_dir, "eda_service_frecuencias", sheet_name="service_frecuencias")

    is_attack = None
    if attack_col:
        att_norm = norm_cat(df[attack_col])
        is_attack = ~att_norm.apply(is_normal_label)

    if proto_col and is_attack is not None and is_attack.notna().any():
        proto_by_attack = pd.crosstab(df[proto_col], is_attack, dropna=False).rename(columns={False: "normal", True: "attack"})
        _save_via_converter(proto_by_attack.reset_index(), out_dir, "eda_proto_normal_vs_attack", sheet_name="proto_normal_vs_attack")

    if service_col and is_attack is not None and is_attack.notna().any():
        service_by_attack = pd.crosstab(df[service_col], is_attack, dropna=False).rename(columns={False: "normal", True: "attack"})
        _save_via_converter(service_by_attack.reset_index(), out_dir, "eda_service_normal_vs_attack", sheet_name="service_normal_vs_attack")

        only_attack_mask = (service_by_attack.get("attack", pd.Series(dtype=int)) > 0) & \
                           (service_by_attack.get("normal", pd.Series(dtype=int)) == 0)
        services_only_attack = service_by_attack[only_attack_mask].reset_index()[["service"]]
        _save_via_converter(services_only_attack, out_dir, "eda_services_solo_ataque", sheet_name="services_solo_ataque")

    quality_checks = []
    for c in [proto_col, service_col, attack_col, "id.orig_p", "id.resp_p"]:
        if c and c in df.columns:
            quality_checks.append({"check": f"nulls_in_{c}", "value": int(df[c].isna().sum())})
    for name in ["id.orig_p", "id.resp_p"]:
        if name in df.columns:
            s = to_numeric_safe(df[name])
            bad = s.dropna().astype(float)
            out_of_range = ((bad < 0) | (bad > 65535)).sum()
            quality_checks.append({"check": f"out_of_range_{name}", "value": int(out_of_range)})
    if quality_checks:
        quality_df = pd.DataFrame(quality_checks)
        _save_via_converter(quality_df, out_dir, "eda_quality_checks", sheet_name="quality_checks")

    _plot_top_protocols(proto_freq, out_dir)
    _plot_top_services(service_freq, out_dir)
    _plot_full_heatmap(df, out_dir, proto_col=proto_col, service_col=service_col)
    if not service_by_attack.empty:
        _plot_services_stacked(service_by_attack, out_dir)

    print(f"\ communication_type listo. Salidas en: {os.path.abspath(out_dir)}")

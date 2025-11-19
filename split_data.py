# split_data.py — División 80/20 estratificada por ataque (is_attack) + reporte JSON
import argparse, os, sys, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- Configuración según tus clases RT-IoT2022 ---
ATTACK_LABELS = {
    "dos_syn_hping", "ddos_slowloris", "arp_poisioning"
}
BENIGN_LABELS = {
    "thing_speak", "mqtt_publish", "wipro_bulb"
}
UNKNOWN_SENTINELS = {"", "unknown", "none", "-"}

# --- Nombres de columna candidatos ---
LABEL_CANDIDATES = [
    "Attack_type", "attack_type",
    "Label", "label",
    " Attack_type", " attack_type",
    " Label", " label",
]

def here(*parts):
    return os.path.join(os.path.dirname(__file__), *parts)

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def normalize_str(s):
    return str(s).strip().lower() if pd.notna(s) else ""

def guess_label_col(df):
    cols_norm = {c: c.strip() for c in df.columns}
    # candidatos exactos o con espacios
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
        cand_stripped = cand.strip()
        if cand_stripped in cols_norm.values():
            for k, v in cols_norm.items():
                if v == cand_stripped:
                    return k
    # fallback: algo que contenga 'attack' o sea 'label'
    for c in df.columns:
        c_low = c.lower().strip()
        if "attack" in c_low or c_low == "label":
            return c
    raise ValueError(f"No se encontró columna de etiqueta. Columnas disponibles: {list(df.columns)}")

def derive_is_attack(df, col):
    """
    Deriva is_attack con dtype pandas nullable Float64 (soporta pd.NA sin castear a float nativo).
    """
    s = df[col].apply(normalize_str)

    # serie anulable
    is_attack = pd.Series(pd.NA, index=df.index, dtype="Float64")

    unknown_mask = s.isin(UNKNOWN_SENTINELS)
    is_attack[s.isin(ATTACK_LABELS)] = 1
    is_attack[s.isin(BENIGN_LABELS)]  = 0
    is_attack[unknown_mask]           = pd.NA
    is_attack[~(s.isin(ATTACK_LABELS | BENIGN_LABELS | UNKNOWN_SENTINELS))] = pd.NA

    df["is_attack"] = is_attack
    return df

def report_block(name, df):
    """
    Calcula conteos y porcentajes de is_attack (incluye NA), imprime y devuelve un dict serializable.
    """
    y = df["is_attack"]

    counts = y.value_counts(dropna=False)
    pct_series = (y.value_counts(normalize=True, dropna=False) * 100)
    pct_series = pd.to_numeric(pct_series, errors="coerce").fillna(0.0).round(2)

    # Serialización segura (claves str; valores nativos)
    counts_dict = {str(k): int(v) if pd.notna(v) else 0 for k, v in counts.items()}
    pct_dict    = {str(k): float(v) if pd.notna(v) else 0.0 for k, v in pct_series.items()}

    info = {
        "rows": int(len(df)),
        "is_attack_counts": counts_dict,
        "is_attack_pct": pct_dict,
    }

    print(f"\n[Reporte] {name}\n{json.dumps(info, indent=2, ensure_ascii=False)}")
    return name, info

def parse_args_with_defaults():
    ap = argparse.ArgumentParser(description="Split 80/20 estratificado por ataque (is_attack).")
    ap.add_argument("--input", help="Archivo CSV o Excel limpio.")
    ap.add_argument("--outdir", help="Carpeta donde se guardarán los splits.")
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--test",  type=float, default=0.20)
    ap.add_argument("--seed",  type=int,   default=42)
    args = ap.parse_args()

    # Defaults automáticos
    if not args.input:
        xlsx = here("clean_dataset_RT-IoT2022.xlsx")
        csv  = here("clean_dataset_RT-IoT2022.csv")
        if os.path.exists(xlsx):
            args.input = xlsx
        elif os.path.exists(csv):
            args.input = csv
        else:
            raise FileNotFoundError(
                "No se encontró archivo de entrada por defecto. "
                "Pasa --input o deja 'clean_dataset_RT-IoT2022.xlsx/csv' junto al script."
            )
    if not args.outdir:
        args.outdir = here("SPLITS")

    return args

def main():
    args = parse_args_with_defaults()

    if round(args.train + args.test, 8) != 1.0:
        raise ValueError("Las proporciones deben sumar 1.0 (train + test)")

    print("[split] Input:", args.input)
    print("[split] Outdir:", args.outdir)

    # --- Lectura flexible ---
    if args.input.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)

    # --- Detectar etiqueta y derivar is_attack ---
    label_col = guess_label_col(df)
    print(f"[split] Columna de etiqueta detectada: '{label_col}'")
    df = derive_is_attack(df, col=label_col)

    # --- Separar unknown ---
    df_unknown = df[df["is_attack"].isna()].copy()
    df_labeled = df[~df["is_attack"].isna()].copy()
    df_labeled["is_attack"] = df_labeled["is_attack"].astype(int)

    if df_labeled.empty:
        raise RuntimeError("No se encontraron registros etiquetados (is_attack 0/1).")

    # --- Split 80/20 estratificado ---
    y = df_labeled["is_attack"]
    df_tr, df_te = train_test_split(
        df_labeled, test_size=args.test, random_state=args.seed, stratify=y
    )

    # --- Guardar ---
    ensure_outdir(args.outdir)
    df_tr.to_csv(os.path.join(args.outdir, "train.csv"), index=False)
    df_te.to_csv(os.path.join(args.outdir, "test.csv"),  index=False)
    if not df_unknown.empty:
        df_unknown.to_csv(os.path.join(args.outdir, "unknown.csv"), index=False)

    print(f"\n✅ Archivos generados en {args.outdir}:")
    print(" - train.csv (80%)")
    print(" - test.csv  (20%)")
    if not df_unknown.empty:
        print(" - unknown.csv (registros sin etiqueta usada en el split)")

    # --- Reportes (acumulados y guardados a JSON al final) ---
    reports = {}
    for title, dataf in [
        ("Dataset completo", df),
        ("Etiquetados (solo 0/1)", df_labeled),
        ("Train", df_tr),
        ("Test", df_te),
    ]:
        try:
            k, v = report_block(title, dataf)
            reports[k] = v
        except Exception as e:
            print(f"[WARN] No se pudo generar reporte '{title}': {e}")

    # Guardar JSON al final
    out_json = os.path.join(args.outdir, "split_report.json")
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Reporte JSON guardado en: {out_json}")
    except Exception as e:
        print(f"[WARN] No se pudo guardar el JSON: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)

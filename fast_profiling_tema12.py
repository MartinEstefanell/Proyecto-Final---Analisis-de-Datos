import os
import argparse
import pandas as pd
from ydata_profiling import ProfileReport

DATA_PATH = r"C:\Users\marti\OneDrive - Universidad de Montevideo\Analisis de Datos\Tema_12_limpio.xlsx"


def parse_args():
    p = argparse.ArgumentParser(description='Generar data profiling (modo rápido por defecto si el dataset es grande)')
    p.add_argument('--full', action='store_true', help='Forzar profiling completo (puede consumir mucha memoria/CPU)')
    p.add_argument('--fast', action='store_true', help='Forzar modo rápido (muestra pequeña y minimal=True)')
    p.add_argument('--sample-size', type=int, default=2000, help='Tamaño de la muestra en modo rápido (por defecto 2000)')
    p.add_argument('--threshold', type=int, default=50000, help='Umbral de filas para activar muestreo automático')
    return p.parse_args()


def main():
    args = parse_args()

    # ENV override to force full profiling
    force_env = os.getenv('FORCE_FULL_PROFILING', '0') in ('1', 'true', 'True')
    FORCE_FULL = args.full or force_env
    FORCE_FAST = args.fast

    print(f"Cargando datos desde: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    n_rows = len(df)
    print(f"Dimensiones: {n_rows:,} filas × {df.shape[1]} columnas")

    # Decide mode
    if FORCE_FULL:
        mode = 'full'
    elif FORCE_FAST:
        mode = 'fast'
    else:
        mode = 'fast' if n_rows > args.threshold else 'full'

    if mode == 'fast':
        sample_size = min(args.sample_size, n_rows)
        print(f"Modo RÁPIDO: usando muestra de {sample_size:,} filas y perfil minimal para reducir carga.")
        sample_df = df.sample(n=sample_size, random_state=42)
        # minimal=True hace el reporte mucho más rápido; explorative=False para ahorrar tiempo
        profile = ProfileReport(sample_df, title="Perfil Rápido - Tema 12 (Muestra)", minimal=True, explorative=False)
        out_name = "Tema_12_Perfil_Rapido_Muestra_minimal.html"
        profile.to_file(out_name)
        print(f"✅ Perfil rápido guardado: {out_name}")
        print("Si querés más detalle, ejecutá con --full o define FORCE_FULL_PROFILING=1 en tu entorno (puede tardar mucho).")

    else:
        print("Generando perfil completo (esto puede tardar y consumir muchos recursos).")
        profile = ProfileReport(df, title="Perfil Completo - Tema 12", explorative=True, minimal=False)
        out_name = "Tema_12_Perfil_Completo.html"
        profile.to_file(out_name)
        print(f"✅ Perfil completo guardado: {out_name}")


if __name__ == '__main__':
    main()

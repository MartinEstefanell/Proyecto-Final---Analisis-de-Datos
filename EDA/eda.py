#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA Runner (orquestador)
------------------------
- Recibe la ruta a un CSV (por CLI o via input()).
- Carga el CSV en un DataFrame sin limpiar.
- Llama a main(df, out_dir) en cada módulo EDA listado.
- Si un módulo no define main(df, out_dir), se omite con aviso.

Uso:
  python eda.py ruta/al/archivo.csv [--out Outputs]

Requisitos: pandas
"""

import sys
import argparse
import importlib
import pandas as pd
from pathlib import Path

# Lista de módulos EDA (crealos luego con su main(df, out_dir))
MODULES = [
    "communication_type",
    "intensity_size",
    "payload_size",
    "balance_efficiency",
    "burstiness_subflow_1",
    "burstiness_subflow_2",
    "traffic_rate",
    "tcp_controls",
]

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Orquestador EDA: lee CSV y despacha a módulos main(df, out_dir).")
    p.add_argument("csv_path", nargs="?", help="Ruta al CSV de entrada")
    return p.parse_args(argv)

def ask_path_if_missing(path_str: str | None = None) -> str:

    fixed_path = r"C:\Users\agust\Escritorio\Estudio\Semestres\6to Semestre\Análisis de Datos\Proyecto Final\Proyecto-Final---Analisis-de-Datos\dataset_RT-IoT2022.csv"

    import os
    if not os.path.exists(fixed_path):
        raise FileNotFoundError(f"No se encontró el archivo en: {fixed_path}")

    return fixed_path
def safe_import(module_name: str):
    """Importa un módulo sin romper la ejecución si no existe."""
    # Try importing as a submodule of the EDA package first, then as a top-level module.
    try:
        # e.g. import EDA.burstiness_subflow
        return importlib.import_module(f"EDA.{module_name}")
    except ModuleNotFoundError:
        pass
    except Exception as e:
        print(f"⚠️  Error importando EDA.{module_name}: {e}")
        return None

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        # Last resort: try loading from the EDA directory by file path
        try:
            from importlib import util
            from pathlib import Path
            eda_dir = Path(__file__).parent
            candidate = eda_dir / f"{module_name}.py"
            if candidate.exists():
                spec = util.spec_from_file_location(module_name, str(candidate))
                mod = util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                return mod
            else:
                print(f"⚠️  Módulo no encontrado: {module_name} (se omite)")
                return None
        except Exception as e:
            print(f"⚠️  Error cargando desde archivo {module_name}.py: {e}")
            return None
    except Exception as e:
        print(f"⚠️  Error importando {module_name}: {e}")
        return None

def call_main_if_exists(mod, df: pd.DataFrame, module_name: str, base_dir: Path):
    """Ejecuta main(df, out_dir) si existe, creando carpeta específica para el módulo."""
    if mod is None:
        return
    fn = getattr(mod, "main", None)
    if callable(fn):
        try:
            # Crear carpeta específica para este módulo: EDA/Outputs_{nombre_módulo}
            out_dir = base_dir / f"Outputs_{module_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"▶️  {mod.__name__}.main(df, out_dir='{out_dir}')")
            fn(df=df, out_dir=str(out_dir))
        except Exception as e:
            print(f"❌ Error ejecutando {mod.__name__}.main(df, out_dir): {e}")
    else:
        print(f"⚠️  {mod.__name__} no define main(df, out_dir). Se omite.")

def main():
    args = parse_args()
    csv_path = ask_path_if_missing(args.csv_path)
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise SystemExit(f"No se encontró el CSV: {csv_file}")

    # Directorio base de EDA (donde está este script)
    eda_dir = Path(__file__).parent

    # Cargar archivo (detectar si es CSV o Excel)
    print(f"📥 Cargando archivo: {csv_file}")
    
    if csv_file.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(csv_file)
    else:
        # Intentar con diferentes codificaciones para CSV
        try:
            df = pd.read_csv(csv_file, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_file, low_memory=False, encoding='latin-1')
            except:
                df = pd.read_csv(csv_file, low_memory=False, encoding='iso-8859-1')
    
    print(f"✅ DataFrame cargado: {len(df):,} filas × {len(df.columns)} columnas")

    # Importar y despachar a cada módulo con su carpeta específica
    for name in MODULES:
        mod = safe_import(name)
        call_main_if_exists(mod, df, name, eda_dir)

    print(f" EDA finalizado. Salidas en carpetas EDA/Outputs_{{módulo}}")

if __name__ == "__main__":
    main()

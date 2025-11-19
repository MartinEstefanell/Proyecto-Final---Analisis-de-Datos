# CLEANING/cleaning.py
from __future__ import annotations
import json, sys, os, importlib.util, inspect
from typing import Dict, Any, List
import pandas as pd
import math

THIS_DIR = os.path.dirname(__file__)
CHECKPOINTS_DIR = os.path.join(THIS_DIR, ".checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

OUTPUT_NAME = "clean_dataset_RT-IoT2022.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(THIS_DIR), OUTPUT_NAME)
def find_input_csv_arg_or_first() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    # primer .csv en CLEANING/ que no empiece con "clean_"
    cands = [f for f in os.listdir(THIS_DIR)
             if f.lower().endswith(".csv") and not os.path.basename(f).startswith("clean_")]
    if not cands:
        raise FileNotFoundError("No encontré CSV de entrada en CLEANING/ ni se pasó ruta por argumento.")
    return os.path.join(THIS_DIR, sorted(cands)[0])

def load_cleaning_modules() -> Dict[str, Any]:
    """
    Importa cada cleaning_*.py (excepto este) y toma la primera clase que tenga:
      - atributo 'name' (str)
      - atributo 'depends_on' (list[str], opcional)
      - método run(self, df) -> (df, report)
    Devuelve {stage_name: instancia}
    """
    stages: Dict[str, Any] = {}
    py_files = [f for f in os.listdir(THIS_DIR)
                if f.startswith("cleaning_") and f.endswith(".py") and f != "cleaning.py"]
    for fname in sorted(py_files):
        fpath = os.path.join(THIS_DIR, fname)
        spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
        if not spec or not spec.loader: 
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore

        cls = None
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == mod.__name__ and hasattr(obj, "run") and hasattr(obj, "name"):
                cls = obj
                break
        if not cls:
            continue

        inst = cls()
        stage_name = getattr(inst, "name", None)
        if not isinstance(stage_name, str):
            continue
        # si hay colisión de 'name', nos quedamos con el primero que vimos (evita duplicados)
        if stage_name not in stages:
            stages[stage_name] = inst
    return stages

def topo_sort(stages: Dict[str, Any]) -> List[str]:
    # grafo: dep -> node
    deps_map = {name: list(getattr(inst, "depends_on", []) or []) for name, inst in stages.items()}
    indeg = {name: 0 for name in stages}
    for name, deps in deps_map.items():
        for d in deps:
            if d in stages:
                indeg[name] += 1

    # nodos sin entrada
    queue = sorted([n for n, d in indeg.items() if d == 0])
    order: List[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m, deps in deps_map.items():
            if n in deps:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
                    queue.sort()

    # si quedó algo (ciclo/deps ausentes), agregamos en orden estable
    if len(order) != len(stages):
        remaining = [n for n in stages.keys() if n not in order]
        order += sorted(remaining)
    return order

def _nan_to_none(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj

def main():
    input_csv = find_input_csv_arg_or_first()
    print(f"[cleaning] Leyendo: {os.path.relpath(input_csv, THIS_DIR)}")
    df = pd.read_csv(input_csv)

    stages = load_cleaning_modules()
    if not stages:
        raise RuntimeError("No se encontraron módulos cleaning_*.py")

    order = topo_sort(stages)
    print("[cleaning] Orden topológico:", " -> ".join(order))

    reports: Dict[str, Any] = {"order": order, "modules": {}}
    cur = df
    for name in order:
        inst = stages[name]
        print(f"[cleaning] Ejecutando {name} ({inst.__class__.__name__})")
        cur, rep = inst.run(cur)
        reports["modules"][name] = rep
        # checkpoint por módulo
        ckpt = os.path.join(CHECKPOINTS_DIR, f"{name}.csv.gz")
        cur.to_csv(ckpt, index=False, compression="gzip")

    # guardar CSV final en carpeta padre
    cur.to_csv(OUTPUT_PATH, index=False)
    print(f"[cleaning] Guardado final: ../{os.path.basename(OUTPUT_PATH)}")

    with open(os.path.join(CHECKPOINTS_DIR, "cleaning_report.json"), "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(reports), f, ensure_ascii=False, indent=2)
    print("[cleaning] Reporte: .checkpoints/cleaning_report.json")

if __name__ == "__main__":
    main()

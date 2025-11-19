import pandas as pd
import os

# Ruta del CSV (ajustá según tu estructura)
csv_path = r"C:\Users\agust\Escritorio\Estudio\Semestres\6to Semestre\Análisis de Datos\Proyecto Final\Proyecto-Final---Analisis-de-Datos\clean_dataset_RT-IoT2022.csv"
# Ruta de salida para el Excel
xlsx_path = os.path.splitext(csv_path)[0] + ".xlsx"  # mismo nombre, distinta extensión

# Leer el CSV
df = pd.read_csv(csv_path)

# Guardar como Excel
df.to_excel(xlsx_path, index=False)

print(f"x Archivo convertido: {xlsx_path}")

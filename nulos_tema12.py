import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar el dataset (mismo archivo que usás en profiling)
df = pd.read_excel(r"C:\Users\marti\OneDrive - Universidad de Montevideo\Analisis de Datos\Tema_12_limpio.xlsx")

# 2. Definir categorías y colores
categories = {
    'Identificadores Clave': ['id.orig_p', 'id.resp_p', 'proto', 'service', 'Unnamed: 0'],
    'Variables Temporales': ['flow_duration', 'active.min', 'active.max', 'active.tot', 'active.avg', 'active.std', 'idle.min', 'idle.max', 'idle.tot', 'idle.avg', 'idle.std'],
    'Volumen y Tráfico': ['fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_data_pkts_tot', 'bwd_data_pkts_tot', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec', 'down_up_ratio'],
    'Tamaño de Cabecera': ['fwd_header_size_tot', 'bwd_header_size_tot', 'fwd_header_size_min', 'fwd_header_size_max', 'bwd_header_size_min', 'bwd_header_size_max'],
    'Variables Tipo Bandera': ['flow_FIN_flag_count', 'flow_SYN_flag_count', 'flow_RST_flag_count', 'fwd_PSH_flag_count', 'bwd_PSH_flag_count', 'flow_ACK_flag_count', 'fwd_URG_flag_count', 'bwd_URG_flag_count', 'flow_CWR_flag_count', 'flow_ECE_flag_count'],
    'Estadísticas de Payload': ['fwd_pkts_payload.min', 'fwd_pkts_payload.max', 'fwd_pkts_payload.tot', 'fwd_pkts_payload.avg', 'fwd_pkts_payload.std', 'bwd_pkts_payload.min', 'bwd_pkts_payload.max', 'bwd_pkts_payload.tot', 'bwd_pkts_payload.avg', 'bwd_pkts_payload.std', 'flow_pkts_payload.min', 'flow_pkts_payload.max', 'flow_pkts_payload.tot', 'flow_pkts_payload.avg', 'flow_pkts_payload.std', 'payload_bytes_per_second'],
    'Tiempo entre Paquetes': ['fwd_iat.min', 'fwd_iat.max', 'fwd_iat.tot', 'fwd_iat.avg', 'fwd_iat.std', 'bwd_iat.min', 'bwd_iat.max', 'bwd_iat.tot', 'bwd_iat.avg', 'bwd_iat.std', 'flow_iat.min', 'flow_iat.max', 'flow_iat.tot', 'flow_iat.avg', 'flow_iat.std'],
    'Subflujos y Ventanas TCP': ['fwd_subflow_pkts', 'bwd_subflow_pkts', 'fwd_subflow_bytes', 'bwd_subflow_bytes', 'fwd_bulk_bytes', 'bwd_bulk_bytes', 'fwd_bulk_packets', 'bwd_bulk_packets', 'fwd_bulk_rate', 'bwd_bulk_rate', 'fwd_init_window_size', 'bwd_init_window_size', 'fwd_last_window_size'],
    'Variable Objetivo': ['Attack_type']
}

category_colors = {
    'Tiempo entre Paquetes': '#F18F01',
    'Variable Objetivo': '#795548', 
    'Variables Temporales': '#A23B72',
    'Tamaño de Cabecera': '#C73E1D',
    'Identificadores Clave': '#2E86AB',
    'Estadísticas de Payload': '#9C27B0',
    'Subflujos y Ventanas TCP': '#607D8B',
    'Volumen y Tráfico': '#4CAF50',
    'Variables Tipo Bandera': '#4CAF50'
}

# 3. Calcular valores nulos (solo los que tienen nulos)
null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0].sort_values(ascending=False)

# 4. Preparar datos para el gráfico
top_nulls = null_counts.head(25)
colors = []
clean_labels = []

for column in top_nulls.index:
    # Encontrar categoría
    category = 'Otras Variables'
    for cat, cols in categories.items():
        if column in cols:
            category = cat
            break
    colors.append(category_colors.get(category, '#808080'))
    clean_labels.append(column.replace('_', ' ').replace('.', ' '))

# 5. Crear el gráfico
plt.figure(figsize=(16, 10))
bars = plt.bar(range(len(top_nulls)), top_nulls.values, color=colors, alpha=0.8)

# 6. Configurar el gráfico exactamente como tu imagen
plt.title('Cantidad de valores nulos por columna', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Variables', fontsize=12)
plt.ylabel('Cantidad', fontsize=12)

# Configurar ejes con rango ajustado como en tu imagen
plt.xticks(range(len(top_nulls)), clean_labels, rotation=45, ha='right')
max_value = max(top_nulls.values)
min_value = min(top_nulls.values)
range_values = max_value - min_value
margin = range_values * 0.05
plt.ylim(min_value - margin, max_value + margin)

# Añadir valores encima de las barras
for bar, value in zip(bars, top_nulls.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + range_values*0.01, 
            f'{int(value):,}', ha='center', va='bottom', fontsize=9)

# 7. Crear leyenda por categorías (como en tu imagen)
legend_elements = []
categories_in_chart = set()
for column in top_nulls.index:
    for cat, cols in categories.items():
        if column in cols:
            categories_in_chart.add(cat)
            break

for category in sorted(categories_in_chart):
    color = category_colors.get(category, '#808080')
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=category))

plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

# 8. Ajustar y guardar
plt.tight_layout()
plt.grid(axis='y', alpha=0.3)
plt.savefig('Tema_12_Valores_Nulos_por_Categoria.png', dpi=300, bbox_inches='tight', facecolor='white')

print("✅ Gráfico generado: Tema_12_Valores_Nulos_por_Categoria.png")
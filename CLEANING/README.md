REGLAS PCA X MODULO


2 — Flow Intensity

Subespacio: Volumen y ritmo del flujo

Variables (ejemplos): flow_pkts_payload.tot, payload_bytes_per_second, flow_pkts_per_sec, flow_iat.avg, active.avg, idle.avg, down_up_ratio

Transformaciones previas: Imputación por mediana → (log1p opcional en sesgadas) → Z-score

Componentes retenidas: 2–3 (≥ 85% var.)

Variables preservadas fuera del PCA: Representantes de magnitud/ritmo/regularidad (si definiste conservarlos)

3 — Payload Features

Subespacio: Tamaños y estadísticas de payload

Variables (ejemplos): …payload.tot/avg/std, payload_bytes_per_second, etc.

Transformaciones previas: Imputación por mediana → log1p (skew>1) → Z-score

Componentes retenidas: 3–4 (≥ 80% var.)

Variables preservadas fuera del PCA: payload_ratio_log

4 — Traffic Rate

Subespacio: Ritmo temporal / IAT

Variables (ejemplos): flow_byts_s, flow_pkts_s, flow_iat.avg/std, active.*, idle.*

Transformaciones previas: Imputación por mediana → log1p → Z-score

Componentes retenidas: 3–4 (≥ 80% var.)

Variables preservadas fuera del PCA: Ninguna (todas reemplazadas por PCs)

5 — TCP Control

Subespacio: Flags y ventanas TCP

Variables (ejemplos): Flags: flow_SYN/ACK/FIN/RST/PSH/URG_flag_count; Ventanas: fwd/bwd_init_window_size, fwd/bwd_last_window_size

Transformaciones previas: Validación semántica (no negativos, enterizado) → Clip P1–P99 por clase → Mediana/IQR (robust) en ventanas → PCA

Componentes retenidas: hasta 4 (≥ 80% var., según datos)

Variables preservadas fuera del PCA: Sí: flags originales y ventanas robustas (*_r) para interpretabilidad

6 — Efficiency & Balance

Subespacio: Eficiencia, simetrías y ratios direccionales

Variables (ejemplos): payload_efficiency, bytes_per_pkt_flow, header_payload_ratio, rate_ratio_log

Transformaciones previas: Imputación por mediana → log1p (no negativas) → Z-score

Componentes retenidas: 2–3 (≥ 80% var.)

Variables preservadas fuera del PCA: payload_ratio_log, sym_bytes, sym_pkts

7 — Burstiness & Subflow

Subespacio: Ráfagas y microflujos

Variables (ejemplos): burst_size_mean/std/max/min, burst_rate_mean/std/max/min, subflow_count, subflow_duration_mean/std

Transformaciones previas: Imputación por mediana → log1p (skew>1) → Z-score

Componentes retenidas: hasta 4 (≥ 80% var.)

Variables preservadas fuera del PCA: Ninguna (el bloque se reemplaza por burst_pca_*)






El proceso completo se orquesta con el script maestro cleaning.py, que:

Descubre automáticamente todos los archivos cleaning_*.py del directorio.

Resuelve dependencias entre módulos (depends_on).

Ejecuta los módulos en orden topológico.

Guarda checkpoints parciales (.csv.gz o .parquet) y un reporte final en JSON.

Genera el CSV limpio final en la carpeta padre.
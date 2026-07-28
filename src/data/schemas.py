PHENOLOGY_FILES = [
    "consolidado_fechas_2024_limpio.csv",
    "consolidado_fechas_2025_limpio.csv",
]

TEMPERATURE_FILE = "temperaturas_horarias_por_lugar.csv"

REQUIRED_PHENOLOGY_COLUMNS = [
    "temporada",
    "codigo_lugar",
    "lugar",
    "codigo_variedad",
    "variedad",
    "ue",
    "codigo_tratamiento_entrada",
    "codigo_tratamiento_salida",
    "fecha_caida_hoja_50",
    "fecha_brotacion_camara_50",
    "fecha_brotacion_campo_50",
]

REQUIRED_TEMPERATURE_COLUMNS = [
    "codigo_lugar",
    "lugar",
    "condicion",
    "latitud",
    "longitud",
    "ds",
    "temperatura_c",
]

DATE_COLUMNS = [
    "fecha_caida_hoja_50",
    "fecha_brotacion_camara_50",
    "fecha_brotacion_campo_50",
]

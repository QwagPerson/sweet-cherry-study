from __future__ import annotations

import pandas as pd


CATEGORICAL_COLD_FEATURES = [
    "codigo_variedad",
    "condicion",
    "codigo_tratamiento_entrada",
]

CATEGORICAL_HEAT_FEATURES = [
    "codigo_variedad",
    "condicion",
    "codigo_tratamiento_entrada",
    "codigo_tratamiento_salida",
    "interaccion_entrada_salida",
]

BASE_NUMERIC_FEATURES = [
    "latitud",
    "longitud",
    "temp_media_historica",
    "temp_min_historica",
    "temp_max_historica",
    "amplitud_termica_historica",
    "horas_helada_historicas",
    "horas_calidas_historicas",
    "horas_bajo_7_2_historicas",
    "horas_bajo_10_historicas",
    "hgc_historicas",
    "variabilidad_interanual_temp",
    "variabilidad_interanual_frio",
    "variabilidad_interanual_calor",
]

MONTHLY_FEATURES = [f"temp_media_historica_mes_{month:02d}" for month in range(1, 13)]

COLD_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + MONTHLY_FEATURES + [
    "dia_del_ano_caida_50",
    "dias_desde_1_abril_hasta_caida",
    "temperatura_media_14_dias_antes_caida",
    "frio_7_2_acumulado_antes_caida",
]

HEAT_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + [
    "temp_media_historica_mes_07",
    "temp_media_historica_mes_08",
    "temp_media_historica_mes_09",
    "fecha_fin_frio_dia_del_ano",
    "fecha_fin_frio_dias_desde_1_abril",
]


def add_heat_date_features(
    df: pd.DataFrame,
    fecha_fin_frio_column: str,
) -> pd.DataFrame:
    result = df.copy()
    values = pd.to_datetime(result[fecha_fin_frio_column], errors="coerce")
    result["fecha_fin_frio_dia_del_ano"] = values.dt.dayofyear
    result["fecha_fin_frio_dias_desde_1_abril"] = [
        (
            (date_value - pd.Timestamp(year=int(season), month=4, day=1))
            / pd.Timedelta(days=1)
        )
        if pd.notna(date_value)
        else float("nan")
        for date_value, season in zip(values, result["temporada"], strict=False)
    ]
    return result


def cold_feature_columns() -> tuple[list[str], list[str]]:
    return COLD_NUMERIC_FEATURES, CATEGORICAL_COLD_FEATURES


def heat_feature_columns() -> tuple[list[str], list[str]]:
    return HEAT_NUMERIC_FEATURES, CATEGORICAL_HEAT_FEATURES

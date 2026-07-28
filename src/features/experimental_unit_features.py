from __future__ import annotations

import pandas as pd

from src.targets.accumulation import accumulate_between
from src.thermal_models import chill_hours_7_2


def days_from_april_first(date_value: pd.Timestamp, season: int) -> float:
    if pd.isna(date_value):
        return float("nan")
    april_first = pd.Timestamp(year=int(season), month=4, day=1)
    return float((date_value - april_first) / pd.Timedelta(days=1))


def add_experimental_unit_features(
    dataset: pd.DataFrame,
    temperatures: pd.DataFrame,
) -> pd.DataFrame:
    result = dataset.copy()
    result["dia_del_ano_caida_50"] = result["fecha_caida_hoja_50"].dt.dayofyear
    result["dias_desde_1_abril_hasta_caida"] = [
        days_from_april_first(row.fecha_caida_hoja_50, row.temporada)
        for row in result.itertuples(index=False)
    ]
    result["interaccion_entrada_salida"] = (
        result["codigo_tratamiento_entrada"].astype(str)
        + "__"
        + result["codigo_tratamiento_salida"].astype(str)
    )

    temp14: list[float] = []
    chill_before_leaf_fall: list[float] = []
    for row in result.itertuples(index=False):
        start_14 = row.fecha_caida_hoja_50 - pd.Timedelta(days=14)
        zone_window = temperatures.loc[
            (temperatures["codigo_lugar"] == row.codigo_lugar)
            & (temperatures["ds"] >= start_14)
            & (temperatures["ds"] < row.fecha_caida_hoja_50),
            "temperatura_c",
        ]
        temp14.append(float(zone_window.mean()) if not zone_window.empty else float("nan"))

        april_first = pd.Timestamp(year=int(row.temporada), month=4, day=1)
        chill_before_leaf_fall.append(
            accumulate_between(
                temperatures,
                row.codigo_lugar,
                april_first,
                row.fecha_caida_hoja_50,
                chill_hours_7_2,
            )
        )

    result["temperatura_media_14_dias_antes_caida"] = temp14
    result["frio_7_2_acumulado_antes_caida"] = chill_before_leaf_fall
    return result

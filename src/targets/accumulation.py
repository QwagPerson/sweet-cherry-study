from __future__ import annotations

from collections.abc import Callable

import pandas as pd


ThermalFunction = Callable[[pd.Series], pd.Series]


def accumulate_between(
    temperatures: pd.DataFrame,
    codigo_lugar: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    thermal_function: ThermalFunction,
) -> float:
    if pd.isna(start) or pd.isna(end) or end < start:
        return float("nan")

    zone = temperatures.loc[
        (temperatures["codigo_lugar"] == codigo_lugar)
        & (temperatures["ds"] >= start)
        & (temperatures["ds"] <= end),
        ["temperatura_c"],
    ]
    if zone.empty:
        return float("nan")

    return float(thermal_function(zone["temperatura_c"]).sum())

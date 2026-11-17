from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


ThermalFunction = Callable[[pd.Series], pd.Series]


def build_thermal_lookup(
    temperatures: pd.DataFrame,
    thermal_function: ThermalFunction,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code, zone in temperatures.sort_values("ds").groupby("codigo_lugar"):
        times = zone["ds"].astype("datetime64[ns]").astype("int64").to_numpy()
        increments = thermal_function(zone["temperatura_c"]).to_numpy(dtype=float)
        lookup[str(code)] = (times, increments)
    return lookup


def first_crossing_date_from_lookup(
    lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    codigo_lugar: str,
    start: pd.Timestamp,
    requirement: float,
) -> pd.Timestamp | pd.NaT:
    if pd.isna(start) or pd.isna(requirement):
        return pd.NaT

    if str(codigo_lugar) not in lookup:
        return pd.NaT

    times, increments = lookup[str(codigo_lugar)]
    start_idx = np.searchsorted(times, pd.Timestamp(start).value, side="left")
    if start_idx >= len(times):
        return pd.NaT

    requirement = max(float(requirement), 0.0)
    cumulative = np.cumsum(increments[start_idx:])
    reached = np.flatnonzero(cumulative >= requirement)
    if len(reached) == 0:
        return pd.NaT
    return pd.Timestamp(times[start_idx + int(reached[0])])


def first_crossing_date(
    temperatures: pd.DataFrame,
    codigo_lugar: str,
    start: pd.Timestamp,
    requirement: float,
    thermal_function: ThermalFunction,
) -> pd.Timestamp | pd.NaT:
    if pd.isna(start) or pd.isna(requirement):
        return pd.NaT

    requirement = max(float(requirement), 0.0)
    zone = temperatures.loc[
        (temperatures["codigo_lugar"] == codigo_lugar)
        & (temperatures["ds"] >= start),
        ["ds", "temperatura_c"],
    ].sort_values("ds")
    if zone.empty:
        return pd.NaT

    cumulative = thermal_function(zone["temperatura_c"]).cumsum()
    reached = zone.loc[cumulative >= requirement, "ds"]
    if reached.empty:
        return pd.NaT
    return reached.iloc[0]

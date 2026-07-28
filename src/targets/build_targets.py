from __future__ import annotations

import numpy as np
import pandas as pd

from src.thermal_models import THERMAL_PAIRS


def _build_cumulative_by_place(
    temperatures: pd.DataFrame,
    thermal_function,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cumulative: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code, zone in temperatures.sort_values("ds").groupby("codigo_lugar"):
        values = thermal_function(zone["temperatura_c"]).cumsum().to_numpy(dtype=float)
        times = zone["ds"].astype("int64").to_numpy()
        cumulative[str(code)] = (times, values)
    return cumulative


def _range_sum(
    cumulative: dict[str, tuple[np.ndarray, np.ndarray]],
    codigo_lugar: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    if pd.isna(start) or pd.isna(end) or end < start:
        return float("nan")

    if str(codigo_lugar) not in cumulative:
        return float("nan")

    times, values = cumulative[str(codigo_lugar)]
    start_ns = pd.Timestamp(start).value
    end_ns = pd.Timestamp(end).value
    if start_ns < times[0] or end_ns > times[-1]:
        return float("nan")

    end_idx = np.searchsorted(times, end_ns, side="right") - 1
    before_start_idx = np.searchsorted(times, start_ns, side="left") - 1
    if end_idx < 0 or end_idx >= len(values):
        return float("nan")

    before_start_value = values[before_start_idx] if before_start_idx >= 0 else 0.0
    return float(values[end_idx] - before_start_value)


def add_thermal_targets(
    phenology: pd.DataFrame,
    temperatures: pd.DataFrame,
) -> pd.DataFrame:
    dataset = phenology.copy()

    for pair_name, (cold_function, heat_function) in THERMAL_PAIRS.items():
        cold_cumulative = _build_cumulative_by_place(temperatures, cold_function)
        heat_cumulative = _build_cumulative_by_place(temperatures, heat_function)
        cold_target = f"target_frio__{pair_name}"
        heat_target = f"target_calor__{pair_name}"

        dataset[cold_target] = [
            _range_sum(
                cold_cumulative,
                row.codigo_lugar,
                row.fecha_caida_hoja_50,
                row.fecha_brotacion_camara_50,
            )
            for row in dataset.itertuples(index=False)
        ]
        dataset[heat_target] = [
            _range_sum(
                heat_cumulative,
                row.codigo_lugar,
                row.fecha_brotacion_camara_50,
                row.fecha_brotacion_campo_50,
            )
            for row in dataset.itertuples(index=False)
        ]

    return dataset

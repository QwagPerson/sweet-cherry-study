import numpy as np
import pandas as pd
from src.targets.build_targets import _build_cumulative_by_place, _range_sum
from src.data.temperature_support import temperature_support_by_unit


def test_microsecond_input_dates_produce_inclusive_hourly_targets():
    temperatures = pd.DataFrame({"codigo_lugar": "MEL", "temperatura_c": [-1., 7.2, 8.],
                                 "ds": pd.date_range("2025-05-01", periods=3, freq="h").astype("datetime64[us]")})
    cumulative = _build_cumulative_by_place(temperatures, lambda values: (values <= 7.2).astype(float))
    assert _range_sum(cumulative, "MEL", temperatures.ds.iloc[0], temperatures.ds.iloc[-1]) == 2.


def test_target_support_counts_keep_imputation_separate():
    times = pd.date_range("2025-05-01", periods=3, freq="h")
    temperatures = pd.DataFrame({"codigo_lugar": "MEL", "ds": times,
                                 "tipo_dato": ["observado", "interpolado", "climatologia"]})
    dataset = pd.DataFrame({"unidad_experimental_id": ["unit"], "codigo_lugar": ["MEL"],
                            "temporada": [2025], "fecha_caida_hoja_50": [times[0]],
                            "fecha_brotacion_camara_50": [times[1]], "fecha_brotacion_campo_50": [times[2]]})
    support = temperature_support_by_unit(dataset, temperatures).iloc[0]
    assert support.frio_observado_horas == support.frio_interpolado_horas == 1
    assert support.calor_interpolado_horas == support.calor_climatologia_horas == 1
    assert support.frio_observado_pct == 50.
    assert support.calor_faltante_horas == 0

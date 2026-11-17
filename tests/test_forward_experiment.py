import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiment_2024_to_2025.py"
spec = importlib.util.spec_from_file_location("forward_experiment", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def temperatures():
    frame = pd.DataFrame({
        "ds": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2021-01-01 01:00", "2022-01-01 00:00"]),
        "temperatura_c": [5., 7., 9., 99.], "tipo_dato": "observado", "codigo_lugar": "A",
        "lugar": "A", "condicion": "frio", "latitud": -34., "longitud": -71.,
    })
    frame["year"], frame["month"], frame["hour"] = frame.ds.dt.year, frame.ds.dt.month, frame.ds.dt.hour
    return frame


def test_missing_hour_cannot_use_current_or_future_year_donors():
    result, audit = experiment.past_year_temperatures(temperatures(), last_year=2021)
    row = result[result.ds.eq(pd.Timestamp("2021-01-01"))].iloc[0]
    assert row.temperatura_c == 5.
    assert row.source == "past_year_climatology"
    assert audit[audit.year.eq(2021)].donor_last_year.iloc[0] == 2020
    assert audit[audit.year.eq(2020)].missing_hours.iloc[0] > 0


def test_changing_future_observations_does_not_change_past_dataset():
    raw = temperatures()
    before, before_audit = experiment.past_year_temperatures(raw, last_year=2021)
    raw.loc[raw.year.eq(2022), "temperatura_c"] = -999.
    after, after_audit = experiment.past_year_temperatures(raw, last_year=2021)
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(before_audit, after_audit)

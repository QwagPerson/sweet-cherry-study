"""Fixed-model forward diagnostic: train 2024, evaluate 2025, never promote.

Reproduce the published retrospective result, then remove future-year climate
donors. Report conditional historical-weather and frozen 2024 Prophet weather
scenarios separately. The model specification was previously chosen using these
years, so this is a diagnostic, not a new independent model-selection test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from src.data.load_data import load_phenology, load_temperatures
from src.features.place_features import build_place_season_features
from src.features.experimental_unit_features import add_experimental_unit_features
from src.features.build_features import cold_feature_columns, heat_feature_columns, add_heat_date_features
from src.targets.build_targets import add_thermal_targets
from src.models.model_factory import make_model
from src.simulation.thresholds import build_thermal_lookup, first_crossing_date_from_lookup
from src.thermal_models import THERMAL_PAIRS

PAIR = "chill_10__gdh_4_5"
COLD_TARGET = f"target_frio__{PAIR}"
HEAT_TARGET = f"target_calor__{PAIR}"
DATE_COLS = ["fecha_caida_hoja_50", "fecha_brotacion_camara_50", "fecha_brotacion_campo_50"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def past_year_temperatures(raw, last_year=2025):
    """Use measured values; fill gaps only from measured strictly prior years.

    Seasonal median month/day/hour -> month/hour -> hour -> place median.
    The first year has no donors: retain its observations without synthetic fill.
    This is an isolated experiment, not a change to canonical silver policy.
    """
    measured = raw[raw.tipo_dato.eq("observado") & raw.year.le(last_year)].copy()
    frames, audit = [], []
    for code, zone in measured.groupby("codigo_lugar", sort=True):
        zone = zone.sort_values("ds")
        metadata = zone.iloc[0][["codigo_lugar", "lugar", "condicion", "latitud", "longitud"]].to_dict()
        for year in range(int(zone.year.min()), last_year + 1):
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h").as_unit("ns")
            actual = zone[zone.year.eq(year)].set_index("ds").temperatura_c.reindex(dates)
            original = actual.notna()
            donors = zone[zone.year.lt(year)].copy()
            if not donors.empty:
                assert donors.ds.max() < pd.Timestamp(year, 1, 1)
                donors["day"] = donors.ds.dt.day
                for keys, index in [
                    (["month", "day", "hour"], pd.MultiIndex.from_arrays([dates.month, dates.day, dates.hour])),
                    (["month", "hour"], pd.MultiIndex.from_arrays([dates.month, dates.hour])),
                    (["hour"], pd.Index(dates.hour)),
                ]:
                    climate = donors.groupby(keys).temperatura_c.median()
                    fill = pd.Series(climate.reindex(index).to_numpy(), index=dates)
                    actual = actual.fillna(fill)
                actual = actual.fillna(donors.temperatura_c.median())
            frame = pd.DataFrame({"ds": dates, "temperatura_c": actual.to_numpy(),
                                  "source": np.where(original, "observed", "past_year_climatology")})
            audit.append(dict(code=code, year=year, observed_hours=int(original.sum()),
                              imputed_hours=int((~original & actual.notna()).sum()),
                              missing_hours=int(actual.isna().sum()),
                              donor_last_year=int(donors.year.max()) if len(donors) else None))
            for key, value in metadata.items():
                frame[key] = value
            frame = frame.dropna(subset=["temperatura_c"])
            frame["year"], frame["month"], frame["hour"] = frame.ds.dt.year, frame.ds.dt.month, frame.ds.dt.hour
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audit)


def dataset(phenology, temperatures):
    features = build_place_season_features(phenology, temperatures)
    result = phenology.merge(features, on=["codigo_lugar", "temporada"], how="left", suffixes=("", "_lugar_feature"))
    result = result.drop(columns=["lugar_lugar_feature"], errors="ignore")
    result = add_experimental_unit_features(result, temperatures)
    return add_thermal_targets(result, temperatures)


def fit_models(train, name):
    assert train.temporada.eq(2024).all()
    cold_numeric, cold_cat = cold_feature_columns()
    heat_numeric, heat_cat = heat_feature_columns()
    cold = make_model(name, cold_numeric, cold_cat).fit(train, train[COLD_TARGET])
    heat_train = add_heat_date_features(train, "fecha_brotacion_camara_50")
    heat = make_model(name, heat_numeric, heat_cat).fit(heat_train, heat_train[HEAT_TARGET])
    return cold, heat


def predictions(models, test, temperatures):
    cold, heat = models
    cold_fn, heat_fn = THERMAL_PAIRS[PAIR]
    cold_lookup = build_thermal_lookup(temperatures, cold_fn)
    heat_lookup = build_thermal_lookup(temperatures, heat_fn)
    cold_req = cold.predict(test).to_numpy()
    cold_dates = [first_crossing_date_from_lookup(cold_lookup, row.codigo_lugar, row.fecha_caida_hoja_50, value)
                  for row, value in zip(test.itertuples(index=False), cold_req)]
    heat_test = test.copy()
    heat_test["estimated_cold_end"] = cold_dates
    heat_test = add_heat_date_features(heat_test, "estimated_cold_end")
    heat_req = heat.predict(heat_test).to_numpy()
    bud_dates = [first_crossing_date_from_lookup(heat_lookup, row.codigo_lugar, end, value)
                 for row, end, value in zip(test.itertuples(index=False), cold_dates, heat_req)]
    return pd.DataFrame({"unidad_experimental_id": test.unidad_experimental_id.to_numpy(),
                         "codigo_lugar": test.codigo_lugar.to_numpy(), "codigo_variedad": test.codigo_variedad.to_numpy(),
                         "pred_cold_requirement": cold_req, "pred_heat_requirement": heat_req,
                         "pred_cold_end": pd.to_datetime(cold_dates), "pred_budbreak": pd.to_datetime(bud_dates)})


def score(frame, test, scenario):
    frame = frame.copy()
    frame["actual_cold_end"] = test.fecha_brotacion_camara_50.to_numpy()
    frame["actual_budbreak"] = test.fecha_brotacion_campo_50.to_numpy()
    frame["cold_error_days"] = (frame.pred_cold_end - frame.actual_cold_end) / pd.Timedelta(days=1)
    frame["budbreak_error_days"] = (frame.pred_budbreak - frame.actual_budbreak) / pd.Timedelta(days=1)
    frame["scenario"] = scenario
    errors = frame.budbreak_error_days.dropna()
    cold_errors = frame.cold_error_days.dropna()
    metrics = dict(scenario=scenario, test_year=2025, n_test=len(test), valid_budbreak=len(errors),
                   coverage_percent=float(len(errors) / len(test) * 100),
                   mae_budbreak_days=float(errors.abs().mean()), rmse_budbreak_days=float(np.sqrt((errors ** 2).mean())),
                   bias_budbreak_days=float(errors.mean()), mae_cold_days=float(cold_errors.abs().mean()),
                   within_7_days_percent=float(errors.abs().le(7).mean() * 100))
    return metrics, frame


def forecast_2025(raw, out):
    from prophet import Prophet
    from prophet.serialize import model_to_json
    measured = raw[raw.tipo_dato.eq("observado") & raw.ds.lt(pd.Timestamp("2025-01-01"))].copy()
    frames, audit = [], []
    for code, history in measured.groupby("codigo_lugar", sort=True):
        print(f"Forecast {code}: observed through {history.ds.max()}", flush=True)
        assert history.ds.max() < pd.Timestamp("2025-01-01")
        model = Prophet(growth="flat", yearly_seasonality=True, daily_seasonality=True,
                        weekly_seasonality=False, uncertainty_samples=0)
        model.fit(history[["ds", "temperatura_c"]].rename(columns={"temperatura_c": "y"}), seed=2027)
        future = pd.DataFrame({"ds": pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h").as_unit("ns")})
        predicted = model.predict(future)
        frame = predicted[["ds", "yhat"]].rename(columns={"yhat": "temperatura_c"})
        frame["codigo_lugar"] = code
        frame["source"] = "forecast_frozen_before_2025"
        assert len(frame) == 8760 and np.isfinite(frame.temperatura_c).all()
        frames.append(frame)
        (out / f"temperature_{code}.json").write_text(model_to_json(model), encoding="utf-8")
        audit.append(dict(code=code, n_train=len(history), training_end=history.ds.max().isoformat(), forecast_hours=len(frame)))
    return pd.concat(frames, ignore_index=True), audit


def serialize_regressor(model, path):
    def value(x):
        if hasattr(x, "to_dict"):
            return x.to_dict()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x
    path.write_text(json.dumps({k: value(v) for k, v in model.__dict__.items()}, indent=2), encoding="utf-8")


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "models"
    model_dir.mkdir()
    raw = load_temperatures()
    phenology = load_phenology()
    phenology = phenology[phenology[DATE_COLS].notna().all(axis=1)].copy()
    assert set(phenology.temporada.unique()) == {2024, 2025}
    clean_temperatures, imputation_audit = past_year_temperatures(raw)
    imputation_audit.to_csv(output / "imputation_audit.csv", index=False)
    assert imputation_audit.loc[imputation_audit.donor_last_year.notna()].eval("donor_last_year < year").all()
    clean_temperatures.to_parquet(output / "temperatures_past_year_donors.parquet", index=False)
    print("Building dataset with strictly past-year temperature donors", flush=True)
    clean = dataset(phenology, clean_temperatures)
    assert clean[[COLD_TARGET, HEAT_TARGET]].notna().all().all()
    clean.to_csv(output / "dataset_past_year_donors.csv", index=False)
    train = clean[clean.temporada.eq(2024)].copy().reset_index(drop=True)
    test = clean[clean.temporada.eq(2025)].copy().reset_index(drop=True)
    assert len(train) == 409 and len(test) == 312
    assert set(train.unidad_experimental_id).isdisjoint(test.unidad_experimental_id)
    original = pd.read_csv(PROJECT / "data/processed/modeling_dataset.csv", parse_dates=DATE_COLS)
    original_train = original[original.temporada.eq(2024)].reset_index(drop=True)
    original_test = original[original.temporada.eq(2025)].reset_index(drop=True)
    raw_2025 = raw[raw.year.eq(2025)].copy()
    original_models = fit_models(original_train, "ridge")
    record, details = score(predictions(original_models, original_test, raw_2025), original_test, "original_ridge_actual_weather")
    assert abs(record["mae_budbreak_days"] - 14.739706) < 1e-4, record
    metrics, all_predictions = [record], [details]
    print("Reproduced original:", json.dumps(record), flush=True)
    forecast, forecast_audit = forecast_2025(raw, model_dir)
    forecast.to_parquet(output / "forecast_2025_frozen_2024.parquet", index=False)
    weather = clean_temperatures[clean_temperatures.year.eq(2025)].copy()
    leakage_checks = []
    for model_name in ["ridge", "media_global_requisito"]:
        models = fit_models(train, model_name)
        for stage, model in zip(["cold", "heat"], models):
            serialize_regressor(model, model_dir / f"{model_name}_{stage}.json")
        for weather_name, temperatures in [("actual_weather", weather), ("forecast_weather", forecast)]:
            scenario = f"past_donors_{model_name}_{weather_name}"
            prediction = predictions(models, test, temperatures)
            record, details = score(prediction, test, scenario)
            metrics.append(record)
            all_predictions.append(details)
            print(json.dumps(record), flush=True)
            if model_name == "ridge":
                changed = test.copy()
                changed["fecha_brotacion_camara_50"] = pd.Timestamp("2099-01-01")
                changed["fecha_brotacion_campo_50"] = pd.Timestamp("2099-12-31")
                check = predictions(models, changed, temperatures)
                pd.testing.assert_frame_equal(prediction, check)
                leakage_checks.append(dict(scenario=scenario, test_outcome_mutation_does_not_change_predictions=True))
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    detailed = pd.concat(all_predictions, ignore_index=True)
    detailed.to_csv(output / "predictions.csv", index=False)
    detailed.groupby(["scenario", "codigo_lugar"]).agg(n=("unidad_experimental_id", "size"),
        valid=("budbreak_error_days", "count"), mae_budbreak_days=("budbreak_error_days", lambda x: x.abs().mean()),
        mae_cold_days=("cold_error_days", lambda x: x.abs().mean())).to_csv(output / "metrics_by_location.csv")
    source = WORKSPACE / "sweet-cherry-data/data/silver/2026/temperaturas_horarias_por_lugar.csv"
    manifest = dict(train_year=2024, test_year=2025, n_train=len(train), n_test=len(test),
        pair=PAIR, ridge_alpha=10, temperature_input_sha256=sha(source), script_sha256=sha(Path(__file__)),
        forecast=forecast_audit, leakage_checks=leakage_checks,
        protocol="Fixed existing pair and Ridge; no tuning or winner selection on these results",
        imputation="Only original observations; missing hours filled by medians from strictly prior years, no interpolation. First year has no donors and keeps gaps.",
        forecast_protocol="Temperature Prophet flat, daily+annual, trained only with measured temperatures before 2025-01-01; frozen forecast for all 2025. Phenology prediction at the observed entry date; pre-entry 2025 temperatures are available inputs.",
        limitations=["Diagnostic, not independent model-selection validation: pair/Ridge/Prophet specification had already been examined using 2025 and later years",
                     "All historical trial treatments included; not a control-only front evaluation",
                     "Actual-weather scenario knows subsequent 2025 weather and is a conditional retrospective calculation",
                     "Past-year filling also changes reconstructed training targets; observed-only complete-case labels would be a different experiment",
                     "Phenology entries and planned treatment categories are treated as known at entry",
                     "No tuning, nested selection, or predictive-interval calibration; one held-out season only",
                     "No canonical model, report, temperature file or front was replaced"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())

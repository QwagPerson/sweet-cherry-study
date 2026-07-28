from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = pd.notna(y_true) & pd.notna(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask].astype(float) - y_pred[mask].astype(float))))


def r2_score_oof(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = pd.notna(y_true) & pd.notna(y_pred)
    if mask.sum() < 2:
        return float("nan")
    truth = y_true[mask].astype(float)
    pred = y_pred[mask].astype(float)
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def date_to_days(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    return dates.map(lambda value: value.toordinal() if pd.notna(value) else np.nan)


def date_error_days(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    actual_dates = pd.to_datetime(actual, errors="coerce")
    predicted_dates = pd.to_datetime(predicted, errors="coerce")
    return (predicted_dates - actual_dates) / pd.Timedelta(days=1)


def percent_abs_error_leq(errors_days: pd.Series, threshold_days: int) -> float:
    valid = errors_days.dropna()
    if valid.empty:
        return float("nan")
    return float((valid.abs() <= threshold_days).mean() * 100.0)

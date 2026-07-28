from __future__ import annotations

import pandas as pd


class MeanRegressor:
    def __init__(self, group_columns: list[str] | None = None) -> None:
        self.group_columns = group_columns or []
        self.global_mean_: float | None = None
        self.group_means_: pd.Series | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "MeanRegressor":
        y = pd.to_numeric(y, errors="coerce")
        self.global_mean_ = float(y.mean())
        if self.group_columns:
            frame = x[self.group_columns].copy()
            frame["_target"] = y.values
            self.group_means_ = frame.groupby(self.group_columns)["_target"].mean()
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        if self.global_mean_ is None:
            raise RuntimeError("El modelo no ha sido ajustado")

        prediction = pd.Series(self.global_mean_, index=x.index, dtype=float)
        if not self.group_columns or self.group_means_ is None:
            return prediction

        keys = x[self.group_columns].apply(tuple, axis=1)
        if len(self.group_columns) == 1:
            keys = x[self.group_columns[0]]
        mapped = keys.map(self.group_means_)
        return mapped.fillna(prediction).astype(float)

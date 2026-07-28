from __future__ import annotations

import numpy as np
import pandas as pd


class RidgeRegressor:
    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        alpha: float = 10.0,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.alpha = alpha
        self.numeric_medians_: pd.Series | None = None
        self.numeric_means_: pd.Series | None = None
        self.numeric_stds_: pd.Series | None = None
        self.category_levels_: dict[str, list[str]] = {}
        self.coef_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "RidgeRegressor":
        design = self._fit_transform_design(x)
        target = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)

        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("El modelo no ha sido ajustado")
        design = self._transform_design(x)
        return pd.Series(design @ self.coef_, index=x.index, dtype=float)

    def _fit_transform_design(self, x: pd.DataFrame) -> np.ndarray:
        numeric = x[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        self.numeric_medians_ = numeric.median().fillna(0.0)
        numeric = numeric.fillna(self.numeric_medians_)
        self.numeric_means_ = numeric.mean()
        self.numeric_stds_ = numeric.std(ddof=0).replace(0.0, 1.0).fillna(1.0)

        self.category_levels_ = {}
        for column in self.categorical_columns:
            levels = (
                x[column]
                .fillna("__missing__")
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
            self.category_levels_[column] = levels

        return self._transform_design(x)

    def _transform_design(self, x: pd.DataFrame) -> np.ndarray:
        if (
            self.numeric_medians_ is None
            or self.numeric_means_ is None
            or self.numeric_stds_ is None
        ):
            raise RuntimeError("El transformador no ha sido ajustado")

        parts = [np.ones((len(x), 1))]
        numeric = x[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(self.numeric_medians_)
        numeric = (numeric - self.numeric_means_) / self.numeric_stds_
        parts.append(numeric.to_numpy(dtype=float))

        for column in self.categorical_columns:
            values = x[column].fillna("__missing__").astype(str)
            for level in self.category_levels_[column]:
                parts.append((values == level).astype(float).to_numpy().reshape(-1, 1))

        return np.hstack(parts)

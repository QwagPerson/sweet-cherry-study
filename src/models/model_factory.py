from __future__ import annotations

from src.models.baselines import MeanRegressor
from src.models.ridge import RidgeRegressor


MODEL_NAMES = [
    "media_global_requisito",
    "media_por_variedad",
    "media_por_variedad_y_condicion",
    "ridge",
]


def make_model(
    model_name: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
):
    if model_name == "media_global_requisito":
        return MeanRegressor([])
    if model_name == "media_por_variedad":
        return MeanRegressor(["codigo_variedad"])
    if model_name == "media_por_variedad_y_condicion":
        return MeanRegressor(["codigo_variedad", "condicion"])
    if model_name == "ridge":
        return RidgeRegressor(numeric_columns, categorical_columns, alpha=10.0)
    raise ValueError(f"Modelo no soportado: {model_name}")

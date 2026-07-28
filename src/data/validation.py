from __future__ import annotations

import pandas as pd


def require_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"{source} no contiene columnas requeridas: {missing}")


def require_non_empty(df: pd.DataFrame, source: str) -> None:
    if df.empty:
        raise ValueError(f"{source} no tiene filas disponibles")


def require_no_nulls(df: pd.DataFrame, columns: list[str], source: str) -> None:
    null_counts = df[columns].isna().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"{source} contiene nulos no permitidos: {bad.to_dict()}")

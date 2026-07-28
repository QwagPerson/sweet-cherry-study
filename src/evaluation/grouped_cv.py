from __future__ import annotations

import numpy as np
import pandas as pd


def group_kfold_indices(
    df: pd.DataFrame,
    group_column: str,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = df[group_column].astype(str)
    group_sizes = groups.value_counts().sort_values(ascending=False)
    n_splits = min(n_splits, len(group_sizes))
    if n_splits < 2:
        raise ValueError(f"No hay suficientes grupos en {group_column}")

    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]
    for group, size in group_sizes.items():
        fold_id = int(np.argmin(fold_sizes))
        fold_groups[fold_id].append(group)
        fold_sizes[fold_id] += int(size)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for test_groups in fold_groups:
        is_test = groups.isin(test_groups).to_numpy()
        folds.append((np.flatnonzero(~is_test), np.flatnonzero(is_test)))
    return folds


def leave_one_group_out_indices(
    df: pd.DataFrame,
    group_column: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = df[group_column].astype(str)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for group in sorted(groups.unique()):
        is_test = (groups == group).to_numpy()
        folds.append((np.flatnonzero(~is_test), np.flatnonzero(is_test)))
    return folds


def make_validation_folds(
    df: pd.DataFrame,
    strategy: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if strategy == "GroupKFold_lugar_variedad_temporada":
        return group_kfold_indices(df, "grupo_lugar_variedad_temporada", n_splits=5)
    if strategy == "LeaveOneLocationOut":
        return leave_one_group_out_indices(df, "codigo_lugar")
    if strategy == "LeaveOneSeasonOut":
        return leave_one_group_out_indices(df, "temporada")
    if strategy == "LeaveOneVarietyOut":
        return leave_one_group_out_indices(df, "codigo_variedad")
    raise ValueError(f"Estrategia de validacion no soportada: {strategy}")


VALIDATION_STRATEGIES = [
    "GroupKFold_lugar_variedad_temporada",
    "LeaveOneLocationOut",
    "LeaveOneSeasonOut",
    "LeaveOneVarietyOut",
]

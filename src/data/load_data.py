from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schemas import (
    DATE_COLUMNS,
    PHENOLOGY_FILES,
    REQUIRED_PHENOLOGY_COLUMNS,
    REQUIRED_TEMPERATURE_COLUMNS,
    TEMPERATURE_FILE,
)
from src.data.validation import require_columns, require_non_empty, require_no_nulls


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SILVER_ROOT = (
    PROJECT_ROOT.parent / "sweet-cherry-data" / "data" / "silver" / "2026"
)


def load_phenology(silver_root: Path = DEFAULT_SILVER_ROOT) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file_name in PHENOLOGY_FILES:
        path = silver_root / file_name
        df = pd.read_csv(path)
        require_columns(df, REQUIRED_PHENOLOGY_COLUMNS, str(path))
        frames.append(df)

    phenology = pd.concat(frames, ignore_index=True)
    require_non_empty(phenology, "fenologia")

    for column in DATE_COLUMNS:
        phenology[column] = pd.to_datetime(
            phenology[column],
            errors="coerce",
            format="mixed",
        )

    phenology["temporada"] = phenology["temporada"].astype(int)
    phenology["ue"] = phenology["ue"].astype(str).str.strip()
    phenology["unidad_experimental_id"] = (
        phenology["temporada"].astype(str)
        + "_"
        + phenology["codigo_lugar"].astype(str)
        + "_"
        + phenology["codigo_variedad"].astype(str)
        + "_"
        + phenology["ue"]
        + "_"
        + phenology["codigo_tratamiento_entrada"].astype(str)
        + "_"
        + phenology["codigo_tratamiento_salida"].astype(str)
    )
    phenology["grupo_lugar_variedad_temporada"] = (
        phenology["codigo_lugar"].astype(str)
        + "_"
        + phenology["codigo_variedad"].astype(str)
        + "_"
        + phenology["temporada"].astype(str)
    )
    return phenology


def load_temperatures(silver_root: Path = DEFAULT_SILVER_ROOT) -> pd.DataFrame:
    path = silver_root / TEMPERATURE_FILE
    temperatures = pd.read_csv(path)
    require_columns(temperatures, REQUIRED_TEMPERATURE_COLUMNS, str(path))
    require_non_empty(temperatures, "temperaturas")

    temperatures["ds"] = pd.to_datetime(temperatures["ds"], errors="coerce")
    temperatures["temperatura_c"] = pd.to_numeric(
        temperatures["temperatura_c"], errors="coerce"
    )
    require_no_nulls(temperatures, ["codigo_lugar", "ds", "temperatura_c"], str(path))
    temperatures["year"] = temperatures["ds"].dt.year
    temperatures["month"] = temperatures["ds"].dt.month
    temperatures["hour"] = temperatures["ds"].dt.hour
    return temperatures

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_data import PROJECT_ROOT, load_phenology, load_temperatures
from src.data.variable_dictionary import build_variable_dictionary
from src.features.experimental_unit_features import add_experimental_unit_features
from src.features.place_features import build_place_season_features
from src.targets.build_targets import add_thermal_targets


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "modeling_dataset.csv"
VARIABLE_DICTIONARY_PATH = PROCESSED_DIR / "variable_dictionary.csv"


def build_modeling_dataset() -> pd.DataFrame:
    phenology = load_phenology()
    temperatures = load_temperatures()

    complete_dates = phenology[
        [
            "fecha_caida_hoja_50",
            "fecha_brotacion_camara_50",
            "fecha_brotacion_campo_50",
        ]
    ].notna().all(axis=1)
    excluded = int((~complete_dates).sum())
    phenology = phenology.loc[complete_dates].copy()

    place_features = build_place_season_features(phenology, temperatures)
    dataset = phenology.merge(
        place_features,
        on=["codigo_lugar", "temporada"],
        how="left",
        suffixes=("", "_lugar_feature"),
    )
    if "lugar_lugar_feature" in dataset.columns:
        dataset = dataset.drop(columns=["lugar_lugar_feature"])

    dataset = add_experimental_unit_features(dataset, temperatures)
    dataset = add_thermal_targets(dataset, temperatures)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET_PATH, index=False)
    variable_dictionary = build_variable_dictionary(dataset)
    variable_dictionary.to_csv(VARIABLE_DICTIONARY_PATH, index=False)

    print(f"Dataset limpio: {DATASET_PATH}")
    print(f"Diccionario de variables: {VARIABLE_DICTIONARY_PATH}")
    print(f"Filas modelables: {len(dataset)}")
    print(f"Filas excluidas por fechas incompletas: {excluded}")
    print(
        "Grupos lugar-variedad-temporada: "
        f"{dataset['grupo_lugar_variedad_temporada'].nunique()}"
    )
    return dataset


def main() -> None:
    build_modeling_dataset()


if __name__ == "__main__":
    main()

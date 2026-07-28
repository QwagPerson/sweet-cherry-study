from __future__ import annotations

import pandas as pd

from src.features.build_features import (
    CATEGORICAL_COLD_FEATURES,
    CATEGORICAL_HEAT_FEATURES,
    COLD_NUMERIC_FEATURES,
    HEAT_NUMERIC_FEATURES,
)
from src.thermal_models import THERMAL_PAIRS


BASE_VARIABLES = [
    (
        "unidad_experimental_id",
        "Identificador estable de unidad experimental y temporada.",
        "id",
        "consolidado_fechas_*_limpio.csv",
        "unidad_experimental",
        "ambos",
        "inicio_modelamiento",
        "sin_imputacion",
        "bajo",
        False,
    ),
    (
        "grupo_lugar_variedad_temporada",
        "Grupo minimo para validacion agrupada.",
        "categoria",
        "derivado",
        "unidad_experimental",
        "ambos",
        "inicio_modelamiento",
        "sin_imputacion",
        "bajo",
        False,
    ),
    (
        "fecha_caida_hoja_50",
        "Fecha de 50% de caida de hojas; marca entrada operacional a latencia.",
        "fecha",
        "consolidado_fechas_*_limpio.csv",
        "unidad_experimental",
        "frio",
        "observada antes de iniciar prediccion de frio",
        "excluir si falta",
        "medio si se usa antes de observarse",
        True,
    ),
    (
        "fecha_brotacion_camara_50",
        "Fecha de cumplimiento del criterio de brotacion en camara.",
        "fecha",
        "consolidado_fechas_*_limpio.csv",
        "unidad_experimental",
        "calor",
        "solo entrenamiento/evaluacion; en produccion se reemplaza por estimacion",
        "excluir si falta",
        "alto si se entrega real en produccion",
        False,
    ),
    (
        "fecha_brotacion_campo_50",
        "Fecha observada de brotacion en campo.",
        "fecha",
        "consolidado_fechas_*_limpio.csv",
        "unidad_experimental",
        "evaluacion",
        "solo evaluacion",
        "excluir si falta",
        "alto si se usa como feature",
        False,
    ),
]


def _record(
    name: str,
    description: str,
    unit: str,
    source: str,
    level: str,
    model_target: str,
    availability: str,
    imputation: str,
    leakage: str,
    production: bool,
    completeness: float,
) -> dict[str, object]:
    return {
        "nombre": name,
        "descripcion": description,
        "unidad": unit,
        "fuente": source,
        "nivel": level,
        "modelo_destino": model_target,
        "momento_disponibilidad": availability,
        "porcentaje_completitud": round(float(completeness), 2),
        "metodo_imputacion": imputation,
        "riesgo_leakage": leakage,
        "uso_en_produccion": production,
    }


def build_variable_dictionary(dataset: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for item in BASE_VARIABLES:
        name = item[0]
        completeness = dataset[name].notna().mean() * 100 if name in dataset else 0.0
        records.append(_record(*item, completeness=completeness))

    for column in sorted(set(COLD_NUMERIC_FEATURES + HEAT_NUMERIC_FEATURES)):
        target = "ambos"
        if column in COLD_NUMERIC_FEATURES and column not in HEAT_NUMERIC_FEATURES:
            target = "frio"
        if column in HEAT_NUMERIC_FEATURES and column not in COLD_NUMERIC_FEATURES:
            target = "calor"
        records.append(
            _record(
                column,
                "Feature numerica derivada para modelamiento.",
                "variable",
                "temperaturas_horarias_por_lugar.csv / consolidado fechas",
                "lugar" if "historica" in column or column in {"latitud", "longitud"} else "unidad_experimental",
                target,
                "antes de prediccion operacional",
                "mediana ajustada dentro de fold en Ridge si falta",
                "bajo si se calcula sin usar la temporada objetivo",
                True,
                dataset[column].notna().mean() * 100 if column in dataset else 0.0,
            )
        )

    for column in sorted(set(CATEGORICAL_COLD_FEATURES + CATEGORICAL_HEAT_FEATURES)):
        target = "ambos"
        if column in CATEGORICAL_COLD_FEATURES and column not in CATEGORICAL_HEAT_FEATURES:
            target = "frio"
        if column in CATEGORICAL_HEAT_FEATURES and column not in CATEGORICAL_COLD_FEATURES:
            target = "calor"
        records.append(
            _record(
                column,
                "Feature categorica disponible para la unidad experimental.",
                "categoria",
                "consolidado_fechas_*_limpio.csv / temperaturas",
                "variedad" if "variedad" in column else "unidad_experimental",
                target,
                "antes de prediccion operacional",
                "categoria __missing__ dentro del modelo si falta",
                "medio para tratamientos no observados",
                True,
                dataset[column].notna().mean() * 100 if column in dataset else 0.0,
            )
        )

    for pair_name in THERMAL_PAIRS:
        for stage in ["frio", "calor"]:
            column = f"target_{stage}__{pair_name}"
            records.append(
                _record(
                    column,
                    f"Requisito termico derivado de {stage} para pareja {pair_name}.",
                    "unidades termicas acumuladas",
                    "fechas fenologicas + temperaturas horarias",
                    "unidad_experimental",
                    stage,
                    "solo entrenamiento/evaluacion",
                    "excluir si falta",
                    "alto si se usa fuera del target de entrenamiento",
                    False,
                    dataset[column].notna().mean() * 100 if column in dataset else 0.0,
                )
            )

    return pd.DataFrame(records).sort_values(["modelo_destino", "nombre"])

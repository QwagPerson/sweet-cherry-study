from __future__ import annotations

import pandas as pd

from src.evaluation.grouped_cv import make_validation_folds
from src.evaluation.metrics import (
    date_error_days,
    date_to_days,
    date_to_season_days,
    mae,
    percent_abs_error_leq,
    r2_score_oof,
)
from src.features.build_features import (
    add_heat_date_features,
    cold_feature_columns,
    heat_feature_columns,
)
from src.models.model_factory import make_model
from src.simulation.thresholds import (
    build_thermal_lookup,
    first_crossing_date_from_lookup,
)
from src.thermal_models import THERMAL_PAIRS


def evaluate_pair_model(
    dataset: pd.DataFrame,
    temperatures: pd.DataFrame,
    pair_name: str,
    model_name: str,
    validation_strategy: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    cold_function, heat_function = THERMAL_PAIRS[pair_name]
    cold_lookup = build_thermal_lookup(temperatures, cold_function)
    heat_lookup = build_thermal_lookup(temperatures, heat_function)
    cold_target = f"target_frio__{pair_name}"
    heat_target = f"target_calor__{pair_name}"

    work = dataset.loc[
        dataset[[cold_target, heat_target]].notna().all(axis=1)
    ].reset_index(drop=True)
    cold_numeric, cold_categorical = cold_feature_columns()
    heat_numeric, heat_categorical = heat_feature_columns()

    predictions: list[pd.DataFrame] = []
    folds = make_validation_folds(work, validation_strategy)

    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        train = work.iloc[train_idx].copy()
        test = work.iloc[test_idx].copy()

        cold_model = make_model(model_name, cold_numeric, cold_categorical)
        cold_model.fit(train, train[cold_target])
        cold_requirement_pred = cold_model.predict(test)

        fecha_fin_frio_estimada = [
            first_crossing_date_from_lookup(
                cold_lookup,
                row.codigo_lugar,
                row.fecha_caida_hoja_50,
                requirement,
            )
            for row, requirement in zip(
                test.itertuples(index=False),
                cold_requirement_pred,
                strict=False,
            )
        ]

        heat_train = add_heat_date_features(
            train,
            "fecha_brotacion_camara_50",
        )
        heat_test = test.copy()
        heat_test["fecha_fin_frio_estimada"] = fecha_fin_frio_estimada
        heat_test = add_heat_date_features(heat_test, "fecha_fin_frio_estimada")

        heat_model = make_model(model_name, heat_numeric, heat_categorical)
        heat_model.fit(heat_train, train[heat_target])
        heat_requirement_pred = heat_model.predict(heat_test)

        fecha_brotacion_estimada = [
            first_crossing_date_from_lookup(
                heat_lookup,
                row.codigo_lugar,
                estimated_cold_date,
                requirement,
            )
            for row, estimated_cold_date, requirement in zip(
                test.itertuples(index=False),
                fecha_fin_frio_estimada,
                heat_requirement_pred,
                strict=False,
            )
        ]

        fold_predictions = test[
            [
                "unidad_experimental_id",
                "grupo_lugar_variedad_temporada",
                "temporada",
                "codigo_lugar",
                "lugar",
                "codigo_variedad",
                "variedad",
                "codigo_tratamiento_entrada",
                "codigo_tratamiento_salida",
                "fecha_caida_hoja_50",
                "fecha_brotacion_camara_50",
                "fecha_brotacion_campo_50",
                cold_target,
                heat_target,
            ]
        ].copy()
        fold_predictions["thermal_pair"] = pair_name
        fold_predictions["model_name"] = model_name
        fold_predictions["validation_strategy"] = validation_strategy
        fold_predictions["fold_id"] = fold_id
        fold_predictions["pred_requisito_frio"] = cold_requirement_pred.values
        fold_predictions["pred_requisito_calor"] = heat_requirement_pred.values
        fold_predictions["pred_fecha_fin_frio"] = fecha_fin_frio_estimada
        fold_predictions["pred_fecha_brotacion"] = fecha_brotacion_estimada
        predictions.append(fold_predictions)

    oof = pd.concat(predictions, ignore_index=True)
    error_fin_frio = date_error_days(
        oof["fecha_brotacion_camara_50"], oof["pred_fecha_fin_frio"]
    )
    error_brotacion = date_error_days(
        oof["fecha_brotacion_campo_50"], oof["pred_fecha_brotacion"]
    )
    fecha_fin_frio_valida = oof["pred_fecha_fin_frio"].notna()
    fecha_brotacion_valida = oof["pred_fecha_brotacion"].notna()

    metrics = {
        "thermal_pair": pair_name,
        "model_name": model_name,
        "validation_strategy": validation_strategy,
        "n_observaciones": len(oof),
        "n_pred_fecha_fin_frio_validas": int(fecha_fin_frio_valida.sum()),
        "n_pred_fecha_brotacion_validas": int(fecha_brotacion_valida.sum()),
        "porcentaje_pred_fecha_brotacion_valida": float(
            fecha_brotacion_valida.mean() * 100.0
        ),
        "r2_requisito_frio": r2_score_oof(
            oof[cold_target], oof["pred_requisito_frio"]
        ),
        "mae_requisito_frio": mae(oof[cold_target], oof["pred_requisito_frio"]),
        "r2_requisito_calor": r2_score_oof(
            oof[heat_target], oof["pred_requisito_calor"]
        ),
        "mae_requisito_calor": mae(oof[heat_target], oof["pred_requisito_calor"]),
        "r2_fecha_fin_frio": r2_score_oof(
            date_to_season_days(
                oof["fecha_brotacion_camara_50"],
                oof["temporada"],
            ),
            date_to_season_days(oof["pred_fecha_fin_frio"], oof["temporada"]),
        ),
        "r2_fecha_fin_frio_calendario_absoluto": r2_score_oof(
            date_to_days(oof["fecha_brotacion_camara_50"]),
            date_to_days(oof["pred_fecha_fin_frio"]),
        ),
        "mae_fecha_fin_frio_dias": float(error_fin_frio.abs().mean()),
        "r2_fecha_brotacion": r2_score_oof(
            date_to_season_days(
                oof["fecha_brotacion_campo_50"],
                oof["temporada"],
            ),
            date_to_season_days(oof["pred_fecha_brotacion"], oof["temporada"]),
        ),
        "r2_fecha_brotacion_calendario_absoluto": r2_score_oof(
            date_to_days(oof["fecha_brotacion_campo_50"]),
            date_to_days(oof["pred_fecha_brotacion"]),
        ),
        "mae_fecha_brotacion_dias": float(error_brotacion.abs().mean()),
        "porcentaje_error_menor_igual_3_dias": percent_abs_error_leq(
            error_brotacion, 3
        ),
        "porcentaje_error_menor_igual_5_dias": percent_abs_error_leq(
            error_brotacion, 5
        ),
        "porcentaje_error_menor_igual_7_dias": percent_abs_error_leq(
            error_brotacion, 7
        ),
    }
    oof["error_fecha_fin_frio_dias"] = error_fin_frio
    oof["error_fecha_brotacion_dias"] = error_brotacion
    return metrics, oof

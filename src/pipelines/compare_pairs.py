from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.load_data import PROJECT_ROOT, load_temperatures
from src.evaluation.evaluate_pair import evaluate_pair_model
from src.evaluation.grouped_cv import VALIDATION_STRATEGIES
from src.models.model_factory import MODEL_NAMES
from src.pipelines.build_dataset import DATASET_PATH, build_modeling_dataset
from src.thermal_models import THERMAL_PAIRS


REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics_by_pair.csv"
OOF_PREDICTIONS_PATH = REPORTS_DIR / "oof_predictions.csv"
ERROR_BY_SEGMENT_PATH = REPORTS_DIR / "error_by_segment.csv"
SELECTED_PAIR_PATH = REPORTS_DIR / "selected_pair.json"
LIMITATIONS_PATH = REPORTS_DIR / "leakage_and_limitations.md"


def load_or_build_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        return pd.read_csv(
            DATASET_PATH,
            parse_dates=[
                "fecha_caida_hoja_50",
                "fecha_brotacion_camara_50",
                "fecha_brotacion_campo_50",
            ],
        )
    return build_modeling_dataset()


def build_error_by_segment(selected_oof: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for segment in [
        "codigo_lugar",
        "codigo_variedad",
        "codigo_tratamiento_entrada",
        "codigo_tratamiento_salida",
        "temporada",
    ]:
        summary = (
            selected_oof.groupby(segment, dropna=False)
            .agg(
                n=("unidad_experimental_id", "size"),
                mae_fecha_brotacion_dias=(
                    "error_fecha_brotacion_dias",
                    lambda values: values.abs().mean(),
                ),
                bias_fecha_brotacion_dias=("error_fecha_brotacion_dias", "mean"),
            )
            .reset_index()
            .rename(columns={segment: "segmento_valor"})
        )
        summary.insert(0, "segmento", segment)
        records.append(summary)
    return pd.concat(records, ignore_index=True)


def write_limitations() -> None:
    LIMITATIONS_PATH.write_text(
        """# Fugas y limitaciones de generalizacion

- La validacion critica es LeaveOneLocationOut; GroupKFold por lugar-variedad-temporada no reemplaza esa prueba.
- El modelo de calor se entrena con la fecha real de termino de frio, pero en evaluacion end-to-end predice usando la fecha estimada por el modelo de frio.
- Las features climaticas historicas se calculan con anos previos a la temporada de la muestra. Si no existe historia previa, se usa historia de otros anos como respaldo.
- No se usa el nombre del lugar como feature de modelamiento; se conservan latitud, longitud y condicion agroclimatica.
- El modelo `dynamic_proxy__gdh_4_5` no es una implementacion biofisica completa de porciones de frio; es un proxy para comparar la interfaz.
- XGBoost, GAM y modelos jerarquicos quedan preparados como extensiones, pero no se ejecutan en esta version para evitar dependencias no instaladas.
- La incertidumbre operacional todavia no esta calibrada; esta iteracion solo produce error historico out-of-fold.
""",
        encoding="utf-8",
    )


def main() -> None:
    dataset = load_or_build_dataset()
    temperatures = load_temperatures()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []

    for strategy in VALIDATION_STRATEGIES:
        for pair_name in THERMAL_PAIRS:
            for model_name in MODEL_NAMES:
                metrics, predictions = evaluate_pair_model(
                    dataset,
                    temperatures,
                    pair_name,
                    model_name,
                    strategy,
                )
                all_metrics.append(metrics)
                all_predictions.append(predictions)
                print(
                    f"{strategy} | {pair_name} | {model_name}: "
                    f"MAE brotacion={metrics['mae_fecha_brotacion_dias']:.2f} dias"
                )

    metrics_df = pd.DataFrame(all_metrics).sort_values(
        ["validation_strategy", "mae_fecha_brotacion_dias"]
    )
    oof = pd.concat(all_predictions, ignore_index=True)

    metrics_df.to_csv(METRICS_PATH, index=False)
    oof.to_csv(OOF_PREDICTIONS_PATH, index=False)

    selection_pool = metrics_df.loc[
        metrics_df["validation_strategy"] == "GroupKFold_lugar_variedad_temporada"
    ].sort_values("mae_fecha_brotacion_dias")
    eligible = selection_pool.loc[
        selection_pool["porcentaje_pred_fecha_brotacion_valida"] >= 95.0
    ]
    if not eligible.empty:
        selection_pool = eligible.sort_values("mae_fecha_brotacion_dias")
    selected = selection_pool.iloc[0].to_dict()
    SELECTED_PAIR_PATH.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    selected_oof = oof.loc[
        (oof["thermal_pair"] == selected["thermal_pair"])
        & (oof["model_name"] == selected["model_name"])
        & (oof["validation_strategy"] == selected["validation_strategy"])
    ].copy()
    build_error_by_segment(selected_oof).to_csv(ERROR_BY_SEGMENT_PATH, index=False)
    write_limitations()

    print("\nMejor pareja inicial segun MAE de fecha de brotacion:")
    print(json.dumps(selected, indent=2, ensure_ascii=False))
    print(f"\nMetricas: {METRICS_PATH}")
    print(f"Predicciones OOF: {OOF_PREDICTIONS_PATH}")
    print(f"Error por segmento: {ERROR_BY_SEGMENT_PATH}")
    print(f"Limitaciones: {LIMITATIONS_PATH}")


if __name__ == "__main__":
    main()

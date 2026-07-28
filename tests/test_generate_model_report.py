from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_model_report import (
    compute_prediction_metrics,
    date_to_season_days,
    date_to_days,
    r2_score_days,
    r2_score_season_days,
    generate_report,
    load_inputs,
    percent_within_abs_error,
    require_columns,
)


def write_minimal_reports(root: Path, mae_value: float = 1.0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        [
            {
                "thermal_pair": "pair_a",
                "model_name": "ridge",
                "validation_strategy": "GroupKFold_lugar_variedad_temporada",
                "n_observaciones": 2,
                "n_pred_fecha_fin_frio_validas": 2,
                "n_pred_fecha_brotacion_validas": 2,
                "porcentaje_pred_fecha_brotacion_valida": 100.0,
                "r2_requisito_frio": 1.0,
                "mae_requisito_frio": 0.0,
                "r2_requisito_calor": 1.0,
                "mae_requisito_calor": 0.0,
                "r2_fecha_fin_frio": 1.0,
                "mae_fecha_fin_frio_dias": 1.0,
                "r2_fecha_brotacion": 1.0,
                "mae_fecha_brotacion_dias": mae_value,
                "porcentaje_error_menor_igual_3_dias": 100.0,
                "porcentaje_error_menor_igual_5_dias": 100.0,
                "porcentaje_error_menor_igual_7_dias": 100.0,
            }
        ]
    )
    oof = pd.DataFrame(
        [
            {
                "unidad_experimental_id": "u1",
                "grupo_lugar_variedad_temporada": "g1",
                "temporada": 2024,
                "codigo_lugar": "L1",
                "lugar": "Lugar 1",
                "codigo_variedad": "V1",
                "variedad": "Variedad 1",
                "codigo_tratamiento_entrada": "E1",
                "codigo_tratamiento_salida": "S1",
                "fecha_caida_hoja_50": "2024-05-01",
                "fecha_brotacion_camara_50": "2024-07-01",
                "fecha_brotacion_campo_50": "2024-09-01",
                "target_frio__pair_a": 10.0,
                "target_calor__pair_a": 100.0,
                "thermal_pair": "pair_a",
                "model_name": "ridge",
                "validation_strategy": "GroupKFold_lugar_variedad_temporada",
                "fold_id": 1,
                "pred_requisito_frio": 10.0,
                "pred_requisito_calor": 100.0,
                "pred_fecha_fin_frio": "2024-07-02",
                "pred_fecha_brotacion": "2024-09-02",
                "error_fecha_fin_frio_dias": 1.0,
                "error_fecha_brotacion_dias": 1.0,
            },
            {
                "unidad_experimental_id": "u2",
                "grupo_lugar_variedad_temporada": "g2",
                "temporada": 2025,
                "codigo_lugar": "L2",
                "lugar": "Lugar 2",
                "codigo_variedad": "V2",
                "variedad": "Variedad 2",
                "codigo_tratamiento_entrada": "E2",
                "codigo_tratamiento_salida": "S2",
                "fecha_caida_hoja_50": "2025-05-01",
                "fecha_brotacion_camara_50": "2025-07-01",
                "fecha_brotacion_campo_50": "2025-09-01",
                "target_frio__pair_a": 20.0,
                "target_calor__pair_a": 200.0,
                "thermal_pair": "pair_a",
                "model_name": "ridge",
                "validation_strategy": "GroupKFold_lugar_variedad_temporada",
                "fold_id": 2,
                "pred_requisito_frio": 20.0,
                "pred_requisito_calor": 200.0,
                "pred_fecha_fin_frio": "2025-06-30",
                "pred_fecha_brotacion": "2025-08-31",
                "error_fecha_fin_frio_dias": -1.0,
                "error_fecha_brotacion_dias": -1.0,
            },
        ]
    )
    segments = pd.DataFrame(
        [
            {
                "segmento": "codigo_lugar",
                "segmento_valor": "L1",
                "n": 1,
                "mae_fecha_brotacion_dias": 1.0,
                "bias_fecha_brotacion_dias": 1.0,
            }
        ]
    )
    selected = {
        "thermal_pair": "pair_a",
        "model_name": "ridge",
        "validation_strategy": "GroupKFold_lugar_variedad_temporada",
        "n_observaciones": 2,
        "mae_fecha_brotacion_dias": mae_value,
    }
    metrics.to_csv(root / "metrics_by_pair.csv", index=False)
    oof.to_csv(root / "oof_predictions.csv", index=False)
    segments.to_csv(root / "error_by_segment.csv", index=False)
    (root / "selected_pair.json").write_text(json.dumps(selected), encoding="utf-8")
    (root / "leakage_and_limitations.md").write_text("# Riesgos\n\n- Prueba", encoding="utf-8")


def test_load_inputs_and_columns(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    write_minimal_reports(reports)
    data = load_inputs(reports)
    assert len(data.metrics) == 1
    assert len(data.oof) == 2
    require_columns(data.metrics, ["thermal_pair", "model_name"], "metrics")


def test_require_columns_detects_missing() -> None:
    with pytest.raises(ValueError):
        require_columns(pd.DataFrame({"a": [1]}), ["a", "b"], "dummy")


def test_mae_and_percent_within_7_days() -> None:
    errors = pd.Series([-8.0, -7.0, 0.0, 6.5, 10.0])
    assert percent_within_abs_error(errors, 7) == pytest.approx(60.0)
    frame = pd.DataFrame(
        {
            "actual": ["2024-01-10", "2024-01-20"],
            "pred": ["2024-01-12", "2024-01-19"],
            "err": [2.0, -1.0],
        }
    )
    metrics = compute_prediction_metrics(frame, "actual", "pred", "err")
    assert metrics["mae_dias"] == pytest.approx(1.5)


def test_safe_date_to_days() -> None:
    values = date_to_days(pd.Series(["2024-01-01 12:00:00", "bad-date", None]))
    assert values.notna().sum() == 1
    assert np.isfinite(values.dropna().iloc[0])
    season_values = date_to_season_days(
        pd.Series(["2024-04-02 12:00:00", "bad-date"]),
        pd.Series([2024, 2024]),
    )
    assert season_values.iloc[0] == pytest.approx(1.5)
    assert pd.isna(season_values.iloc[1])


def test_season_r2_avoids_calendar_year_inflation() -> None:
    actual = pd.Series(["2024-09-01", "2026-09-10"])
    predicted = pd.Series(["2024-09-20", "2026-08-20"])
    seasons = pd.Series([2024, 2026])

    calendar_r2 = r2_score_days(actual, predicted)
    season_r2 = r2_score_season_days(actual, predicted, seasons)

    assert calendar_r2 > 0.99
    assert season_r2 < 0


def test_inconsistency_detection_and_html_generation(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    write_minimal_reports(reports, mae_value=99.0)
    output = reports / "model_evaluation_report.html"
    generate_report(reports, output)
    html = output.read_text(encoding="utf-8")
    assert output.exists()
    assert output.stat().st_size > 1000
    assert "Evaluación del modelamiento térmico" in html
    assert "Inconsistencia" in html


def test_html_generation_success(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    write_minimal_reports(reports)
    output = reports / "model_evaluation_report.html"
    generate_report(reports, output)
    assert output.exists()
    assert output.stat().st_size > 1000

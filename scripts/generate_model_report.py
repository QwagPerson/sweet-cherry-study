from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_OUTPUT_PATH = DEFAULT_REPORTS_DIR / "model_evaluation_report.html"
MIN_SEGMENT_SUPPORT = 20
CONSISTENCY_TOLERANCE = 1e-6

REQUIRED_FILES = {
    "metrics": "metrics_by_pair.csv",
    "oof": "oof_predictions.csv",
    "segments": "error_by_segment.csv",
    "selected": "selected_pair.json",
    "limitations": "leakage_and_limitations.md",
}

REQUIRED_METRICS_COLUMNS = [
    "thermal_pair",
    "model_name",
    "validation_strategy",
    "n_observaciones",
    "r2_fecha_fin_frio",
    "mae_fecha_fin_frio_dias",
    "r2_fecha_brotacion",
    "mae_fecha_brotacion_dias",
    "porcentaje_error_menor_igual_3_dias",
    "porcentaje_error_menor_igual_5_dias",
    "porcentaje_error_menor_igual_7_dias",
]

OPTIONAL_METRICS_COLUMNS = [
    "n_pred_fecha_fin_frio_validas",
    "n_pred_fecha_brotacion_validas",
    "porcentaje_pred_fecha_brotacion_valida",
]

REQUIRED_OOF_COLUMNS = [
    "unidad_experimental_id",
    "thermal_pair",
    "model_name",
    "validation_strategy",
    "fold_id",
    "fecha_brotacion_camara_50",
    "fecha_brotacion_campo_50",
    "pred_fecha_fin_frio",
    "pred_fecha_brotacion",
    "error_fecha_fin_frio_dias",
    "error_fecha_brotacion_dias",
]

REQUIRED_SEGMENT_COLUMNS = [
    "segmento",
    "segmento_valor",
    "n",
    "mae_fecha_brotacion_dias",
    "bias_fecha_brotacion_dias",
]

SELECTED_KEYS = ["thermal_pair", "model_name", "validation_strategy"]
IDENTIFIER_COLUMNS = [
    "unidad_experimental_id",
    "grupo_lugar_variedad_temporada",
    "fold_id",
    "temporada",
    "codigo_lugar",
    "lugar",
    "codigo_variedad",
    "variedad",
    "codigo_tratamiento_entrada",
    "codigo_tratamiento_salida",
]


@dataclass
class InputData:
    metrics: pd.DataFrame
    oof: pd.DataFrame
    segments: pd.DataFrame
    selected: dict[str, Any]
    limitations_markdown: str
    file_inventory: pd.DataFrame
    schemas: dict[str, dict[str, Any]]
    warnings: list[str]


def warn(warnings: list[str], message: str) -> None:
    warnings.append(message)
    print(f"ADVERTENCIA: {message}")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit(root: Path = PROJECT_ROOT) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "No disponible"


def require_files(reports_dir: Path) -> dict[str, Path]:
    paths = {name: reports_dir / file_name for name, file_name in REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"No existen archivos requeridos: {missing}")
    empty = [str(path) for path in paths.values() if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"Archivos requeridos vacios: {empty}")
    return paths


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    source: str,
    warnings: list[str] | None = None,
) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        message = f"{source} no contiene columnas obligatorias: {missing}"
        if warnings is None:
            raise ValueError(message)
        warn(warnings, message)
        raise ValueError(message)


def date_to_days(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce", format="mixed")
    return dates.map(
        lambda value: (
            value.toordinal()
            + (
                value.hour * 3600
                + value.minute * 60
                + value.second
                + value.microsecond / 1_000_000
            )
            / 86_400
        )
        if pd.notna(value)
        else np.nan
    ).astype(float)


def date_to_season_days(values: pd.Series, seasons: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce", format="mixed")
    season_values = pd.to_numeric(seasons, errors="coerce")
    bases = pd.to_datetime(
        season_values.dropna().astype(int).astype(str) + "-04-01",
        errors="coerce",
    )
    aligned_bases = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    aligned_bases.loc[bases.index] = bases
    return ((dates - aligned_bases) / pd.Timedelta(days=1)).astype(float)


def signed_date_error_days(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    actual_dates = pd.to_datetime(actual, errors="coerce", format="mixed")
    predicted_dates = pd.to_datetime(predicted, errors="coerce", format="mixed")
    return (predicted_dates - actual_dates) / pd.Timedelta(days=1)


def mae_from_errors(errors: pd.Series) -> float:
    valid = pd.to_numeric(errors, errors="coerce").dropna()
    return float(valid.abs().mean()) if not valid.empty else float("nan")


def percent_within_abs_error(errors: pd.Series, threshold_days: float) -> float:
    valid = pd.to_numeric(errors, errors="coerce").dropna()
    return float((valid.abs() <= threshold_days).mean() * 100.0) if not valid.empty else float("nan")


def rmse_from_errors(errors: pd.Series) -> float:
    valid = pd.to_numeric(errors, errors="coerce").dropna()
    return float(np.sqrt(np.mean(np.square(valid)))) if not valid.empty else float("nan")


def r2_score_numeric(actual_values: pd.Series, predicted_values: pd.Series) -> float:
    actual_days = pd.to_numeric(actual_values, errors="coerce")
    predicted_days = pd.to_numeric(predicted_values, errors="coerce")
    mask = actual_days.notna() & predicted_days.notna()
    if mask.sum() < 2:
        return float("nan")
    y = actual_days[mask].astype(float)
    yhat = predicted_days[mask].astype(float)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def r2_score_days(actual: pd.Series, predicted: pd.Series) -> float:
    return r2_score_numeric(date_to_days(actual), date_to_days(predicted))


def r2_score_season_days(actual: pd.Series, predicted: pd.Series, seasons: pd.Series) -> float:
    return r2_score_numeric(
        date_to_season_days(actual, seasons),
        date_to_season_days(predicted, seasons),
    )


def compute_prediction_metrics(
    frame: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    stored_error_col: str | None = None,
    season_col: str | None = None,
) -> dict[str, float]:
    errors = signed_date_error_days(frame[actual_col], frame[pred_col])
    if stored_error_col and stored_error_col in frame.columns:
        stored = pd.to_numeric(frame[stored_error_col], errors="coerce")
        if stored.notna().sum() == errors.notna().sum():
            errors = stored

    valid = errors.dropna()
    abs_valid = valid.abs()
    percentiles = abs_valid.quantile([0.50, 0.75, 0.90, 0.95, 0.99])
    r2_calendar = r2_score_days(frame[actual_col], frame[pred_col])
    r2_season = (
        r2_score_season_days(frame[actual_col], frame[pred_col], frame[season_col])
        if season_col and season_col in frame.columns
        else float("nan")
    )
    r2_preferred = r2_season if pd.notna(r2_season) else r2_calendar
    return {
        "n_validas": int(valid.size),
        "mae_dias": mae_from_errors(errors),
        "mediana_error_abs_dias": float(abs_valid.median()) if not abs_valid.empty else float("nan"),
        "rmse_dias": rmse_from_errors(errors),
        "r2": r2_preferred,
        "r2_calendario_absoluto": r2_calendar,
        "r2_intra_temporada": r2_season,
        "sesgo_medio_dias": float(valid.mean()) if not valid.empty else float("nan"),
        "std_error_dias": float(valid.std(ddof=0)) if not valid.empty else float("nan"),
        "p50_error_abs_dias": float(percentiles.loc[0.50]) if not percentiles.empty else float("nan"),
        "p75_error_abs_dias": float(percentiles.loc[0.75]) if not percentiles.empty else float("nan"),
        "p90_error_abs_dias": float(percentiles.loc[0.90]) if not percentiles.empty else float("nan"),
        "p95_error_abs_dias": float(percentiles.loc[0.95]) if not percentiles.empty else float("nan"),
        "p99_error_abs_dias": float(percentiles.loc[0.99]) if not percentiles.empty else float("nan"),
        "pct_abs_leq_3_dias": percent_within_abs_error(errors, 3),
        "pct_abs_leq_5_dias": percent_within_abs_error(errors, 5),
        "pct_abs_leq_7_dias": percent_within_abs_error(errors, 7),
        "pct_abs_leq_10_dias": percent_within_abs_error(errors, 10),
        "pct_abs_leq_14_dias": percent_within_abs_error(errors, 14),
    }


def format_number(value: Any, suffix: str = "") -> str:
    if value is None:
        return "No disponible"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(numeric):
        return "No disponible"
    return f"{numeric:,.2f}{suffix}"


def format_int(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "No disponible"


def infer_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    lower_map = {column.lower(): column for column in df.columns}
    return {
        "fechas_observadas": [
            lower_map[name]
            for name in ["fecha_brotacion_camara_50", "fecha_brotacion_campo_50"]
            if name in lower_map
        ],
        "fechas_predichas": [
            lower_map[name]
            for name in ["pred_fecha_fin_frio", "pred_fecha_brotacion"]
            if name in lower_map
        ],
        "errores": [column for column in df.columns if column.startswith("error_")],
        "grupos": [column for column in ["grupo_lugar_variedad_temporada", "fold_id"] if column in df.columns],
        "lugares": [column for column in ["codigo_lugar", "lugar"] if column in df.columns],
        "variedades": [column for column in ["codigo_variedad", "variedad"] if column in df.columns],
        "temporadas": [column for column in ["temporada"] if column in df.columns],
        "tratamientos": [
            column
            for column in ["codigo_tratamiento_entrada", "codigo_tratamiento_salida"]
            if column in df.columns
        ],
        "parejas_modelos": [
            column for column in ["thermal_pair", "model_name", "validation_strategy"] if column in df.columns
        ],
    }


def build_file_inventory(paths: dict[str, Path], schemas: dict[str, dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for key, path in paths.items():
        info = schemas.get(key, {})
        try:
            display_path = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            display_path = str(path)
        records.append(
            {
                "archivo": display_path,
                "modificado": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "tamano_bytes": path.stat().st_size,
                "filas": info.get("rows", "No aplica"),
                "columnas": info.get("columns_count", "No aplica"),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(records)


def load_inputs(reports_dir: Path = DEFAULT_REPORTS_DIR) -> InputData:
    warnings: list[str] = []
    paths = require_files(reports_dir)

    metrics = pd.read_csv(paths["metrics"])
    oof = pd.read_csv(paths["oof"])
    segments = pd.read_csv(paths["segments"])
    selected = json.loads(paths["selected"].read_text(encoding="utf-8"))
    limitations_markdown = paths["limitations"].read_text(encoding="utf-8")

    require_columns(metrics, REQUIRED_METRICS_COLUMNS, "metrics_by_pair.csv", warnings)
    require_columns(oof, REQUIRED_OOF_COLUMNS, "oof_predictions.csv", warnings)
    require_columns(segments, REQUIRED_SEGMENT_COLUMNS, "error_by_segment.csv", warnings)
    metrics = enrich_metrics_with_oof_coverage(metrics, oof, warnings)
    missing_selected = sorted(set(SELECTED_KEYS).difference(selected))
    if missing_selected:
        raise ValueError(f"selected_pair.json no contiene claves obligatorias: {missing_selected}")

    schemas = {
        "metrics": {
            "rows": len(metrics),
            "columns_count": len(metrics.columns),
            "columns": metrics.columns.tolist(),
            "dtypes": metrics.dtypes.astype(str).to_dict(),
            "inferred": infer_columns(metrics),
        },
        "oof": {
            "rows": len(oof),
            "columns_count": len(oof.columns),
            "columns": oof.columns.tolist(),
            "dtypes": oof.dtypes.astype(str).to_dict(),
            "inferred": infer_columns(oof),
        },
        "segments": {
            "rows": len(segments),
            "columns_count": len(segments.columns),
            "columns": segments.columns.tolist(),
            "dtypes": segments.dtypes.astype(str).to_dict(),
            "inferred": infer_columns(segments),
        },
        "selected": {"rows": "No aplica", "columns_count": len(selected), "columns": list(selected.keys())},
        "limitations": {"rows": len(limitations_markdown.splitlines()), "columns_count": "No aplica", "columns": []},
    }
    file_inventory = build_file_inventory(paths, schemas)

    validate_inputs(metrics, oof, segments, selected, warnings)
    return InputData(metrics, oof, segments, selected, limitations_markdown, file_inventory, schemas, warnings)


def enrich_metrics_with_oof_coverage(
    metrics: pd.DataFrame,
    oof: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    missing_optional = sorted(set(OPTIONAL_METRICS_COLUMNS).difference(metrics.columns))
    if not missing_optional:
        return metrics

    warn(
        warnings,
        "metrics_by_pair.csv no contiene columnas opcionales de cobertura "
        f"{missing_optional}; se recalculan desde oof_predictions.csv.",
    )
    coverage = (
        oof.groupby(SELECTED_KEYS, dropna=False)
        .agg(
            n_pred_fecha_fin_frio_validas=("pred_fecha_fin_frio", lambda values: int(pd.to_datetime(values, errors="coerce", format="mixed").notna().sum())),
            n_pred_fecha_brotacion_validas=("pred_fecha_brotacion", lambda values: int(pd.to_datetime(values, errors="coerce", format="mixed").notna().sum())),
            n_oof=("unidad_experimental_id", "size"),
        )
        .reset_index()
    )
    coverage["porcentaje_pred_fecha_brotacion_valida"] = (
        coverage["n_pred_fecha_brotacion_validas"] / coverage["n_oof"] * 100.0
    )
    coverage = coverage.drop(columns=["n_oof"])
    enriched = metrics.merge(coverage, on=SELECTED_KEYS, how="left")
    return enriched


def validate_inputs(
    metrics: pd.DataFrame,
    oof: pd.DataFrame,
    segments: pd.DataFrame,
    selected: dict[str, Any],
    warnings: list[str],
) -> None:
    if metrics.empty:
        raise ValueError("metrics_by_pair.csv esta vacio")
    if oof.empty:
        raise ValueError("oof_predictions.csv esta vacio")
    if segments.empty:
        warn(warnings, "error_by_segment.csv esta vacio; no se podra analizar desempeno por segmento.")

    metric_key_dups = metrics.duplicated(SELECTED_KEYS).sum()
    if metric_key_dups:
        warn(warnings, f"metrics_by_pair.csv contiene {metric_key_dups} combinaciones duplicadas por pareja/modelo/validacion.")

    oof_base_key = SELECTED_KEYS + ["unidad_experimental_id"]
    oof_dups = int(oof.duplicated(oof_base_key).sum())
    if oof_dups:
        extended_columns = [
            column
            for column in [
                "lugar",
                "fecha_caida_hoja_50",
                "fecha_brotacion_camara_50",
                "fecha_brotacion_campo_50",
            ]
            if column in oof.columns
        ]
        extended_key = oof_base_key + extended_columns
        extended_dups = int(oof.duplicated(extended_key).sum())
        if extended_dups:
            warn(
                warnings,
                "oof_predictions.csv contiene duplicados reales por combinacion, "
                f"unidad experimental y columnas fenologicas disponibles: {extended_dups}.",
            )
        else:
            print(
                "NOTA: oof_predictions.csv tiene "
                f"{oof_dups} colisiones de unidad_experimental_id, pero se resuelven "
                "al incluir lugar y fechas fenologicas; no se tratan como duplicados OOF reales."
            )

    selected_mask = selected_filter(metrics, selected)
    if selected_mask.sum() != 1:
        warn(
            warnings,
            "selected_pair.json no coincide univocamente con metrics_by_pair.csv; "
            f"coincidencias={int(selected_mask.sum())}.",
        )

    selected_oof_mask = selected_filter(oof, selected)
    if selected_oof_mask.sum() == 0:
        warn(warnings, "No hay predicciones OOF para la alternativa seleccionada.")

    critical_oof = ["unidad_experimental_id", "thermal_pair", "model_name", "validation_strategy", "fold_id"]
    null_counts = oof[critical_oof].isna().sum()
    if null_counts.sum():
        warn(warnings, f"Nulos en campos criticos de OOF: {null_counts[null_counts > 0].to_dict()}")

    date_columns = [
        "fecha_brotacion_camara_50",
        "fecha_brotacion_campo_50",
        "pred_fecha_fin_frio",
        "pred_fecha_brotacion",
    ]
    for column in date_columns:
        parsed = pd.to_datetime(oof[column], errors="coerce", format="mixed")
        invalid = int(oof[column].notna().sum() - parsed.notna().sum())
        if invalid:
            warn(warnings, f"Columna {column} contiene {invalid} fechas invalidas.")

    numeric_metrics = metrics.select_dtypes(include=[np.number])
    if np.isinf(numeric_metrics.to_numpy()).any():
        warn(warnings, "metrics_by_pair.csv contiene valores infinitos.")
    numeric_oof = oof.select_dtypes(include=[np.number])
    if np.isinf(numeric_oof.to_numpy()).any():
        warn(warnings, "oof_predictions.csv contiene valores infinitos.")

    percentage_columns = [column for column in metrics.columns if "porcentaje" in column]
    for column in percentage_columns:
        bad = metrics[column].notna() & ~metrics[column].between(0, 100)
        if bad.any():
            warn(warnings, f"Metrica porcentual fuera de [0, 100] en {column}: {int(bad.sum())} filas.")

    mae_columns = [column for column in metrics.columns if column.startswith("mae_")]
    for column in mae_columns:
        bad = metrics[column].notna() & (metrics[column] < 0)
        if bad.any():
            warn(warnings, f"MAE negativo en {column}: {int(bad.sum())} filas.")

    if "fold_id" in oof.columns:
        group_column = "grupo_lugar_variedad_temporada" if "grupo_lugar_variedad_temporada" in oof.columns else None
        if group_column:
            group_fold_counts = (
                oof.groupby(SELECTED_KEYS + [group_column])["fold_id"].nunique()
            )
            bad_groups = int((group_fold_counts > 1).sum())
            if bad_groups:
                warn(
                    warnings,
                    f"{bad_groups} grupos aparecen en mas de un fold dentro de una misma combinacion.",
                )
        else:
            warn(warnings, "No existe columna de grupo para verificar separacion de folds.")


def selected_filter(df: pd.DataFrame, selected: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key in SELECTED_KEYS:
        if key not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= df[key].astype(str) == str(selected[key])
    return mask


def selected_target_columns(selected: dict[str, Any]) -> tuple[str, str]:
    pair = selected["thermal_pair"]
    return f"target_frio__{pair}", f"target_calor__{pair}"


def recompute_selected_metrics(selected_oof: pd.DataFrame, selected: dict[str, Any]) -> dict[str, float]:
    cold_target, heat_target = selected_target_columns(selected)
    metrics = compute_prediction_metrics(
        selected_oof,
        "fecha_brotacion_camara_50",
        "pred_fecha_fin_frio",
        "error_fecha_fin_frio_dias",
        "temporada",
    )
    bud_metrics = compute_prediction_metrics(
        selected_oof,
        "fecha_brotacion_campo_50",
        "pred_fecha_brotacion",
        "error_fecha_brotacion_dias",
        "temporada",
    )
    requirement_cold = requirement_metrics(selected_oof, cold_target, "pred_requisito_frio")
    requirement_heat = requirement_metrics(selected_oof, heat_target, "pred_requisito_calor")
    return {
        "n_observaciones": int(len(selected_oof)),
        "n_pred_fecha_fin_frio_validas": int(metrics["n_validas"]),
        "n_pred_fecha_brotacion_validas": int(bud_metrics["n_validas"]),
        "porcentaje_pred_fecha_brotacion_valida": float(bud_metrics["n_validas"] / len(selected_oof) * 100.0)
        if len(selected_oof)
        else float("nan"),
        "r2_requisito_frio": requirement_cold["r2"],
        "mae_requisito_frio": requirement_cold["mae"],
        "r2_requisito_calor": requirement_heat["r2"],
        "mae_requisito_calor": requirement_heat["mae"],
        "r2_fecha_fin_frio": metrics["r2_intra_temporada"],
        "r2_fecha_fin_frio_intra_temporada": metrics["r2_intra_temporada"],
        "r2_fecha_fin_frio_calendario_absoluto": metrics["r2_calendario_absoluto"],
        "mae_fecha_fin_frio_dias": metrics["mae_dias"],
        "r2_fecha_brotacion": bud_metrics["r2_intra_temporada"],
        "r2_fecha_brotacion_intra_temporada": bud_metrics["r2_intra_temporada"],
        "r2_fecha_brotacion_calendario_absoluto": bud_metrics["r2_calendario_absoluto"],
        "mae_fecha_brotacion_dias": bud_metrics["mae_dias"],
        "porcentaje_error_menor_igual_3_dias": bud_metrics["pct_abs_leq_3_dias"],
        "porcentaje_error_menor_igual_5_dias": bud_metrics["pct_abs_leq_5_dias"],
        "porcentaje_error_menor_igual_7_dias": bud_metrics["pct_abs_leq_7_dias"],
    }


def requirement_metrics(df: pd.DataFrame, actual_col: str, pred_col: str) -> dict[str, float]:
    if actual_col not in df.columns or pred_col not in df.columns:
        return {"mae": float("nan"), "r2": float("nan")}
    actual = pd.to_numeric(df[actual_col], errors="coerce")
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    mask = actual.notna() & pred.notna()
    if not mask.any():
        return {"mae": float("nan"), "r2": float("nan")}
    y = actual[mask].astype(float)
    yhat = pred[mask].astype(float)
    mae_value = float((y - yhat).abs().mean())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_value = float("nan") if ss_tot == 0 else 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot
    return {"mae": mae_value, "r2": r2_value}


def compare_selected_metrics(
    selected_row: pd.Series | None,
    recalculated: dict[str, float],
    warnings: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if selected_row is None:
        warn(warnings, "No se pudo comparar selected_pair.json contra metrics_by_pair.csv.")
        return pd.DataFrame(rows)

    for metric, recalculated_value in recalculated.items():
        if metric not in selected_row.index:
            rows.append(
                {
                    "metrica": metric,
                    "valor_leido": "No disponible",
                    "valor_recalculado": recalculated_value,
                    "diferencia_abs": np.nan,
                    "consistente": "Metrica nueva del reporte",
                }
            )
            continue
        read_value = selected_row.get(metric, np.nan)
        diff = (
            abs(float(read_value) - float(recalculated_value))
            if pd.notna(read_value) and pd.notna(recalculated_value)
            else np.nan
        )
        consistent = pd.notna(diff) and diff <= CONSISTENCY_TOLERANCE
        rows.append(
            {
                "metrica": metric,
                "valor_leido": read_value,
                "valor_recalculado": recalculated_value,
                "diferencia_abs": diff,
                "consistente": consistent,
            }
        )
        if pd.notna(diff) and diff > max(CONSISTENCY_TOLERANCE, 0.01):
            warn(
                warnings,
                f"Inconsistencia en {metric}: leido={read_value}, recalculado={recalculated_value}, diff={diff}.",
            )
    return pd.DataFrame(rows)


def markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    output: list[str] = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                output.append("</ul>")
                in_list = False
            continue
        if stripped.startswith("#"):
            if in_list:
                output.append("</ul>")
                in_list = False
            level = min(len(stripped) - len(stripped.lstrip("#")), 6)
            text = stripped[level:].strip()
            output.append(f"<h{level}>{html.escape(text)}</h{level}>")
        elif stripped.startswith("- "):
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{inline_markdown(stripped[2:])}</li>")
        else:
            if in_list:
                output.append("</ul>")
                in_list = False
            output.append(f"<p>{inline_markdown(stripped)}</p>")
    if in_list:
        output.append("</ul>")
    return "\n".join(output)


def inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    parts = escaped.split("`")
    for idx in range(1, len(parts), 2):
        parts[idx] = f"<code>{parts[idx]}</code>"
    return "".join(parts)


def df_to_html_table(df: pd.DataFrame, max_rows: int | None = None, css_class: str = "data-table") -> str:
    show = df.copy()
    if max_rows is not None:
        show = show.head(max_rows)
    for column in show.columns:
        if pd.api.types.is_float_dtype(show[column]):
            show[column] = show[column].map(lambda value: format_number(value) if pd.notna(value) else "No disponible")
    return show.to_html(index=False, escape=True, classes=css_class, border=0)


def make_chart_html(fig: go.Figure, div_id: str) -> str:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=25, t=55, b=45),
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


def metric_cards(recalculated: dict[str, float], selected_oof: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"label": "Calor - MAE requisito", "value": format_number(recalculated["mae_requisito_calor"])},
        {"label": "Calor - R2 requisito", "value": format_number(recalculated["r2_requisito_calor"])},
        {"label": "Frio - MAE requisito", "value": format_number(recalculated["mae_requisito_frio"])},
        {"label": "Frio - R2 requisito", "value": format_number(recalculated["r2_requisito_frio"])},
        {"label": "Brotacion - MAE fecha", "value": format_number(recalculated["mae_fecha_brotacion_dias"], " dias")},
        {"label": "Brotacion - R2 intra-temporada", "value": format_number(recalculated["r2_fecha_brotacion_intra_temporada"])},
        {"label": "Brotacion - error <= 3 dias", "value": format_number(recalculated["porcentaje_error_menor_igual_3_dias"], "%")},
        {"label": "Brotacion - error <= 7 dias", "value": format_number(recalculated["porcentaje_error_menor_igual_7_dias"], "%")},
        {"label": "Observaciones evaluadas", "value": format_int(recalculated["n_observaciones"])},
        {"label": "Grupos", "value": format_int(selected_oof["grupo_lugar_variedad_temporada"].nunique()) if "grupo_lugar_variedad_temporada" in selected_oof else "No disponible"},
        {"label": "Lugares", "value": format_int(selected_oof["codigo_lugar"].nunique()) if "codigo_lugar" in selected_oof else "No disponible"},
        {"label": "Variedades", "value": format_int(selected_oof["codigo_variedad"].nunique()) if "codigo_variedad" in selected_oof else "No disponible"},
        {"label": "Temporadas", "value": format_int(selected_oof["temporada"].nunique()) if "temporada" in selected_oof else "No disponible"},
    ]


def build_charts(metrics: pd.DataFrame, selected_oof: pd.DataFrame) -> dict[str, str]:
    charts: dict[str, str] = {}
    selected_strategy = selected_oof["validation_strategy"].iloc[0]
    primary_metrics = metrics.loc[metrics["validation_strategy"] == selected_strategy].copy()
    primary_metrics["seleccionada"] = (
        (primary_metrics["thermal_pair"] == selected_oof["thermal_pair"].iloc[0])
        & (primary_metrics["model_name"] == selected_oof["model_name"].iloc[0])
    )

    fig = px.bar(
        primary_metrics,
        x="thermal_pair",
        y="mae_fecha_brotacion_dias",
        color="model_name",
        barmode="group",
        title="MAE de brotacion por pareja termica y modelo",
        labels={"thermal_pair": "Pareja termica", "mae_fecha_brotacion_dias": "MAE brotacion (dias)", "model_name": "Modelo"},
    )
    charts["mae_bud"] = make_chart_html(fig, "chart-mae-bud")

    fig = px.bar(
        primary_metrics,
        x="thermal_pair",
        y="mae_fecha_fin_frio_dias",
        color="model_name",
        barmode="group",
        title="MAE de termino de frio por pareja termica y modelo",
        labels={"thermal_pair": "Pareja termica", "mae_fecha_fin_frio_dias": "MAE termino de frio (dias)", "model_name": "Modelo"},
    )
    charts["mae_cold"] = make_chart_html(fig, "chart-mae-cold")

    fig = px.bar(
        primary_metrics,
        x="thermal_pair",
        y="n_pred_fecha_brotacion_validas",
        color="model_name",
        barmode="group",
        title="Cobertura de fechas de brotacion simuladas",
        labels={"thermal_pair": "Pareja termica", "n_pred_fecha_brotacion_validas": "Predicciones validas", "model_name": "Modelo"},
    )
    charts["coverage"] = make_chart_html(fig, "chart-coverage")

    fig = px.scatter(
        primary_metrics,
        x="mae_fecha_fin_frio_dias",
        y="mae_fecha_brotacion_dias",
        color="model_name",
        size="n_pred_fecha_brotacion_validas",
        hover_data=["thermal_pair", "porcentaje_pred_fecha_brotacion_valida"],
        title="Relacion entre MAE de frio y MAE de brotacion",
        labels={"mae_fecha_fin_frio_dias": "MAE termino de frio (dias)", "mae_fecha_brotacion_dias": "MAE brotacion (dias)"},
    )
    charts["tradeoff"] = make_chart_html(fig, "chart-tradeoff")

    charts.update(build_prediction_charts(selected_oof, "brotacion", "fecha_brotacion_campo_50", "pred_fecha_brotacion", "error_fecha_brotacion_dias"))
    charts.update(build_prediction_charts(selected_oof, "fin_frio", "fecha_brotacion_camara_50", "pred_fecha_fin_frio", "error_fecha_fin_frio_dias"))

    if selected_oof[["error_fecha_fin_frio_dias", "error_fecha_brotacion_dias"]].dropna().shape[0] >= 2:
        fig = px.scatter(
            selected_oof,
            x="error_fecha_fin_frio_dias",
            y="error_fecha_brotacion_dias",
            color="codigo_lugar" if "codigo_lugar" in selected_oof.columns else None,
            hover_data=[column for column in IDENTIFIER_COLUMNS if column in selected_oof.columns],
            title="Propagacion del error: termino de frio versus brotacion",
            labels={"error_fecha_fin_frio_dias": "Error termino de frio (dias)", "error_fecha_brotacion_dias": "Error brotacion (dias)"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
        charts["propagation"] = make_chart_html(fig, "chart-propagation")

    if "fold_id" in selected_oof.columns:
        fig = px.box(
            selected_oof,
            x="fold_id",
            y="error_fecha_brotacion_dias",
            points="all",
            title="Distribucion del error de brotacion por fold",
            labels={"fold_id": "Fold", "error_fecha_brotacion_dias": "Error brotacion (dias)"},
        )
        charts["fold_errors"] = make_chart_html(fig, "chart-fold-errors")

    return charts


def build_prediction_charts(
    df: pd.DataFrame,
    prefix: str,
    actual_col: str,
    pred_col: str,
    error_col: str,
) -> dict[str, str]:
    charts: dict[str, str] = {}
    work = df.copy()
    work[actual_col] = pd.to_datetime(work[actual_col], errors="coerce", format="mixed")
    work[pred_col] = pd.to_datetime(work[pred_col], errors="coerce", format="mixed")
    work[error_col] = pd.to_numeric(work[error_col], errors="coerce")
    valid = work.loc[work[actual_col].notna() & work[pred_col].notna()].copy()
    if valid.empty:
        return charts

    metrics = compute_prediction_metrics(valid, actual_col, pred_col, error_col)
    min_date = min(valid[actual_col].min(), valid[pred_col].min())
    max_date = max(valid[actual_col].max(), valid[pred_col].max())
    fig = px.scatter(
        valid,
        x=actual_col,
        y=pred_col,
        color="codigo_lugar" if "codigo_lugar" in valid.columns else None,
        opacity=0.55,
        hover_data=[column for column in IDENTIFIER_COLUMNS if column in valid.columns],
        title=f"Observado versus predicho ({prefix.replace('_', ' ')}) - n={metrics['n_validas']}, MAE={metrics['mae_dias']:.2f}, R2={metrics['r2']:.3f}",
    )
    fig.add_trace(
        go.Scatter(
            x=[min_date, max_date],
            y=[min_date, max_date],
            mode="lines",
            line=dict(color="#111827", dash="dash"),
            name="Identidad",
        )
    )
    fig.update_xaxes(range=[min_date, max_date])
    fig.update_yaxes(range=[min_date, max_date])
    charts[f"{prefix}_obs_pred"] = make_chart_html(fig, f"chart-{prefix}-obs-pred")

    fig = px.histogram(
        valid,
        x=error_col,
        nbins=40,
        title=f"Histograma de error firmado ({prefix.replace('_', ' ')})",
        labels={error_col: "Error firmado (dias)"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#111827")
    charts[f"{prefix}_signed_hist"] = make_chart_html(fig, f"chart-{prefix}-signed-hist")

    abs_errors = valid.assign(error_abs_dias=valid[error_col].abs()).sort_values("error_abs_dias")
    fig = px.histogram(
        abs_errors,
        x="error_abs_dias",
        nbins=40,
        cumulative=False,
        title=f"Distribucion del error absoluto ({prefix.replace('_', ' ')})",
        labels={"error_abs_dias": "Error absoluto (dias)"},
    )
    charts[f"{prefix}_abs_hist"] = make_chart_html(fig, f"chart-{prefix}-abs-hist")

    fig = px.scatter(
        valid,
        x=pred_col,
        y=error_col,
        color="codigo_lugar" if "codigo_lugar" in valid.columns else None,
        opacity=0.55,
        title=f"Residuos versus prediccion ({prefix.replace('_', ' ')})",
        labels={pred_col: "Valor predicho", error_col: "Error firmado (dias)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#111827")
    charts[f"{prefix}_residuals"] = make_chart_html(fig, f"chart-{prefix}-residuals")

    sorted_errors = abs_errors.reset_index(drop=True)
    sorted_errors["orden"] = np.arange(1, len(sorted_errors) + 1)
    fig = px.line(
        sorted_errors,
        x="orden",
        y="error_abs_dias",
        title=f"Error absoluto ordenado ({prefix.replace('_', ' ')})",
        labels={"orden": "Observaciones ordenadas", "error_abs_dias": "Error absoluto (dias)"},
    )
    charts[f"{prefix}_ordered_abs"] = make_chart_html(fig, f"chart-{prefix}-ordered-abs")
    return charts


def build_segment_outputs(segments: pd.DataFrame, selected_oof: pd.DataFrame) -> dict[str, Any]:
    segment_tables: dict[str, str] = {}
    segment_charts: dict[str, str] = {}
    work = segments.copy()
    work["soporte_bajo"] = work["n"] < MIN_SEGMENT_SUPPORT
    for segment_name, group in work.groupby("segmento", sort=False):
        ordered = group.sort_values("mae_fecha_brotacion_dias", ascending=False)
        segment_tables[segment_name] = df_to_html_table(ordered)
        fig = px.bar(
            ordered,
            x="segmento_valor",
            y="mae_fecha_brotacion_dias",
            color="soporte_bajo",
            text="n" if "n" in ordered.columns else None,
            title=f"MAE de brotacion por {segment_name}",
            labels={"segmento_valor": segment_name, "mae_fecha_brotacion_dias": "MAE brotacion (dias)", "soporte_bajo": f"n < {MIN_SEGMENT_SUPPORT}"},
        )
        segment_charts[segment_name] = make_chart_html(fig, f"chart-segment-{segment_name}")

    best = work.sort_values("mae_fecha_brotacion_dias").head(10)
    worst = work.sort_values("mae_fecha_brotacion_dias", ascending=False).head(10)
    high_bias = work.assign(abs_bias=work["bias_fecha_brotacion_dias"].abs()).sort_values("abs_bias", ascending=False).head(10)

    heatmap_html = ""
    if {"codigo_lugar", "codigo_variedad"}.issubset(selected_oof.columns):
        pivot = (
            selected_oof.groupby(["codigo_lugar", "codigo_variedad"])["error_fecha_brotacion_dias"]
            .apply(lambda values: values.abs().mean())
            .reset_index(name="mae_fecha_brotacion_dias")
        )
        if pivot["codigo_lugar"].nunique() > 1 and pivot["codigo_variedad"].nunique() > 1:
            matrix = pivot.pivot(index="codigo_lugar", columns="codigo_variedad", values="mae_fecha_brotacion_dias")
            fig = px.imshow(
                matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Blues",
                title="Heatmap MAE brotacion por lugar y variedad",
                labels=dict(color="MAE dias"),
            )
            heatmap_html = make_chart_html(fig, "chart-heatmap-location-variety")

    return {
        "tables": segment_tables,
        "charts": segment_charts,
        "best_table": df_to_html_table(best),
        "worst_table": df_to_html_table(worst),
        "high_bias_table": df_to_html_table(high_bias.drop(columns=["abs_bias"])),
        "heatmap": heatmap_html,
        "low_support_count": int(work["soporte_bajo"].sum()),
    }


def build_extreme_cases(selected_oof: pd.DataFrame) -> dict[str, str]:
    base_columns = [column for column in IDENTIFIER_COLUMNS if column in selected_oof.columns]
    common_cols = base_columns + [
        "thermal_pair",
        "model_name",
        "validation_strategy",
        "fecha_brotacion_camara_50",
        "pred_fecha_fin_frio",
        "error_fecha_fin_frio_dias",
        "fecha_brotacion_campo_50",
        "pred_fecha_brotacion",
        "error_fecha_brotacion_dias",
    ]
    common_cols = [column for column in common_cols if column in selected_oof.columns]
    work = selected_oof.copy()
    work["abs_error_brotacion_dias"] = pd.to_numeric(work["error_fecha_brotacion_dias"], errors="coerce").abs()
    work["abs_error_fin_frio_dias"] = pd.to_numeric(work["error_fecha_fin_frio_dias"], errors="coerce").abs()
    work["cobertura_termica"] = work["pred_fecha_brotacion"].notna() & work["pred_fecha_fin_frio"].notna()

    return {
        "brotacion_abs": df_to_html_table(work.sort_values("abs_error_brotacion_dias", ascending=False)[common_cols + ["abs_error_brotacion_dias", "cobertura_termica"]].head(15)),
        "frio_abs": df_to_html_table(work.sort_values("abs_error_fin_frio_dias", ascending=False)[common_cols + ["abs_error_fin_frio_dias", "cobertura_termica"]].head(15)),
        "sobreestimacion": df_to_html_table(work.sort_values("error_fecha_brotacion_dias", ascending=False)[common_cols + ["abs_error_brotacion_dias", "cobertura_termica"]].head(15)),
        "subestimacion": df_to_html_table(work.sort_values("error_fecha_brotacion_dias", ascending=True)[common_cols + ["abs_error_brotacion_dias", "cobertura_termica"]].head(15)),
    }


def propagation_analysis(selected_oof: pd.DataFrame) -> dict[str, Any]:
    work = selected_oof[["error_fecha_fin_frio_dias", "error_fecha_brotacion_dias"]].dropna().copy()
    if len(work) < 2:
        return {"pearson": float("nan"), "spearman": float("nan"), "quadrants": pd.DataFrame(), "notes": []}
    pearson = float(work["error_fecha_fin_frio_dias"].corr(work["error_fecha_brotacion_dias"], method="pearson"))
    spearman = float(
        work["error_fecha_fin_frio_dias"]
        .rank()
        .corr(work["error_fecha_brotacion_dias"].rank(), method="pearson")
    )
    conditions = [
        (work["error_fecha_fin_frio_dias"] >= 0) & (work["error_fecha_brotacion_dias"] >= 0),
        (work["error_fecha_fin_frio_dias"] < 0) & (work["error_fecha_brotacion_dias"] < 0),
        (work["error_fecha_fin_frio_dias"] >= 0) & (work["error_fecha_brotacion_dias"] < 0),
        (work["error_fecha_fin_frio_dias"] < 0) & (work["error_fecha_brotacion_dias"] >= 0),
    ]
    labels = [
        "frio y brotacion sobreestimados",
        "frio y brotacion subestimados",
        "frio sobreestimado, brotacion subestimada",
        "frio subestimado, brotacion sobreestimada",
    ]
    work["cuadrante"] = np.select(conditions, labels, default="sin clasificar")
    quadrants = (
        work.groupby("cuadrante")
        .agg(n=("error_fecha_brotacion_dias", "size"), mae_brotacion=("error_fecha_brotacion_dias", lambda values: values.abs().mean()))
        .reset_index()
    )
    notes = [
        f"Casos con error de frio > 7 dias y error de brotacion <= 3 dias: {int(((work['error_fecha_fin_frio_dias'].abs() > 7) & (work['error_fecha_brotacion_dias'].abs() <= 3)).sum())}",
        f"Casos con error de frio <= 3 dias y error de brotacion > 7 dias: {int(((work['error_fecha_fin_frio_dias'].abs() <= 3) & (work['error_fecha_brotacion_dias'].abs() > 7)).sum())}",
    ]
    return {"pearson": pearson, "spearman": spearman, "quadrants": quadrants, "notes": notes}


def coverage_tables(selected_oof: pd.DataFrame) -> dict[str, str]:
    work = selected_oof.copy()
    work["cobertura_termica_completa"] = work["pred_fecha_fin_frio"].notna() & work["pred_fecha_brotacion"].notna()
    tables: dict[str, str] = {}
    for column in ["codigo_lugar", "codigo_variedad", "temporada", "codigo_tratamiento_entrada", "codigo_tratamiento_salida", "fold_id"]:
        if column in work.columns:
            summary = (
                work.groupby(column, dropna=False)
                .agg(
                    observaciones=("unidad_experimental_id", "size"),
                    cobertura_completa=("cobertura_termica_completa", "sum"),
                )
                .reset_index()
                .rename(columns={column: "segmento_valor"})
            )
            summary.insert(0, "segmento", column)
            summary["porcentaje_cobertura"] = summary["cobertura_completa"] / summary["observaciones"] * 100.0
            tables[column] = df_to_html_table(summary)
    return tables


def generalization_table(metrics: pd.DataFrame, selected_oof: pd.DataFrame) -> pd.DataFrame:
    strategies = set(metrics["validation_strategy"].dropna().astype(str))
    group_ok = "grupo_lugar_variedad_temporada" in selected_oof.columns and "fold_id" in selected_oof.columns
    return pd.DataFrame(
        [
            {
                "pregunta": "Se evaluo fuera de muestra",
                "estado": "Parcialmente demostrado" if "fold_id" in selected_oof.columns else "No verificable con los archivos",
                "evidencia": "Existe fold_id en oof_predictions.csv; no estan disponibles los indices de entrenamiento.",
            },
            {
                "pregunta": "Las predicciones son out-of-fold",
                "estado": "Parcialmente demostrado" if group_ok else "No verificable con los archivos",
                "evidencia": "Cada unidad aparece una vez por combinacion y fold_id esta registrado; no se puede reconstruir el entrenamiento exacto.",
            },
            {
                "pregunta": "Los folds separan lugar-variedad-temporada",
                "estado": "Parcialmente demostrado" if "GroupKFold_lugar_variedad_temporada" in strategies else "No demostrado",
                "evidencia": "La estrategia esta registrada en metrics_by_pair.csv.",
            },
            {
                "pregunta": "Generalizacion a un lugar completamente excluido",
                "estado": "Parcialmente demostrado" if "LeaveOneLocationOut" in strategies else "No demostrado",
                "evidencia": "Existe LeaveOneLocationOut en metrics_by_pair.csv; corresponde a lugares observados en el estudio, no a un sitio agroclimaticamente nuevo.",
            },
            {
                "pregunta": "Generalizacion a temporada nueva",
                "estado": "Parcialmente demostrado" if "LeaveOneSeasonOut" in strategies else "No demostrado",
                "evidencia": "Existe LeaveOneSeasonOut; el alcance depende de las temporadas presentes en los archivos.",
            },
            {
                "pregunta": "Generalizacion a variedad nueva",
                "estado": "Parcialmente demostrado" if "LeaveOneVarietyOut" in strategies else "No demostrado",
                "evidencia": "Existe LeaveOneVarietyOut; no prueba variedades fuera del universo registrado.",
            },
            {
                "pregunta": "Uso productivo con incertidumbre calibrada",
                "estado": "No demostrado",
                "evidencia": "No hay intervalos predictivos ni evaluacion por horizonte en los archivos.",
            },
        ]
    )


def render_report(data: InputData) -> str:
    metrics = data.metrics.copy()
    selected = data.selected
    selected_metrics_rows = metrics.loc[selected_filter(metrics, selected)]
    selected_row = selected_metrics_rows.iloc[0] if len(selected_metrics_rows) == 1 else None
    selected_oof = data.oof.loc[selected_filter(data.oof, selected)].copy()
    if selected_oof.empty:
        raise ValueError("No hay OOF para la alternativa seleccionada; no se puede generar reporte.")

    recalculated = recompute_selected_metrics(selected_oof, selected)
    consistency = compare_selected_metrics(selected_row, recalculated, data.warnings)
    cold_global = compute_prediction_metrics(
        selected_oof,
        "fecha_brotacion_camara_50",
        "pred_fecha_fin_frio",
        "error_fecha_fin_frio_dias",
        "temporada",
    )
    bud_global = compute_prediction_metrics(
        selected_oof,
        "fecha_brotacion_campo_50",
        "pred_fecha_brotacion",
        "error_fecha_brotacion_dias",
        "temporada",
    )
    for label, metric_values in [("brotacion", bud_global), ("termino de frio", cold_global)]:
        calendar_r2 = metric_values.get("r2_calendario_absoluto", np.nan)
        season_r2 = metric_values.get("r2_intra_temporada", np.nan)
        if pd.notna(calendar_r2) and pd.notna(season_r2) and calendar_r2 > 0.98 and (calendar_r2 - season_r2) > 0.10:
            warn(
                data.warnings,
                "El R2 calendario absoluto para "
                f"{label} es {calendar_r2:.3f}, pero el R2 intra-temporada es "
                f"{season_r2:.3f}; el valor calendario esta inflado por mezclar anos."
            )
    charts = build_charts(metrics, selected_oof)
    segment_outputs = build_segment_outputs(data.segments, selected_oof)
    extremes = build_extreme_cases(selected_oof)
    propagation = propagation_analysis(selected_oof)
    coverage = coverage_tables(selected_oof)
    limitations_html = markdown_to_html(data.limitations_markdown)
    git_commit = get_git_commit()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    primary_strategy = selected["validation_strategy"]
    ranked_primary = metrics.loc[metrics["validation_strategy"] == primary_strategy].sort_values("mae_fecha_brotacion_dias")
    ranked_cold = metrics.loc[metrics["validation_strategy"] == primary_strategy].sort_values("mae_fecha_fin_frio_dias")

    template = jinja2.Environment(autoescape=True).from_string(REPORT_TEMPLATE)
    return template.render(
        plotlyjs=get_plotlyjs(),
        generated_at=generated_at,
        git_commit=git_commit,
        selected=selected,
        selected_pair=selected.get("thermal_pair"),
        selected_model=selected.get("model_name"),
        selected_strategy=selected.get("validation_strategy"),
        warnings=data.warnings,
        cards=metric_cards(recalculated, selected_oof),
        observations_evaluated=format_int(recalculated["n_observaciones"]),
        executive_text=build_executive_text(recalculated, selected, data.metrics),
        methodology=build_methodology_text(data.schemas, selected, selected_oof, metrics),
        metrics_table=df_to_html_table(metrics.sort_values(["validation_strategy", "mae_fecha_brotacion_dias"])),
        ranking_bud_table=df_to_html_table(ranked_primary.head(15)),
        ranking_cold_table=df_to_html_table(ranked_cold.head(15)),
        consistency_table=df_to_html_table(consistency),
        global_bud_table=df_to_html_table(pd.DataFrame([bud_global]).T.reset_index().rename(columns={"index": "metrica", 0: "valor"})),
        global_cold_table=df_to_html_table(pd.DataFrame([cold_global]).T.reset_index().rename(columns={"index": "metrica", 0: "valor"})),
        charts=charts,
        segment_tables=segment_outputs["tables"],
        segment_charts=segment_outputs["charts"],
        segment_best_table=segment_outputs["best_table"],
        segment_worst_table=segment_outputs["worst_table"],
        segment_bias_table=segment_outputs["high_bias_table"],
        segment_heatmap=segment_outputs["heatmap"],
        low_support_count=segment_outputs["low_support_count"],
        extremes=extremes,
        propagation=propagation,
        propagation_quadrants=df_to_html_table(propagation["quadrants"]),
        coverage_tables=coverage,
        coverage_summary=build_coverage_summary(selected_oof, recalculated),
        limitations_html=limitations_html,
        risk_synthesis=df_to_html_table(pd.DataFrame(build_risk_synthesis(data.limitations_markdown, segment_outputs["low_support_count"]))),
        generalization_table=df_to_html_table(generalization_table(metrics, selected_oof)),
        selected_json=json.dumps(selected, indent=2, ensure_ascii=False),
        selected_interpretation=build_selected_interpretation(selected),
        conclusions=build_conclusions(recalculated, metrics, data.warnings),
        inventory_table=df_to_html_table(data.file_inventory),
        schemas=data.schemas,
        used_columns=build_used_columns(),
        python_version=sys.version.replace("\n", " "),
        pandas_version=pd.__version__,
        numpy_version=np.__version__,
        plotly_version=plotly.__version__,
        MIN_SEGMENT_SUPPORT=MIN_SEGMENT_SUPPORT,
    )


def build_executive_text(recalculated: dict[str, float], selected: dict[str, Any], metrics: pd.DataFrame) -> str:
    best = metrics.loc[metrics["validation_strategy"] == selected["validation_strategy"]].sort_values("mae_fecha_brotacion_dias").head(1)
    selection_note = "La alternativa seleccionada coincide con el menor MAE de brotacion dentro de la estrategia registrada." if not best.empty and best.iloc[0]["thermal_pair"] == selected["thermal_pair"] and best.iloc[0]["model_name"] == selected["model_name"] else "La alternativa seleccionada no coincide necesariamente con el primer lugar del ranking visible; revisar advertencias y cobertura."
    return (
        f"En la estrategia {selected['validation_strategy']}, la alternativa {selected['thermal_pair']} "
        f"con regresor {selected['model_name']} alcanza un MAE recalculado de brotacion de "
        f"{format_number(recalculated['mae_fecha_brotacion_dias'], ' dias')} sobre "
        f"{format_int(recalculated['n_pred_fecha_brotacion_validas'])} predicciones validas. "
        f"El R2 de fechas mostrado en las tarjetas se calcula intra-temporada para evitar inflacion por ano calendario. "
        f"{selection_note} Estos resultados describen el desempeno dentro del esquema de validacion disponible; "
        "no equivalen por si solos a validacion operacional con incertidumbre calibrada."
    )


def build_methodology_text(schemas: dict[str, dict[str, Any]], selected: dict[str, Any], selected_oof: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
    return [
        "La unidad de analisis verificable es la fila de oof_predictions.csv identificada por unidad_experimental_id.",
        f"La pareja termica y el regresor seleccionados se leen desde selected_pair.json: {selected['thermal_pair']} / {selected['model_name']}.",
        "Una prediccion out-of-fold se interpreta como una prediccion asociada a un fold_id. Los archivos permiten verificar la presencia de fold_id y unicidad por unidad y combinacion, pero no contienen los indices de entrenamiento.",
        "El criterio de seleccion verificable es mae_fecha_brotacion_dias dentro de metrics_by_pair.csv; el reporte recalcula las metricas desde oof_predictions.csv y advierte discrepancias.",
        f"Las estrategias de validacion registradas son: {', '.join(sorted(metrics['validation_strategy'].dropna().astype(str).unique()))}.",
        "Las observaciones excluidas por falta de cobertura termica solo pueden inferirse por n_observaciones y predicciones validas en metrics_by_pair.csv; no hay una columna de razon de exclusion en los archivos.",
    ]


def build_coverage_summary(selected_oof: pd.DataFrame, recalculated: dict[str, float]) -> dict[str, str]:
    total = len(selected_oof)
    complete = int(recalculated["n_pred_fecha_brotacion_validas"])
    excluded = total - complete
    return {
        "total": format_int(total),
        "complete": format_int(complete),
        "excluded": format_int(excluded),
        "percent": format_number(complete / total * 100.0 if total else np.nan, "%"),
        "reason": "Los archivos no registran razones de exclusion; se identifica cobertura por disponibilidad de fechas predichas.",
    }


def build_risk_synthesis(markdown_text: str, low_support_count: int) -> list[dict[str, str]]:
    return [
        {"categoria": "Leakage", "estado": "Riesgo documentado", "detalle": "El markdown de limitaciones advierte sobre uso de fecha real de termino de frio para entrenamiento del modelo de calor."},
        {"categoria": "Targets termicos derivados", "estado": "Exploratorio", "detalle": "Los requisitos se derivan de fechas observadas y temperatura; no son mediciones fisiologicas directas en los reports."},
        {"categoria": "Propagacion frio a calor", "estado": "Presente", "detalle": "El reporte recalcula errores de termino de frio y brotacion y analiza su relacion sin afirmar causalidad."},
        {"categoria": "Lugares no observados", "estado": "Parcial", "detalle": "Existe LeaveOneLocationOut, pero no prueba un sitio fuera del universo agroclimatico registrado."},
        {"categoria": "Temporadas no observadas", "estado": "Parcial", "detalle": "Existe LeaveOneSeasonOut en metrics_by_pair.csv."},
        {"categoria": "Variedades no observadas", "estado": "Parcial", "detalle": "Existe LeaveOneVarietyOut en metrics_by_pair.csv."},
        {"categoria": "Cobertura termica incompleta", "estado": "A monitorear", "detalle": "Se informa cobertura por combinacion y por segmento cuando las columnas lo permiten."},
        {"categoria": "Segmentos con bajo soporte", "estado": "A monitorear", "detalle": f"{low_support_count} segmentos tienen n menor que {MIN_SEGMENT_SUPPORT}."},
        {"categoria": "Incertidumbre", "estado": "No modelada", "detalle": "No hay intervalos predictivos ni calibracion de incertidumbre en los archivos."},
    ]


def build_selected_interpretation(selected: dict[str, Any]) -> str:
    return (
        "Los archivos de reports contienen el identificador tecnico de la pareja termica, "
        "pero no incluyen metadata formal que defina cada componente. Por esa razon el reporte "
        f"muestra {selected['thermal_pair']} como identificador verificable sin inferir una definicion fisiologica adicional."
    )


def build_conclusions(recalculated: dict[str, float], metrics: pd.DataFrame, warnings: list[str]) -> dict[str, list[str]]:
    return {
        "hallazgos": [
            f"La alternativa seleccionada tiene MAE recalculado de brotacion de {format_number(recalculated['mae_fecha_brotacion_dias'], ' dias')}.",
            f"El porcentaje recalculado con error absoluto de brotacion <= 7 dias es {format_number(recalculated['porcentaje_error_menor_igual_7_dias'], '%')}.",
            f"metrics_by_pair.csv contiene {len(metrics)} combinaciones de pareja termica, modelo y estrategia de validacion.",
        ],
        "limitaciones": [
            "La evidencia de generalizacion debe leerse por estrategia; un buen MAE global no demuestra por si solo generalizacion operacional.",
            "No existen intervalos predictivos, evaluacion por horizonte ni razones detalladas de exclusion en los archivos de entrada.",
            "Los modelos mas complejos no deben recomendarse automaticamente sin superar baselines bajo las mismas validaciones.",
        ],
        "recomendaciones": [
            "Priorizar validacion externa por lugar y temporada con datos nuevos.",
            "Agregar intervalos predictivos y evaluacion por horizonte operativo.",
            "Revisar segmentos con mayor MAE, mayor sesgo o soporte menor que el umbral configurado.",
            "Mantener monitoreo de cobertura termica antes de comparar alternativas por MAE.",
        ],
        "advertencias": warnings,
    }


def build_used_columns() -> dict[str, list[str]]:
    return {
        "seleccion": SELECTED_KEYS,
        "metricas_agregadas": REQUIRED_METRICS_COLUMNS,
        "predicciones_oof": REQUIRED_OOF_COLUMNS + [column for column in IDENTIFIER_COLUMNS if column not in REQUIRED_OOF_COLUMNS],
        "segmentos": REQUIRED_SEGMENT_COLUMNS,
    }


REPORT_TEMPLATE = r"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Evaluación del modelamiento térmico del receso invernal en cerezos</title>
<script>{{ plotlyjs|safe }}</script>
<style>
:root { --bg:#f8fafc; --panel:#ffffff; --text:#111827; --muted:#64748b; --line:#dbe3ef; --accent:#0f766e; --warn:#b45309; --bad:#b91c1c; --soft:#e6f4f1; }
* { box-sizing: border-box; }
body { margin:0; font-family: Arial, sans-serif; background:var(--bg); color:var(--text); line-height:1.45; }
.layout { display:grid; grid-template-columns: 280px minmax(0, 1fr); min-height:100vh; }
nav { position:sticky; top:0; height:100vh; overflow:auto; padding:24px 18px; background:#0f172a; color:white; }
nav h2 { font-size:16px; margin:0 0 16px; }
nav a { display:block; color:#dbeafe; text-decoration:none; padding:7px 0; font-size:13px; }
main { max-width:1280px; padding:32px; }
section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:24px; margin:0 0 22px; box-shadow:0 1px 2px rgba(15,23,42,0.04); }
h1 { font-size:32px; margin:0 0 10px; letter-spacing:0; }
h2 { font-size:22px; margin:0 0 16px; }
h3 { font-size:17px; margin:20px 0 10px; }
.muted { color:var(--muted); }
.badge { display:inline-block; padding:5px 8px; border-radius:6px; background:var(--soft); color:var(--accent); font-weight:700; font-size:12px; }
.warning { border-left:5px solid var(--warn); background:#fff7ed; padding:12px 14px; margin:10px 0; }
.error { border-left:5px solid var(--bad); background:#fef2f2; padding:12px 14px; margin:10px 0; }
.cards { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:12px; margin:18px 0; }
.card { border:1px solid var(--line); border-radius:8px; padding:14px; background:#fbfdff; }
.card .label { color:var(--muted); font-size:12px; margin-bottom:6px; }
.card .value { font-size:21px; font-weight:700; }
.grid-2 { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:16px; }
.chart { margin:18px 0; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
.table-wrap { max-height:520px; overflow:auto; border:1px solid var(--line); border-radius:8px; margin:12px 0; }
table.data-table { width:100%; border-collapse:collapse; font-size:12px; background:white; }
.data-table th { position:sticky; top:0; background:#eef2f7; text-align:left; padding:8px; border-bottom:1px solid var(--line); z-index:1; }
.data-table td { padding:7px 8px; border-bottom:1px solid #edf2f7; vertical-align:top; }
pre { background:#0f172a; color:#e5e7eb; padding:14px; border-radius:8px; overflow:auto; }
code { background:#e5e7eb; border-radius:4px; padding:1px 4px; }
.conclusion-list li { margin-bottom:6px; }
@media print { nav { display:none; } .layout { display:block; } main { max-width:none; padding:0; } section { break-inside:avoid; box-shadow:none; } .chart { break-inside:avoid; } }
@media (max-width: 980px) { .layout { display:block; } nav { position:relative; height:auto; } .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } .grid-2 { grid-template-columns:1fr; } main { padding:16px; } }
</style>
</head>
<body>
<div class="layout">
<nav>
<h2>Indice</h2>
<a href="#portada">1. Portada</a>
<a href="#resumen">2. Resumen ejecutivo</a>
<a href="#metodologia">3. Metodologia</a>
<a href="#comparacion">4. Comparacion</a>
<a href="#oof">5. Predicciones OOF</a>
<a href="#propagacion">6. Propagacion</a>
<a href="#segmentos">7. Segmentos</a>
<a href="#extremos">8. Casos extremos</a>
<a href="#cobertura">9. Cobertura</a>
<a href="#leakage">10. Leakage y riesgos</a>
<a href="#generalizacion">11. Generalizacion</a>
<a href="#seleccionado">12. Modelo seleccionado</a>
<a href="#conclusiones">13. Conclusiones</a>
<a href="#trazabilidad">14. Trazabilidad</a>
</nav>
<main>
<section id="portada">
<h1>Evaluación del modelamiento térmico del receso invernal en cerezos</h1>
<p class="muted">Generado: {{ generated_at }} | Commit Git: {{ git_commit }}</p>
<p><span class="badge">Pareja seleccionada: {{ selected_pair }}</span> <span class="badge">Modelo: {{ selected_model }}</span> <span class="badge">Validacion: {{ selected_strategy }}</span></p>
<p>Observaciones evaluadas: <strong>{{ observations_evaluated }}</strong></p>
{% if warnings %}
<div class="error"><strong>Advertencias de consistencia:</strong><ul>{% for warning in warnings %}<li>{{ warning }}</li>{% endfor %}</ul></div>
{% else %}
<div class="warning">No se detectaron inconsistencias criticas en las validaciones implementadas.</div>
{% endif %}
</section>

<section id="resumen">
<h2>2. Resumen ejecutivo</h2>
<div class="cards">{% for card in cards %}<div class="card"><div class="label">{{ card.label }}</div><div class="value">{{ card.value }}</div></div>{% endfor %}</div>
<p class="muted">Los porcentajes <code>error &lt;= 3 dias</code> y <code>error &lt;= 7 dias</code> se calculan solo para la fecha de brotacion y representan la fraccion de predicciones con error absoluto menor o igual al umbral.</p>
<p>{{ executive_text }}</p>
</section>

<section id="metodologia">
<h2>3. Metodologia reconstruida</h2>
<ul>{% for item in methodology %}<li>{{ item }}</li>{% endfor %}</ul>
<h3>Columnas reales identificadas</h3>
{% for name, schema in schemas.items() %}
<p><strong>{{ name }}</strong>: {{ schema.rows }} filas, {{ schema.columns_count }} columnas.</p>
<pre>{{ schema.columns }}</pre>
{% endfor %}
</section>

<section id="comparacion">
<h2>4. Comparacion de parejas termicas y modelos</h2>
<p>La tabla completa se ordena por estrategia de validacion y MAE de brotacion. La comparacion debe considerar cobertura, no solo error.</p>
<div class="grid-2"><div class="chart">{{ charts.mae_bud|safe }}</div><div class="chart">{{ charts.mae_cold|safe }}</div></div>
<div class="grid-2"><div class="chart">{{ charts.coverage|safe }}</div><div class="chart">{{ charts.tradeoff|safe }}</div></div>
<h3>Ranking por MAE de brotacion</h3><div class="table-wrap">{{ ranking_bud_table|safe }}</div>
<h3>Ranking por MAE de termino de frio</h3><div class="table-wrap">{{ ranking_cold_table|safe }}</div>
<h3>Tabla completa</h3><div class="table-wrap">{{ metrics_table|safe }}</div>
<p class="warning">Trade-off: una alternativa con menor MAE pero menor cobertura no debe presentarse automaticamente como superior. La seleccion debe revisar estabilidad por validacion, soporte y complejidad.</p>
</section>

<section id="oof">
<h2>5. Evaluacion global de predicciones out-of-fold</h2>
<h3>Brotacion</h3>
<div class="table-wrap">{{ global_bud_table|safe }}</div>
<div class="grid-2"><div class="chart">{{ charts.brotacion_obs_pred|safe }}</div><div class="chart">{{ charts.brotacion_signed_hist|safe }}</div></div>
<div class="grid-2"><div class="chart">{{ charts.brotacion_abs_hist|safe }}</div><div class="chart">{{ charts.brotacion_residuals|safe }}</div></div>
<div class="chart">{{ charts.brotacion_ordered_abs|safe }}</div>
{% if charts.fold_errors is defined %}<div class="chart">{{ charts.fold_errors|safe }}</div>{% endif %}
<h3>Termino de frio</h3>
<div class="table-wrap">{{ global_cold_table|safe }}</div>
<div class="grid-2"><div class="chart">{{ charts.fin_frio_obs_pred|safe }}</div><div class="chart">{{ charts.fin_frio_signed_hist|safe }}</div></div>
<div class="grid-2"><div class="chart">{{ charts.fin_frio_abs_hist|safe }}</div><div class="chart">{{ charts.fin_frio_residuals|safe }}</div></div>
<div class="chart">{{ charts.fin_frio_ordered_abs|safe }}</div>
</section>

<section id="propagacion">
<h2>6. Propagacion del error</h2>
<p>Correlacion Pearson: <strong>{{ propagation.pearson|round(3) }}</strong>. Correlacion Spearman: <strong>{{ propagation.spearman|round(3) }}</strong>. Estas correlaciones no implican causalidad.</p>
<div class="chart">{{ charts.propagation|safe }}</div>
<h3>Cuadrantes</h3><div class="table-wrap">{{ propagation_quadrants|safe }}</div>
<ul>{% for note in propagation.notes %}<li>{{ note }}</li>{% endfor %}</ul>
</section>

<section id="segmentos">
<h2>7. Desempeno por segmento</h2>
<p class="warning">Segmentos con n &lt; {{ MIN_SEGMENT_SUPPORT if MIN_SEGMENT_SUPPORT is defined else 20 }} se marcan como soporte bajo. No deben sostener conclusiones fuertes.</p>
<p>Segmentos con bajo soporte: {{ low_support_count }}</p>
<h3>Mejores segmentos</h3><div class="table-wrap">{{ segment_best_table|safe }}</div>
<h3>Peores segmentos</h3><div class="table-wrap">{{ segment_worst_table|safe }}</div>
<h3>Mayor sesgo absoluto</h3><div class="table-wrap">{{ segment_bias_table|safe }}</div>
{% if segment_heatmap %}<div class="chart">{{ segment_heatmap|safe }}</div>{% endif %}
{% for name, chart in segment_charts.items() %}
<h3>{{ name }}</h3><div class="chart">{{ chart|safe }}</div><div class="table-wrap">{{ segment_tables[name]|safe }}</div>
{% endfor %}
</section>

<section id="extremos">
<h2>8. Analisis de casos extremos</h2>
<h3>15 mayores desviaciones de brotacion</h3><div class="table-wrap">{{ extremes.brotacion_abs|safe }}</div>
<h3>15 mayores desviaciones de termino de frio</h3><div class="table-wrap">{{ extremes.frio_abs|safe }}</div>
<h3>Principales sobreestimaciones de brotacion</h3><div class="table-wrap">{{ extremes.sobreestimacion|safe }}</div>
<h3>Principales subestimaciones de brotacion</h3><div class="table-wrap">{{ extremes.subestimacion|safe }}</div>
</section>

<section id="cobertura">
<h2>9. Cobertura termica y datos faltantes</h2>
<div class="cards"><div class="card"><div class="label">Total seleccionado</div><div class="value">{{ coverage_summary.total }}</div></div><div class="card"><div class="label">Cobertura completa</div><div class="value">{{ coverage_summary.complete }}</div></div><div class="card"><div class="label">Excluidas/sin prediccion</div><div class="value">{{ coverage_summary.excluded }}</div></div><div class="card"><div class="label">% cobertura</div><div class="value">{{ coverage_summary.percent }}</div></div></div>
<p>{{ coverage_summary.reason }}</p>
{% for name, table in coverage_tables.items() %}<h3>Cobertura por {{ name }}</h3><div class="table-wrap">{{ table|safe }}</div>{% endfor %}
</section>

<section id="leakage">
<h2>10. Leakage, limitaciones y riesgos</h2>
<h3>Sintesis visual</h3><div class="table-wrap">{{ risk_synthesis|safe }}</div>
<h3>Markdown completo renderizado</h3>
<div>{{ limitations_html|safe }}</div>
</section>

<section id="generalizacion">
<h2>11. Evaluacion de generalizacion</h2>
<div class="table-wrap">{{ generalization_table|safe }}</div>
<p>Las etiquetas distinguen evidencia disponible en archivos de afirmaciones todavia no verificables. Un buen MAE global no demuestra por si solo generalizacion a condiciones nuevas.</p>
</section>

<section id="seleccionado">
<h2>12. Interpretacion del modelo seleccionado</h2>
<p>{{ selected_interpretation }}</p>
<pre>{{ selected_json }}</pre>
<h3>Metricas leidas versus recalculadas</h3>
<div class="table-wrap">{{ consistency_table|safe }}</div>
</section>

<section id="conclusiones">
<h2>13. Conclusiones</h2>
<h3>Hallazgos respaldados</h3><ul class="conclusion-list">{% for item in conclusions.hallazgos %}<li>{{ item }}</li>{% endfor %}</ul>
<h3>Limitaciones</h3><ul class="conclusion-list">{% for item in conclusions.limitaciones %}<li>{{ item }}</li>{% endfor %}</ul>
<h3>Recomendaciones tecnicas</h3><ul class="conclusion-list">{% for item in conclusions.recomendaciones %}<li>{{ item }}</li>{% endfor %}</ul>
</section>

<section id="trazabilidad">
<h2>14. Trazabilidad</h2>
<h3>Archivos utilizados</h3><div class="table-wrap">{{ inventory_table|safe }}</div>
<h3>Columnas utilizadas por analisis</h3>
{% for name, columns in used_columns.items() %}<p><strong>{{ name }}</strong></p><pre>{{ columns }}</pre>{% endfor %}
<h3>Entorno</h3>
<p>Python: {{ python_version }}</p>
<p>pandas: {{ pandas_version }} | numpy: {{ numpy_version }} | plotly: {{ plotly_version }}</p>
</section>
</main>
</div>
</body>
</html>
"""


def generate_report(
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    data = load_inputs(reports_dir)
    html_text = render_report(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    if output_path.stat().st_size == 0:
        raise RuntimeError(f"Reporte generado vacio: {output_path}")
    print(f"Reporte HTML generado: {output_path}")
    return output_path


def main() -> None:
    generate_report()


if __name__ == "__main__":
    main()

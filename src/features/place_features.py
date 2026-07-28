from __future__ import annotations

import pandas as pd

from src.thermal_models import chill_hours_7_2, chill_hours_10, growing_degree_hours


def _summarize_place_history(history: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {}
    temps = history["temperatura_c"]
    summary["temp_media_historica"] = float(temps.mean())
    summary["temp_min_historica"] = float(temps.min())
    summary["temp_max_historica"] = float(temps.max())
    summary["amplitud_termica_historica"] = (
        summary["temp_max_historica"] - summary["temp_min_historica"]
    )
    summary["horas_helada_historicas"] = float((temps <= 0.0).sum())
    summary["horas_calidas_historicas"] = float((temps >= 25.0).sum())
    summary["horas_bajo_7_2_historicas"] = float(chill_hours_7_2(temps).sum())
    summary["horas_bajo_10_historicas"] = float(chill_hours_10(temps).sum())
    summary["hgc_historicas"] = float(growing_degree_hours(temps).sum())
    summary["variabilidad_interanual_temp"] = float(
        history.groupby("year")["temperatura_c"].mean().std(ddof=0)
    )
    summary["variabilidad_interanual_frio"] = float(
        history.groupby("year")["temperatura_c"]
        .apply(lambda values: chill_hours_7_2(values).sum())
        .std(ddof=0)
    )
    summary["variabilidad_interanual_calor"] = float(
        history.groupby("year")["temperatura_c"]
        .apply(lambda values: growing_degree_hours(values).sum())
        .std(ddof=0)
    )

    monthly = history.groupby("month")["temperatura_c"].mean()
    for month in range(1, 13):
        summary[f"temp_media_historica_mes_{month:02d}"] = float(
            monthly.get(month, temps.mean())
        )

    return summary


def build_place_season_features(
    phenology: pd.DataFrame,
    temperatures: pd.DataFrame,
) -> pd.DataFrame:
    metadata = (
        temperatures[
            ["codigo_lugar", "lugar", "condicion", "latitud", "longitud"]
        ]
        .drop_duplicates("codigo_lugar")
        .set_index("codigo_lugar")
    )

    records: list[dict[str, object]] = []
    keys = phenology[["codigo_lugar", "temporada"]].drop_duplicates()
    for row in keys.itertuples(index=False):
        history = temperatures.loc[
            (temperatures["codigo_lugar"] == row.codigo_lugar)
            & (temperatures["year"] < int(row.temporada))
        ]
        if history.empty:
            history = temperatures.loc[
                (temperatures["codigo_lugar"] == row.codigo_lugar)
                & (temperatures["year"] != int(row.temporada))
            ]

        if history.empty:
            raise ValueError(
                f"No hay historia termica para {row.codigo_lugar} temporada {row.temporada}"
            )

        record: dict[str, object] = {
            "codigo_lugar": row.codigo_lugar,
            "temporada": int(row.temporada),
        }
        record.update(metadata.loc[row.codigo_lugar].to_dict())
        record.update(_summarize_place_history(history))
        records.append(record)

    return pd.DataFrame(records)

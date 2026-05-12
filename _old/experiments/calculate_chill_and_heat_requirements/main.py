from pathlib import Path

import numpy as np
import pandas as pd


def chill_hours(df_hourly: pd.DataFrame, threshold_c: float = 7.2) -> pd.Series:
    """Modelo Horas Frio: 1 si temp <= threshold, sino 0."""
    t = df_hourly["temp_c"].to_numpy()
    return pd.Series((t <= threshold_c).astype(float), index=df_hourly.index, name="chill_unit")


def utah_cu(df_hourly: pd.DataFrame) -> pd.Series:
    """Modelo Utah (CU) por rangos con aportes positivos y negativos."""
    t = df_hourly["temp_c"].to_numpy()
    cu = np.zeros_like(t, dtype=float)

    cu[(t <= 1.4)] = 0.0
    cu[(t >= 1.5) & (t <= 2.4)] = 0.5
    cu[(t >= 2.5) & (t <= 9.1)] = 1.0
    cu[(t >= 9.2) & (t <= 12.4)] = 0.5
    cu[(t >= 12.5) & (t <= 15.9)] = 0.0
    cu[(t >= 16.0) & (t <= 18.0)] = -0.5
    cu[(t > 18.0)] = -1.0

    return pd.Series(cu, index=df_hourly.index, name="chill_unit")


def utah_positivo(df_hourly: pd.DataFrame) -> pd.Series:
    """Modelo Utah truncado: no permite aportes negativos."""
    t = df_hourly["temp_c"].to_numpy()
    cu = np.zeros_like(t, dtype=float)

    cu[(t <= 1.4)] = 0.0
    cu[(t >= 1.5) & (t <= 2.4)] = 0.5
    cu[(t >= 2.5) & (t <= 9.1)] = 1.0
    cu[(t >= 9.2) & (t <= 12.4)] = 0.5
    cu[(t >= 12.5) & (t <= 15.9)] = 0.0
    cu[(t >= 16.0) & (t <= 18.0)] = 0.0
    cu[(t > 18.0)] = 0.0

    return pd.Series(cu, index=df_hourly.index, name="chill_unit")


def heat_hours(df_hourly: pd.DataFrame, threshold_c: float = 18.0) -> pd.Series:
    """Modelo Horas Calor: 1 si temp >= threshold, sino 0."""
    t = df_hourly["temp_c"].to_numpy()
    return pd.Series((t >= threshold_c).astype(float), index=df_hourly.index, name="heat_unit")


def hgc(df_hourly: pd.DataFrame, base_c: float = 4.5, cap_c: float = 25.0) -> pd.Series:
    """
    Horas Grado de Crecimiento (HGC) simple:
    max(0, min(T, cap) - base)
    """
    t = df_hourly["temp_c"].to_numpy()
    te = np.minimum(t, cap_c) - base_c
    te = np.maximum(te, 0.0)
    return pd.Series(te, index=df_hourly.index, name="heat_unit")


def normalize_zone_name(raw_zone: str) -> str:
    zone = " ".join(str(raw_zone).strip().lower().split())
    mapping = {
        "sta magdalena (los niches 2)": "santa magdalena",
        "teno oriente": "teno prado",
        "teno montaa": "teno don sergio",
        "teno monta\ufffda": "teno don sergio",
        "teno poniente": "santa ana",
        "wapri (los niches 1)": "wapri",
    }
    return mapping.get(zone, zone)


def normalize_column_name(column_name: str) -> str:
    return "".join(ch for ch in str(column_name).strip().lower() if ch.isalnum())


def get_column_by_normalized_name(df: pd.DataFrame, target_name: str) -> str:
    target_norm = normalize_column_name(target_name)
    for column in df.columns:
        if normalize_column_name(column) == target_norm:
            return column
    raise KeyError(f"No se encontró una columna equivalente a '{target_name}' en {df.columns.tolist()}")


def load_temperatures_by_zone(maestro_path: Path, temp_pattern: str) -> dict:
    maestro_zonas = pd.read_csv(maestro_path)
    temperaturas = {}

    for row in maestro_zonas.itertuples(index=False):
        zona_id = int(getattr(row, "zona_id"))
        zona_name = str(getattr(row, "zona")).strip().lower()
        temp_path = Path(temp_pattern.format(zona_id + 1))

        df_temp = pd.read_csv(temp_path)
        df_temp["fecha_datetime"] = pd.to_datetime(
            df_temp["fecha_datetime"], dayfirst=True, errors="raise", format='mixed'
        )
        df_temp["temperatura"] = pd.to_numeric(df_temp["temperatura"], errors="raise")
        df_temp = df_temp.dropna(subset=["fecha_datetime", "temperatura"]).sort_values("fecha_datetime")
        df_temp = df_temp.ffill()

        temperaturas[zona_name] = df_temp

    return temperaturas


def estimate_row_chill(row: pd.Series, temperaturas: dict, models: dict) -> list:
    zona = row["zona_normalizada"]
    if zona not in temperaturas:
        return []

    start = row["caida_de_hojas_date"]
    end = row["brotacion_en_camara_50_date"]
    if pd.isna(start) or pd.isna(end) or end < start:
        return []

    df_temp = temperaturas[zona]
    mask = (df_temp["fecha_datetime"] >= start) & (
        df_temp["fecha_datetime"] < end
    )
    window = df_temp.loc[mask, ["fecha_datetime", "temperatura"]].copy()
    if window.empty:
        return []

    window = window.rename(columns={"fecha_datetime": "datetime", "temperatura": "temp_c"})

    rows = []
    for model_name, model_fn in models.items():
        chill_units = model_fn(window)
        rows.append(
            {
                "key": row["key"],
                "zona": zona,
                "variedad": row["variedad"],
                "ue_int": row.get("ue_int"),
                "modelo": model_name,
                "horas_frio_estimadas": float(chill_units.sum()),
                "n_horas": int(chill_units.shape[0]),
                "fecha_inicio": start,
                "fecha_fin": end,
            }
        )

    return rows


def estimate_row_heat(row: pd.Series, temperaturas: dict, models: dict) -> list:
    zona = row["zona_normalizada"]
    if zona not in temperaturas:
        return []

    start = row["brotacion_en_camara_date"]
    end = row["brotacion_en_campo_date"]
    if pd.isna(start) or pd.isna(end) or end < start:
        return []

    df_temp = temperaturas[zona]
    mask = (df_temp["fecha_datetime"] >= start) & (df_temp["fecha_datetime"] < end)
    window = df_temp.loc[mask, ["fecha_datetime", "temperatura"]].copy()
    if window.empty:
        return []

    window = window.rename(columns={"fecha_datetime": "datetime", "temperatura": "temp_c"})

    rows = []
    for model_name, model_fn in models.items():
        heat_units = model_fn(window)
        rows.append(
            {
                "key": row.get("key", row.get("Key")),
                "zona": zona,
                "variedad": row["variedad"],
                "ue_int": row.get("ue_int"),
                "modelo": model_name,
                "horas_calor_estimadas": float(heat_units.sum()),
                "n_horas": int(heat_units.shape[0]),
                "fecha_inicio": start,
                "fecha_fin": end,
            }
        )

    return rows


def summarize_results(detail_rows: list, metric_name: str, output_prefix: str) -> pd.DataFrame:
    df_detail = pd.DataFrame(detail_rows)
    if df_detail.empty:
        raise ValueError(f"No se pudieron calcular {metric_name} para los registros de control.")

    value_col = f"{output_prefix}_estimadas"
    summary_prefix = output_prefix

    df_summary = (
        df_detail.groupby(["zona", "variedad", "modelo"], as_index=False)
        .agg(
            **{
                f"{summary_prefix}_promedio": (value_col, "mean"),
                f"{summary_prefix}_std": (value_col, "std"),
                f"{summary_prefix}_min": (value_col, "min"),
                f"{summary_prefix}_max": (value_col, "max"),
                "n_observaciones": ("key", "count"),
            }
        )
        .sort_values(["zona", "variedad", "modelo"])
    )

    return df_summary


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    temp_data_dir = base_dir.parent / "estimate_chill_requirements" / "data"

    maestro_zonas_path = temp_data_dir / "maestro_zonas.csv"
    temperatura_path_pattern = str(temp_data_dir / "temp_zone={}.csv")
    temperaturas = load_temperatures_by_zone(maestro_zonas_path, temperatura_path_pattern)

    # -------- Horas frío --------
    frio_path = data_dir / "horas_frio.csv"
    df_frio = pd.read_csv(frio_path, sep=";")
    df_frio_control = df_frio.loc[df_frio["tratamiento"].astype(str).str.lower() == "control"].copy()

    df_frio_control["caida_de_hojas_date"] = pd.to_datetime(
        df_frio_control["caida_de_hojas_date"], dayfirst=True, format='mixed', errors="raise"
    )
    df_frio_control["brotacion_en_camara_50_date"] = pd.to_datetime(
        df_frio_control["brotacion_en_camara_50_date"], dayfirst=True, errors="raise", format='mixed'
    )
    df_frio_control["zona_normalizada"] = df_frio_control["zona"].apply(normalize_zone_name)

    models = {
        "chill_hours": chill_hours,
        "utah_cu": utah_cu,
        "utah_positivo": utah_positivo,
    }

    detail_rows = []
    for _, row in df_frio_control.iterrows():
        detail_rows.extend(estimate_row_chill(row, temperaturas, models))

    df_summary_frio = summarize_results(detail_rows, "horas de frio", "horas_frio")

    output_path_frio = data_dir / "horas_frio_control_por_zona_variedad_modelo.csv"
    df_summary_frio.to_csv(output_path_frio, index=False)

    print("\n=== HORAS FRÍO ===")
    print(df_summary_frio.to_string(index=False))
    print(f"\nArchivo generado: {output_path_frio}")

    # -------- Horas calor --------
    calor_path = data_dir / "horas_calor.csv"
    df_calor = pd.read_csv(calor_path, sep=";")
    df_calor_control = df_calor.loc[df_calor["tratamiento"].astype(str).str.lower() == "control"].copy()

    calor_camara_col = df_calor_control.columns[1]
    calor_campo_col = df_calor_control.columns[2]

    df_calor_control["brotacion_en_camara_date"] = pd.to_datetime(
        df_calor_control[calor_camara_col], dayfirst=True, format="mixed", errors="raise"
    )
    df_calor_control["brotacion_en_campo_date"] = pd.to_datetime(
        df_calor_control[calor_campo_col], dayfirst=True, format="mixed", errors="raise"
    )
    df_calor_control["zona_normalizada"] = df_calor_control["zona"].apply(normalize_zone_name)

    heat_models = {
        "heat_hours": heat_hours,
        "hgc": hgc
    }

    heat_detail_rows = []
    for _, row in df_calor_control.iterrows():
        heat_detail_rows.extend(estimate_row_heat(row, temperaturas, heat_models))

    df_summary_calor = summarize_results(heat_detail_rows, "horas de calor", "horas_calor")

    output_path_calor = data_dir / "horas_calor_control_por_zona_variedad_modelo.csv"
    df_summary_calor.to_csv(output_path_calor, index=False)

    print("\n=== HORAS CALOR ===")
    print(df_summary_calor.to_string(index=False))
    print(f"\nArchivo generado: {output_path_calor}")


if __name__ == "__main__":
    main()

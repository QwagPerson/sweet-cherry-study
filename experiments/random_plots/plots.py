from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HEAT_CSV = BASE_DIR / "horas_calor_control_por_zona_variedad_modelo.csv"
DEFAULT_COLD_CSV = BASE_DIR / "horas_frio_control_por_zona_variedad_modelo.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "plots_output"

GROUP_COLS = ["zona", "variedad"]
HEAT_VALUE = "horas_calor_promedio"
HEAT_STD = "horas_calor_std"
HEAT_MIN = "horas_calor_min"
HEAT_MAX = "horas_calor_max"
COLD_VALUE = "horas_frio_promedio"
COLD_STD = "horas_frio_std"
COLD_MIN = "horas_frio_min"
COLD_MAX = "horas_frio_max"


def clean_label(value: object) -> str:
    return str(value).replace("_", " ").title()


def slug(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "heat": output_dir / "heat",
        "cold": output_dir / "cold",
        "relationships": output_dir / "relationships",
        "summary": output_dir / "summary",
        "tables": output_dir / "tables",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def add_bar_labels(ax: plt.Axes, bars, values: list[float] | np.ndarray) -> None:
    if len(values) == 0:
        return
    max_abs = max(abs(float(v)) for v in values if pd.notna(v)) or 1.0
    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_abs * 0.015,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )


def load_data(heat_csv: Path, cold_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    heat = pd.read_csv(heat_csv)
    cold = pd.read_csv(cold_csv)

    for name, df, value_col, std_col, min_col, max_col in [
        ("heat", heat, HEAT_VALUE, HEAT_STD, HEAT_MIN, HEAT_MAX),
        ("cold", cold, COLD_VALUE, COLD_STD, COLD_MIN, COLD_MAX),
    ]:
        required = set(GROUP_COLS + ["modelo", value_col, std_col, min_col, max_col])
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {missing}")
        for col in [value_col, std_col, min_col, max_col, "n_observaciones"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["zona_variedad"] = df["zona"].astype(str) + " | " + df["variedad"].astype(str)

    return heat, cold


def write_summary_tables(heat: pd.DataFrame, cold: pd.DataFrame, out_dir: Path) -> None:
    heat.to_csv(out_dir / "heat_long.csv", index=False)
    cold.to_csv(out_dir / "cold_long.csv", index=False)

    heat_wide = heat.pivot_table(
        index=GROUP_COLS,
        columns="modelo",
        values=HEAT_VALUE,
        aggfunc="mean",
    ).reset_index()
    cold_wide = cold.pivot_table(
        index=GROUP_COLS,
        columns="modelo",
        values=COLD_VALUE,
        aggfunc="mean",
    ).reset_index()

    heat_wide.to_csv(out_dir / "heat_by_zone_variety_wide.csv", index=False)
    cold_wide.to_csv(out_dir / "cold_by_zone_variety_wide.csv", index=False)

    merged = heat.merge(
        cold,
        on=GROUP_COLS,
        how="outer",
        suffixes=("_heat", "_cold"),
    )
    merged.to_csv(out_dir / "all_heat_cold_model_pairs.csv", index=False)


def plot_model_bars(
    df: pd.DataFrame,
    value_col: str,
    std_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    group_col: str,
) -> None:
    pivot = df.pivot_table(index=group_col, columns="modelo", values=value_col, aggfunc="mean")
    err = df.pivot_table(index=group_col, columns="modelo", values=std_col, aggfunc="mean")
    pivot = pivot.sort_index()
    err = err.reindex(index=pivot.index, columns=pivot.columns)

    fig_width = max(10, 0.65 * len(pivot.index) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = np.arange(len(pivot.index))
    models = list(pivot.columns)
    width = min(0.8 / max(len(models), 1), 0.35)

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        values = pivot[model].to_numpy()
        yerr = err[model].to_numpy()
        bars = ax.bar(x + offset, values, width, yerr=yerr, capsize=3, label=clean_label(model))
        if len(pivot.index) <= 12:
            add_bar_labels(ax, bars, values)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([clean_label(v) for v in pivot.index], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Modelo")
    savefig(output_path)


def plot_grouped_by_zone_and_variety(
    df: pd.DataFrame,
    value_col: str,
    std_col: str,
    metric_name: str,
    ylabel: str,
    out_dir: Path,
) -> None:
    plot_model_bars(
        df,
        value_col,
        std_col,
        f"{metric_name}: promedio por zona y modelo",
        ylabel,
        out_dir / f"{slug(metric_name)}_por_zona_modelo.png",
        "zona",
    )
    plot_model_bars(
        df,
        value_col,
        std_col,
        f"{metric_name}: promedio por variedad y modelo",
        ylabel,
        out_dir / f"{slug(metric_name)}_por_variedad_modelo.png",
        "variedad",
    )
    plot_model_bars(
        df,
        value_col,
        std_col,
        f"{metric_name}: promedio por combinacion zona-variedad y modelo",
        ylabel,
        out_dir / f"{slug(metric_name)}_por_zona_variedad_modelo.png",
        "zona_variedad",
    )


def plot_single_model_bar(
    df: pd.DataFrame,
    value_col: str,
    std_col: str,
    model: str,
    group_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    subset = df[df["modelo"] == model].copy()
    grouped = (
        subset.groupby(group_col, as_index=False)
        .agg(value=(value_col, "mean"), std=(std_col, "mean"))
        .sort_values(group_col)
    )

    fig_width = max(9, 0.65 * len(grouped) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    x = np.arange(len(grouped))
    bars = ax.bar(x, grouped["value"], yerr=grouped["std"], capsize=4)
    add_bar_labels(ax, bars, grouped["value"].to_numpy())
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped[group_col].map(clean_label), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    savefig(output_path)


def plot_single_model_variability(
    df: pd.DataFrame,
    value_col: str,
    std_col: str,
    min_col: str,
    max_col: str,
    model: str,
    metric_name: str,
    out_dir: Path,
) -> None:
    subset = df[df["modelo"] == model].copy()
    subset["coef_var"] = subset[std_col] / subset[value_col]
    subset["range"] = subset[max_col] - subset[min_col]

    for metric, label in [("coef_var", "Coeficiente de variacion"), ("range", "Rango max-min")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for variety in sorted(subset["variedad"].unique()):
            part = subset[subset["variedad"] == variety]
            ax.scatter(part[value_col], part[metric], s=85, alpha=0.85, label=clean_label(variety))
            for _, row in part.iterrows():
                ax.annotate(
                    clean_label(row["zona_variedad"]),
                    (row[value_col], row[metric]),
                    fontsize=8,
                    xytext=(4, 3),
                    textcoords="offset points",
                )
        ax.set_title(f"{metric_name} - {clean_label(model)}: promedio vs {label.lower()}")
        ax.set_xlabel("Promedio")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(title="Variedad")
        savefig(out_dir / f"{slug(metric_name)}_{slug(model)}_promedio_vs_{metric}.png")


def plot_everything_segmented_by_model(
    df: pd.DataFrame,
    value_col: str,
    std_col: str,
    min_col: str,
    max_col: str,
    metric_name: str,
    ylabel: str,
    out_dir: Path,
) -> None:
    model_root = out_dir / "by_model"
    model_root.mkdir(parents=True, exist_ok=True)

    for model in sorted(df["modelo"].unique()):
        model_dir = model_root / slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)

        plot_single_model_bar(
            df,
            value_col,
            std_col,
            model,
            "zona",
            f"{metric_name} - {clean_label(model)}: promedio por zona",
            ylabel,
            model_dir / f"{slug(metric_name)}_{slug(model)}_por_zona.png",
        )
        plot_single_model_bar(
            df,
            value_col,
            std_col,
            model,
            "variedad",
            f"{metric_name} - {clean_label(model)}: promedio por variedad",
            ylabel,
            model_dir / f"{slug(metric_name)}_{slug(model)}_por_variedad.png",
        )
        plot_single_model_bar(
            df,
            value_col,
            std_col,
            model,
            "zona_variedad",
            f"{metric_name} - {clean_label(model)}: promedio por zona-variedad",
            ylabel,
            model_dir / f"{slug(metric_name)}_{slug(model)}_por_zona_variedad.png",
        )
        plot_heatmap(
            df[df["modelo"] == model],
            value_col,
            metric_name,
            model_dir,
        )
        plot_model_rankings(
            df[df["modelo"] == model],
            value_col,
            metric_name,
            model_dir,
        )
        plot_single_model_variability(
            df,
            value_col,
            std_col,
            min_col,
            max_col,
            model,
            metric_name,
            model_dir,
        )


def plot_model_facets(
    df: pd.DataFrame,
    value_col: str,
    std_col: str,
    metric_name: str,
    ylabel: str,
    out_dir: Path,
) -> None:
    models = sorted(df["modelo"].unique())
    ncols = min(2, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), squeeze=False)

    for ax, model in zip(axes.ravel(), models):
        subset = df[df["modelo"] == model].sort_values(["zona", "variedad"])
        labels = subset["zona_variedad"].map(clean_label).tolist()
        x = np.arange(len(subset))
        values = subset[value_col].to_numpy()
        yerr = subset[std_col].to_numpy()
        ax.bar(x, values, yerr=yerr, capsize=3)
        ax.set_title(clean_label(model))
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.grid(axis="y", alpha=0.25)

    for ax in axes.ravel()[len(models) :]:
        ax.axis("off")

    fig.suptitle(f"{metric_name}: rendimiento por modelo en cada zona-variedad", y=1.01)
    savefig(out_dir / f"{slug(metric_name)}_facetas_por_modelo.png")


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    metric_name: str,
    out_dir: Path,
) -> None:
    for model in sorted(df["modelo"].unique()):
        matrix = df[df["modelo"] == model].pivot_table(
            index="zona",
            columns="variedad",
            values=value_col,
            aggfunc="mean",
        )
        matrix = matrix.sort_index().sort_index(axis=1)

        fig, ax = plt.subplots(figsize=(max(7, matrix.shape[1] * 1.8), max(6, matrix.shape[0] * 0.65)))
        masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
        image = ax.imshow(masked, aspect="auto", cmap="viridis")
        ax.set_title(f"{metric_name}: {clean_label(model)} por zona y variedad")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([clean_label(v) for v in matrix.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([clean_label(v) for v in matrix.index])

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:,.0f}", ha="center", va="center", color="white", fontsize=8)

        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Promedio")
        savefig(out_dir / f"{slug(metric_name)}_heatmap_{slug(model)}.png")


def plot_model_rankings(
    df: pd.DataFrame,
    value_col: str,
    metric_name: str,
    out_dir: Path,
) -> None:
    for model in sorted(df["modelo"].unique()):
        subset = df[df["modelo"] == model].sort_values(value_col, ascending=False)
        fig, ax = plt.subplots(figsize=(10, max(5, len(subset) * 0.45)))
        y = np.arange(len(subset))
        values = subset[value_col].to_numpy()
        labels = subset["zona_variedad"].map(clean_label).tolist()
        ax.barh(y, values)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(f"{metric_name}: ranking {clean_label(model)}")
        ax.set_xlabel("Promedio")
        ax.grid(axis="x", alpha=0.25)
        for yi, value in zip(y, values):
            ax.text(value, yi, f" {value:,.0f}", va="center", fontsize=8)
        savefig(out_dir / f"{slug(metric_name)}_ranking_{slug(model)}.png")


def plot_variability(
    df: pd.DataFrame,
    value_col: str,
    std_col: str,
    min_col: str,
    max_col: str,
    metric_name: str,
    out_dir: Path,
) -> None:
    data = df.copy()
    data["coef_var"] = data[std_col] / data[value_col]
    data["range"] = data[max_col] - data[min_col]

    for metric, label in [("coef_var", "Coeficiente de variacion"), ("range", "Rango max-min")]:
        fig, ax = plt.subplots(figsize=(11, 6))
        for model in sorted(data["modelo"].unique()):
            subset = data[data["modelo"] == model]
            ax.scatter(subset[value_col], subset[metric], s=75, alpha=0.8, label=clean_label(model))
            for _, row in subset.iterrows():
                ax.annotate(
                    clean_label(row["zona_variedad"]),
                    (row[value_col], row[metric]),
                    fontsize=7,
                    alpha=0.8,
                    xytext=(4, 3),
                    textcoords="offset points",
                )
        ax.set_title(f"{metric_name}: promedio vs {label.lower()}")
        ax.set_xlabel("Promedio")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(title="Modelo")
        savefig(out_dir / f"{slug(metric_name)}_promedio_vs_{metric}.png")


def minmax(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_value) / (max_value - min_value)


def plot_normalized_profiles(
    df: pd.DataFrame,
    value_col: str,
    metric_name: str,
    out_dir: Path,
) -> None:
    data = df.copy()
    data["normalized"] = data.groupby("modelo")[value_col].transform(minmax)
    profile = data.pivot_table(
        index="zona_variedad",
        columns="modelo",
        values="normalized",
        aggfunc="mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(max(10, len(profile) * 0.7), 6))
    x = np.arange(len(profile))
    for model in profile.columns:
        ax.plot(x, profile[model], marker="o", linewidth=2, label=clean_label(model))
    ax.set_title(f"{metric_name}: perfil normalizado por combinacion")
    ax.set_ylabel("Valor normalizado dentro de cada modelo")
    ax.set_xticks(x)
    ax.set_xticklabels([clean_label(v) for v in profile.index], rotation=40, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Modelo")
    savefig(out_dir / f"{slug(metric_name)}_perfiles_normalizados.png")


def plot_heat_cold_relationships(heat: pd.DataFrame, cold: pd.DataFrame, out_dir: Path) -> None:
    merged = heat.merge(
        cold,
        on=GROUP_COLS,
        how="inner",
        suffixes=("_heat", "_cold"),
    )
    if merged.empty:
        return
    merged["zona_variedad"] = merged["zona"].astype(str) + " | " + merged["variedad"].astype(str)

    heat_models = sorted(heat["modelo"].unique())
    cold_models = sorted(cold["modelo"].unique())

    fig, axes = plt.subplots(
        len(cold_models),
        len(heat_models),
        figsize=(6 * len(heat_models), 4.8 * len(cold_models)),
        squeeze=False,
    )
    for i, cold_model in enumerate(cold_models):
        for j, heat_model in enumerate(heat_models):
            ax = axes[i][j]
            subset = merged[
                (merged["modelo_cold"] == cold_model)
                & (merged["modelo_heat"] == heat_model)
            ].copy()
            if subset.empty:
                ax.axis("off")
                continue
            for variety in sorted(subset["variedad"].unique()):
                part = subset[subset["variedad"] == variety]
                ax.scatter(
                    part[COLD_VALUE],
                    part[HEAT_VALUE],
                    s=90,
                    alpha=0.85,
                    label=clean_label(variety),
                )
                for _, row in part.iterrows():
                    ax.annotate(
                        clean_label(row["zona"]),
                        (row[COLD_VALUE], row[HEAT_VALUE]),
                        fontsize=8,
                        xytext=(4, 3),
                        textcoords="offset points",
                    )
            if len(subset) >= 2:
                corr = subset[[COLD_VALUE, HEAT_VALUE]].corr().iloc[0, 1]
                ax.text(
                    0.02,
                    0.98,
                    f"r={corr:.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
            ax.set_title(f"{clean_label(cold_model)} vs {clean_label(heat_model)}")
            ax.set_xlabel("Frio promedio")
            ax.set_ylabel("Calor promedio")
            ax.grid(alpha=0.25)
            ax.legend(title="Variedad", fontsize=8)
    fig.suptitle("Complemento frio-calor por misma zona y variedad", y=1.01)
    savefig(out_dir / "matriz_relaciones_frio_vs_calor.png")

    for cold_model in cold_models:
        for heat_model in heat_models:
            subset = merged[
                (merged["modelo_cold"] == cold_model)
                & (merged["modelo_heat"] == heat_model)
            ].copy()
            if subset.empty:
                continue
            subset["cold_norm"] = minmax(subset[COLD_VALUE])
            subset["heat_norm"] = minmax(subset[HEAT_VALUE])
            subset["balance_heat_minus_cold"] = subset["heat_norm"] - subset["cold_norm"]
            subset = subset.sort_values("balance_heat_minus_cold")

            fig, ax = plt.subplots(figsize=(10, max(5, len(subset) * 0.45)))
            y = np.arange(len(subset))
            ax.barh(y - 0.18, subset["cold_norm"], height=0.35, label="Frio normalizado")
            ax.barh(y + 0.18, subset["heat_norm"], height=0.35, label="Calor normalizado")
            ax.set_yticks(y)
            ax.set_yticklabels(subset["zona_variedad"].map(clean_label))
            ax.set_xlim(0, 1.05)
            ax.set_title(f"Balance normalizado: {clean_label(cold_model)} + {clean_label(heat_model)}")
            ax.set_xlabel("0 = menor requerimiento observado, 1 = mayor")
            ax.grid(axis="x", alpha=0.25)
            ax.legend()
            savefig(out_dir / f"balance_{slug(cold_model)}_vs_{slug(heat_model)}.png")

            fig, ax = plt.subplots(figsize=(11, 6))
            x = np.arange(len(subset))
            ax.plot(x, subset["cold_norm"], marker="o", linewidth=2, label="Frio normalizado")
            ax.plot(x, subset["heat_norm"], marker="o", linewidth=2, label="Calor normalizado")
            ax.fill_between(x, subset["cold_norm"], subset["heat_norm"], alpha=0.18)
            ax.set_xticks(x)
            ax.set_xticklabels(subset["zona_variedad"].map(clean_label), rotation=40, ha="right")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"Complementariedad por combinacion: {clean_label(cold_model)} + {clean_label(heat_model)}")
            ax.set_ylabel("Valor normalizado")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            savefig(out_dir / f"complemento_{slug(cold_model)}_vs_{slug(heat_model)}.png")


def plot_availability_summary(heat: pd.DataFrame, cold: pd.DataFrame, out_dir: Path) -> None:
    heat_combos = heat[GROUP_COLS].drop_duplicates().assign(has_heat=1)
    cold_combos = cold[GROUP_COLS].drop_duplicates().assign(has_cold=1)
    availability = heat_combos.merge(cold_combos, on=GROUP_COLS, how="outer").fillna(0)
    availability["zona_variedad"] = availability["zona"] + " | " + availability["variedad"]
    availability = availability.sort_values(["zona", "variedad"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(availability) * 0.4)))
    matrix = availability[["has_cold", "has_heat"]].to_numpy(dtype=float)
    ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_title("Disponibilidad de modelos por combinacion zona-variedad")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Frio", "Calor"])
    ax.set_yticks(np.arange(len(availability)))
    ax.set_yticklabels(availability["zona_variedad"].map(clean_label))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "si" if matrix[i, j] else "no", ha="center", va="center", fontsize=9)
    savefig(out_dir / "disponibilidad_frio_calor_por_combinacion.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matplotlib plots to study heat/cold requirement estimates by model, zone, and variety."
    )
    parser.add_argument("--heat-csv", type=Path, default=DEFAULT_HEAT_CSV)
    parser.add_argument("--cold-csv", type=Path, default=DEFAULT_COLD_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output_dir)
    heat, cold = load_data(args.heat_csv, args.cold_csv)
    write_summary_tables(heat, cold, dirs["tables"])

    plot_grouped_by_zone_and_variety(
        heat,
        HEAT_VALUE,
        HEAT_STD,
        "Horas calor",
        "Horas/calor promedio",
        dirs["heat"],
    )
    plot_grouped_by_zone_and_variety(
        cold,
        COLD_VALUE,
        COLD_STD,
        "Horas frio",
        "Horas/frio promedio",
        dirs["cold"],
    )
    plot_everything_segmented_by_model(
        heat,
        HEAT_VALUE,
        HEAT_STD,
        HEAT_MIN,
        HEAT_MAX,
        "Horas calor",
        "Horas/calor promedio",
        dirs["heat"],
    )
    plot_everything_segmented_by_model(
        cold,
        COLD_VALUE,
        COLD_STD,
        COLD_MIN,
        COLD_MAX,
        "Horas frio",
        "Horas/frio promedio",
        dirs["cold"],
    )

    plot_model_facets(heat, HEAT_VALUE, HEAT_STD, "Horas calor", "Horas/calor promedio", dirs["heat"])
    plot_model_facets(cold, COLD_VALUE, COLD_STD, "Horas frio", "Horas/frio promedio", dirs["cold"])
    plot_heatmap(heat, HEAT_VALUE, "Horas calor", dirs["heat"])
    plot_heatmap(cold, COLD_VALUE, "Horas frio", dirs["cold"])
    plot_model_rankings(heat, HEAT_VALUE, "Horas calor", dirs["heat"])
    plot_model_rankings(cold, COLD_VALUE, "Horas frio", dirs["cold"])
    plot_variability(heat, HEAT_VALUE, HEAT_STD, HEAT_MIN, HEAT_MAX, "Horas calor", dirs["summary"])
    plot_variability(cold, COLD_VALUE, COLD_STD, COLD_MIN, COLD_MAX, "Horas frio", dirs["summary"])
    plot_normalized_profiles(heat, HEAT_VALUE, "Horas calor", dirs["summary"])
    plot_normalized_profiles(cold, COLD_VALUE, "Horas frio", dirs["summary"])
    plot_heat_cold_relationships(heat, cold, dirs["relationships"])
    plot_availability_summary(heat, cold, dirs["summary"])

    png_count = len(list(args.output_dir.rglob("*.png")))
    print(f"Saved {png_count} plots under {args.output_dir}")
    print(f"Saved summary tables under {dirs['tables']}")


if __name__ == "__main__":
    main()

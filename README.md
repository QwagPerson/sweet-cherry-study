# Sweet Cherry Study

Primera iteracion del sistema de modelamiento de receso invernal en cerezos.

El codigo ignora las carpetas historicas `_old` y `_old2` y consume las salidas silver
del proyecto vecino `sweet-cherry-data`.

## Entradas esperadas

```text
../sweet-cherry-data/data/silver/2026/consolidado_fechas_2024_limpio.csv
../sweet-cherry-data/data/silver/2026/consolidado_fechas_2025_limpio.csv
../sweet-cherry-data/data/silver/2026/temperaturas_horarias_por_lugar.csv
```

## Ejecucion

Construir dataset limpio y diccionario de variables:

```powershell
python -m src.pipelines.build_dataset
```

Comparar parejas termicas con validacion agrupada:

```powershell
python -m src.pipelines.compare_pairs
```

## Salidas

```text
data/processed/modeling_dataset.csv
data/processed/variable_dictionary.csv
reports/metrics_by_pair.csv
reports/oof_predictions.csv
reports/error_by_segment.csv
reports/selected_pair.json
reports/leakage_and_limitations.md
```

La metrica principal de seleccion es `mae_fecha_brotacion_dias` en predicciones
out-of-fold agregadas.

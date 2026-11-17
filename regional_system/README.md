# Sistema regional e inferencia directa

El front local ejecuta dos regresores JSON por pareja térmica. Ya no obtiene
requerimientos mediante medianas de predicciones OOF en la ruta pública.

La búsqueda regional está implementada y puede explorarse en el simulador.
La activación automática exige cobertura externa completa y menor MAE de campo
que la referencia fija. Actualmente no supera ese criterio (5,641 vs 4,510 días
en 92 controles); por tanto el modo recomendado usa **Ridge directo con frío
≤10 °C, GDH base 4,5 °C y α=10**, entrenado con todos los registros 2024+2025.
Los errores de ajuste completo no se muestran como precisión de validación.

## Reproducir

Desde `D:/Projects/Sweet-Cherry`, con las dependencias ya utilizadas por el front:

```powershell
python sweet-cherry-study/regional_system/train.py
python sweet-cherry-study/regional_system/export.py
python sweet-cherry-study/regional_system/report.py
```

`train.py` ajusta 54 candidatos, valida la selección por grupos y diagnostica
transferencia entre temporadas. Produce `output/regional_requirement_system/`.
`export.py` verifica equivalencia de features y consultas, incorpora el criterio
de activación y escribe el artefacto local
`sweet-cherry-front/reports/regional_model/bundle.json`.

La función pública `predict_latency` admite `model_strategy='recommended'`
(predeterminado) o `'regional'` (exploración explícita). El simulador expone ese
selector. Mi Campo utiliza el modo recomendado. La operación continúa limitada
a tratamiento control; si se consultan combinaciones zona-variedad no observadas
se señala la extrapolación. Sin artefacto, no se reactiva una mediana histórica.

El JSON contiene transformadores, coeficientes y modelos calibrados para todas
las parejas disponibles, reglas por zona, referencia y decisión de activación,
métricas fuera de muestra y fuentes. No requiere importar el proyecto de estudio
ni cargar pickle en el servidor. El front necesita sus series térmicas habituales
para reconstruir features y simular frío/calor. Copiar solo el JSON no reemplaza
esa dependencia de datos. Limpiar las cachés/reiniciar el proceso local tras
actualizar un artefacto; no se ha hecho un despliegue remoto.

## Pruebas

```powershell
cd sweet-cherry-front
python -m pytest tests -q -p no:cacheprovider --basetemp=D:/Projects/Sweet-Cherry/output/regional_requirement_system/pytest_full
```

Las pruebas históricas de medianas llaman expresamente a la función privada de
reproducción; las pruebas nuevas validan la ruta pública y prohíben leer OOF.
El exportador comprueba 721 filas de features y 184 consultas de controles.
Las fuentes y los experimentos sellados no se modificaron. Los cambios de
validación revisados durante el desarrollo se documentan en PROTOCOL.md y el
informe; no se presentan como pre-registro ni resultado ciego.

Para revisar resultados: `output/regional_requirement_system/informe.html`,
`regional_rules.csv`, `nested_metrics.csv`, `external_predictions.csv`,
`season_holdout.csv`, `runtime_replay.csv` y `deployment.json`.

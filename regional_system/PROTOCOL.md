# Sistema regional de requisitos — 9 de noviembre de 2026

Nueva arquitectura; no modifica los experimentos sellados. Datos: tabla actual
de modelamiento, 409 registros 2024 y 312 de 2025, preservando nombre de huerto
en la identidad. Se conserva el maestro de zonas (incluido NIC compartido).

Objetivo de selección exclusivo: MAE de T50 de campo en controles, población
que actualmente sirve el front. Ajuste de regresores con todos los tratamientos;
inferencia operativa con ECON/SCON. Cámara calibra etapas, no puntúa selección.

Registro: seis contadores de frío (≤7.2, ≤10, 0–13 °C, Utah, Carolina Norte,
Florida), tres de calor (GDH bases 0, 4.5, 7 °C sin techo), Ridge α=1,10,100.
54 candidatos. Proxy dinámico excluido: no representa el modelo biofísico completo.
Los umbrales se recalculan para cada pareja y se predicen individualmente.
Frío desde caída hasta cámara; calor estrictamente después de cámara hasta campo.
En simulación el calor comienza después de la hora del cruce frío. Horizonte:
fin del año de inicio. No se incluyen offsets ni calor solapado.

Features compartidas entre ajuste y front: especificación original del Ridge,
variedad, condición, tratamientos, latitud/longitud, fecha y condiciones previas
a caída, medias mensuales y resúmenes térmicos históricos de años anteriores.
Se conservan las reconstrucciones térmicas del pipeline, con sus limitaciones.
El modelo de calor añade fecha del cruce frío (cámara real
solo durante ajuste; cruce predicho al evaluar y servir). No se usan medianas
OOF como requerimientos ni fechas observadas como entradas de inferencia.
Las medianas de imputación numérica del transformador son parámetros técnicos,
no umbrales agronómicos. Requisitos predichos se limitan inferiormente a 1e-6.

Validación externa: cinco pliegues lugar × variedad × temporada. Selección
interna por cinco pliegues del entrenamiento externo. Comparar selector global
y regional en los mismos pliegues. Regla propia por zona solo si dispone de
≥8 controles y ≥2 temporadas en el conjunto usado para seleccionar; en caso
contrario usa ganador global. Cualquier predicción ausente en el conjunto
de selección asigna MAE infinito; nunca se elimina la fila. Desempate por orden fijo del registro.
La selección final usa las predicciones OOF 2024+2025 de candidatos fijos;
regresores finales se ajustan a todas las filas. Métricas visibles corresponden
al procedimiento regional externo, no al ajuste completo ni a la selección final.

Diagnóstico adicional: dejar cada temporada fuera, elegir solo dentro de la
otra y evaluar el año externo. La fila 2024 es transferencia inversa; la fila
2025 es 2024→2025. No se usa este diagnóstico para reabrir la búsqueda. 2025 ya
se examinó en el proyecto: ningún resultado es una validación externa ciega.
Temperatura de simulación histórica reconstruida; conserva imputación silver.
Las features climáticas usan temperaturas históricas anteriores al año;
se documenta la imputación retrospectiva, que no vuelve causal este análisis.

La primera ejecución con features simplificadas solo medidas se conserva en
output/regional_requirement_system/initial_feature_trial: presentó cobertura
externa incompleta. La segunda preserva las features del Ridge existente para
evitar introducir a la vez un cambio de representación y de arquitectura.
Esta decisión se tomó tras revisar esa validación; no es pre-registro ciego.

Exportar transformadores y coeficientes JSON por candidato, regla por zona,
fallback global, métricas con soporte/cobertura y hashes. El front ejecuta el
mismo motor; parejas manuales requieren su propio modelo calibrado. Si falta
artefacto, historia o soporte de variedad, no inventar una mediana de respaldo.

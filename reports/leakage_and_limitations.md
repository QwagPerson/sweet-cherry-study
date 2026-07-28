# Fugas y limitaciones de generalizacion

- La validacion critica es LeaveOneLocationOut; GroupKFold por lugar-variedad-temporada no reemplaza esa prueba.
- El modelo de calor se entrena con la fecha real de termino de frio, pero en evaluacion end-to-end predice usando la fecha estimada por el modelo de frio.
- Las features climaticas historicas se calculan con anos previos a la temporada de la muestra. Si no existe historia previa, se usa historia de otros anos como respaldo.
- No se usa el nombre del lugar como feature de modelamiento; se conservan latitud, longitud y condicion agroclimatica.
- El modelo `dynamic_proxy__gdh_4_5` no es una implementacion biofisica completa de porciones de frio; es un proxy para comparar la interfaz.
- XGBoost, GAM y modelos jerarquicos quedan preparados como extensiones, pero no se ejecutan en esta version para evitar dependencias no instaladas.
- La incertidumbre operacional todavia no esta calibrada; esta iteracion solo produce error historico out-of-fold.

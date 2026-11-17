"""Hourly support of thermal targets, including retrospective reconstructions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def temperature_support_by_unit(dataset: pd.DataFrame, temperatures: pd.DataFrame) -> pd.DataFrame:
    lookup = {}
    for code, zone in temperatures.sort_values("ds").groupby("codigo_lugar"):
        kinds = zone.get("tipo_dato", pd.Series("observado", index=zone.index))
        masks = np.column_stack([kinds.eq(kind).to_numpy(dtype=int) for kind in
                                 ("observado", "interpolado", "climatologia")])
        counts = np.vstack([np.zeros(3, dtype=int), masks.cumsum(axis=0)])
        lookup[str(code)] = (zone.ds.astype("datetime64[ns]").astype("int64").to_numpy(), counts)
    records = []
    for row in dataset.itertuples(index=False):
        times, cumulative = lookup[str(row.codigo_lugar)]
        record = {"unidad_experimental_id": row.unidad_experimental_id,
                  "codigo_lugar": row.codigo_lugar, "temporada": row.temporada}
        for name, start, end in [
            ("frio", row.fecha_caida_hoja_50, row.fecha_brotacion_camara_50),
            ("calor", row.fecha_brotacion_camara_50, row.fecha_brotacion_campo_50),
        ]:
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            expected = max(int((end.floor("h") - start.ceil("h")) / pd.Timedelta(hours=1)) + 1, 0)
            left, right = np.searchsorted(times, [start.value, end.value], side="left")
            right = np.searchsorted(times, end.value, side="right")
            counts = cumulative[right] - cumulative[left]
            for kind, value in zip(("observado", "interpolado", "climatologia"), counts):
                record[f"{name}_{kind}_horas"] = int(value)
            record[f"{name}_esperado_horas"] = expected
            record[f"{name}_faltante_horas"] = max(expected - int(counts.sum()), 0)
            record[f"{name}_observado_pct"] = 100 * counts[0] / expected if expected else np.nan
        records.append(record)
    return pd.DataFrame(records)

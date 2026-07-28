from __future__ import annotations

import numpy as np
import pandas as pd


def dynamic_portions_proxy(temperature_c: pd.Series) -> pd.Series:
    """Proxy simple de porciones de frio para benchmarking inicial.

    No reemplaza una implementacion biofisica completa del modelo dinamico.
    Se usa para mantener la interfaz de comparacion termica en la primera iteracion.
    """

    t = temperature_c.astype(float)
    values = np.where((t >= 0.0) & (t <= 12.0), 1.0 / 24.0, 0.0)
    values = np.where(t > 18.0, -0.25 / 24.0, values)
    return pd.Series(values, index=temperature_c.index)

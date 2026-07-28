from __future__ import annotations

import numpy as np
import pandas as pd


def utah_units(temperature_c: pd.Series) -> pd.Series:
    t = temperature_c.astype(float)
    values = np.select(
        [
            t <= 1.4,
            (t > 1.4) & (t <= 2.4),
            (t > 2.4) & (t <= 9.1),
            (t > 9.1) & (t <= 12.4),
            (t > 12.4) & (t <= 15.9),
            (t > 15.9) & (t <= 18.0),
            t > 18.0,
        ],
        [0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0],
        default=0.0,
    )
    return pd.Series(values, index=temperature_c.index)

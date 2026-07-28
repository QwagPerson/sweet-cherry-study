from __future__ import annotations

import numpy as np
import pandas as pd


def north_carolina_units(temperature_c: pd.Series) -> pd.Series:
    t = temperature_c.astype(float)
    values = np.select(
        [
            t <= 1.6,
            (t > 1.6) & (t <= 7.2),
            (t > 7.2) & (t <= 13.0),
            (t > 13.0) & (t <= 16.5),
            (t > 16.5) & (t <= 19.0),
            (t > 19.0) & (t <= 20.7),
            (t > 20.7) & (t <= 23.3),
            t > 23.3,
        ],
        [0.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0],
        default=0.0,
    )
    return pd.Series(values, index=temperature_c.index)

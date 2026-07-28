from __future__ import annotations

import numpy as np
import pandas as pd


def growing_degree_hours(temperature_c: pd.Series, base_c: float = 4.5) -> pd.Series:
    return pd.Series(
        np.maximum(temperature_c.astype(float) - base_c, 0.0),
        index=temperature_c.index,
    )

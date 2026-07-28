from __future__ import annotations

import numpy as np
import pandas as pd


def chill_hours_7_2(temperature_c: pd.Series) -> pd.Series:
    return pd.Series(np.where(temperature_c <= 7.2, 1.0, 0.0), index=temperature_c.index)


def chill_hours_10(temperature_c: pd.Series) -> pd.Series:
    return pd.Series(np.where(temperature_c <= 10.0, 1.0, 0.0), index=temperature_c.index)

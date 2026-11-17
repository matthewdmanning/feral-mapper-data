"""Per-edge aggregation helpers for values conflated onto edges."""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def weighted_mean(edge_pos: np.ndarray, values: pd.Series, weights: np.ndarray) -> pd.Series:
    """Weighted mean of ``values`` grouped by ``edge_pos``, ignoring NaN values."""
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    ok = ~np.isnan(vals) & (weights > 0)
    frame = pd.DataFrame({"e": edge_pos[ok], "wx": vals[ok] * weights[ok], "w": weights[ok]})
    sums = frame.groupby("e")[["wx", "w"]].sum()
    return sums["wx"] / sums["w"]


def weighted_majority(edge_pos: np.ndarray, values: pd.Series, weights: np.ndarray) -> pd.Series:
    """Value with the largest total weight per ``edge_pos``."""
    vals = values.to_numpy()
    ok = pd.notna(vals) & (weights > 0)
    frame = pd.DataFrame({"e": edge_pos[ok], "x": vals[ok], "w": weights[ok]})
    totals = frame.groupby(["e", "x"], sort=False)["w"].sum().reset_index()
    best = totals.loc[totals.groupby("e")["w"].idxmax()]
    return best.set_index("e")["x"]

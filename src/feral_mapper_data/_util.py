"""Shared helpers for edge attribute transfer."""

from __future__ import annotations

import re
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name} layer is missing columns: {missing}")


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop null/empty geometries, repair invalid ones, explode multi-part features."""
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf = gdf.copy()
        gdf.loc[invalid, gdf.geometry.name] = gdf.geometry[invalid].make_valid()
    return gdf.explode(index_parts=False, ignore_index=True)


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


def slug(value: object) -> str:
    """Column-safe lowercase name for a category value."""
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_").lower()
    return text or "blank"


def to_edge_frame(edges: gpd.GeoDataFrame, by_pos: dict[str, pd.Series]) -> pd.DataFrame:
    """Build a frame indexed like ``edges`` from series indexed by edge position."""
    n = len(edges)
    out = pd.DataFrame(index=pd.RangeIndex(n))
    for col, series in by_pos.items():
        out[col] = series.reindex(out.index)
    out.index = edges.index
    return out

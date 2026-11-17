"""Shared helpers for edge attribute transfer."""

from __future__ import annotations

import re
from collections.abc import Iterable

import geopandas as gpd
import pandas as pd


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name} layer is missing columns: {missing}")


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

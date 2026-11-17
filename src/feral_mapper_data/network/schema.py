"""Canonical edge-table schema: index names and frame validation."""

from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd
import pandas as pd

EDGE_INDEX: tuple[str, str, str] = ("u", "v", "key")


def check_index(frame: pd.DataFrame) -> None:
    if list(frame.index.names) != list(EDGE_INDEX):
        raise ValueError(f"Frame must be indexed by {EDGE_INDEX}, got {frame.index.names}.")


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name} layer is missing columns: {missing}")


def to_edge_frame(edges: gpd.GeoDataFrame, by_pos: dict[str, pd.Series]) -> pd.DataFrame:
    """Build a frame indexed like ``edges`` from series indexed by edge position."""
    n = len(edges)
    out = pd.DataFrame(index=pd.RangeIndex(n))
    for col, series in by_pos.items():
        out[col] = series.reindex(out.index)
    out.index = edges.index
    return out

"""Line-to-line attribute conflation: transfer attributes from source lines onto edges.

Method: each edge gets a flat-capped corridor of ``buffer_m``. Source lines are clipped
to the corridor; each clipped piece's weight is the edge distance it spans (difference
of its endpoints' linear-referenced positions on the edge). Crossing streets span ~0 m,
so they get ~0 weight without an explicit angle filter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from feral_mapper_data._util import require_columns, to_edge_frame
from feral_mapper_data.geo.aggregate import is_numeric, weighted_majority, weighted_mean
from feral_mapper_data.geo.crs import align_crs
from feral_mapper_data.geo.geometry import clean_geometries, corridors

Aggregation = Literal["weighted_mean", "majority", "max", "min"]


def conflate_lines(
    edges: gpd.GeoDataFrame,
    source: gpd.GeoDataFrame,
    columns: Sequence[str],
    *,
    buffer_m: float = 10.0,
    min_coverage: float = 0.5,
    aggregations: Mapping[str, Aggregation] | None = None,
    prefix: str = "",
    name: str = "source",
) -> pd.DataFrame:
    """Per-edge attributes from overlapping source lines.

    Returns a frame indexed like ``edges`` with ``{prefix}{column}`` per column plus
    ``{prefix}coverage`` (matched fraction of edge length, 0-1). Attributes are NaN where
    coverage < ``min_coverage``. Default aggregation: weighted_mean (numeric), majority.
    """
    require_columns(source, columns, name)
    aggregations = dict(aggregations or {})
    unknown = set(aggregations) - set(columns)
    if unknown:
        raise KeyError(f"aggregations name columns not requested: {sorted(unknown)}")

    source = clean_geometries(align_crs(source, edges, name)[[*columns, source.geometry.name]])
    edge_geoms = edges.geometry.to_numpy()
    zones = corridors(edge_geoms, buffer_m)
    src_geoms = source.geometry.to_numpy()

    src_i, edge_i = shapely.STRtree(zones).query(src_geoms, predicate="intersects")
    clipped = shapely.intersection(src_geoms[src_i], zones[edge_i])
    weights = _spanned_length(clipped, edge_geoms[edge_i])
    keep = weights > 1e-6
    src_i, edge_i, weights = src_i[keep], edge_i[keep], weights[keep]

    lengths = shapely.length(edge_geoms)
    spanned = np.bincount(edge_i, weights=weights, minlength=len(edges))
    coverage = np.divide(spanned, lengths, out=np.zeros_like(spanned), where=lengths > 0)
    coverage = np.clip(coverage, 0.0, 1.0)

    by_pos: dict[str, pd.Series] = {}
    for col in columns:
        values = source[col].iloc[src_i].reset_index(drop=True)
        agg = aggregations.get(col) or ("weighted_mean" if is_numeric(source[col]) else "majority")
        by_pos[f"{prefix}{col}"] = _aggregate(agg, edge_i, values, weights)

    out = to_edge_frame(edges, by_pos)
    out[f"{prefix}coverage"] = coverage
    attr_cols = [f"{prefix}{c}" for c in columns]
    out.loc[coverage < min_coverage, attr_cols] = np.nan
    return out


def _aggregate(agg: str, edge_i: np.ndarray, values: pd.Series, weights: np.ndarray) -> pd.Series:
    if agg == "weighted_mean":
        return weighted_mean(edge_i, values, weights)
    if agg == "majority":
        return weighted_majority(edge_i, values, weights)
    if agg in ("max", "min"):
        frame = pd.DataFrame({"e": edge_i, "x": pd.to_numeric(values, errors="coerce")})
        return frame.groupby("e")["x"].agg(agg)
    raise ValueError(f"Unknown aggregation {agg!r}.")


def _spanned_length(clipped: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Edge distance spanned by each clipped geometry (sum over its line parts)."""
    parts, owner = shapely.get_parts(clipped, return_index=True)
    is_line = shapely.get_type_id(parts) == 1
    parts, owner = parts[is_line], owner[is_line]
    edge = edges[owner]
    a = shapely.line_locate_point(edge, shapely.get_point(parts, 0))
    b = shapely.line_locate_point(edge, shapely.get_point(parts, -1))
    return np.bincount(owner, weights=np.abs(b - a), minlength=len(clipped))

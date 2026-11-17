"""Polygon layers (land cover, parcels, census tracts, habitat zones) onto edges.

Measures overlap along the edge line (length) or, with ``buffer_m``, over an edge
flat-capped corridor (area). Overlapping polygons in one layer can push fractions above 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from feral_mapper_data._util import (
    clean_geometries,
    require_columns,
    slug,
    to_edge_frame,
    weighted_majority,
    weighted_mean,
)
from feral_mapper_data.crs import align_crs


def polygon_attributes(
    edges: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    *,
    category_column: str | None = None,
    value_columns: Sequence[str] = (),
    buffer_m: float | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    """Per-edge polygon overlap attributes.

    Columns: ``{prefix}coverage`` (overlapped fraction); with ``category_column``,
    ``{prefix}{category}_frac`` per category and ``{prefix}majority``; per value column,
    ``{prefix}{column}`` as overlap-weighted mean.
    """
    if category_column is None and not value_columns:
        raise ValueError("Provide category_column and/or value_columns.")
    if buffer_m is not None and buffer_m <= 0:
        raise ValueError("buffer_m must be positive.")
    cols = [c for c in (category_column, *value_columns) if c is not None]
    require_columns(polygons, cols, "polygon")
    polygons = clean_geometries(
        align_crs(polygons, edges, "polygon")[[*cols, polygons.geometry.name]]
    )
    poly_geoms = polygons.geometry.to_numpy()
    if not np.all(np.isin(shapely.get_type_id(poly_geoms), (3, 6))):
        raise TypeError("Polygon layer must contain only (Multi)Polygon geometries.")

    targets = edges.geometry.to_numpy()
    measure = shapely.length
    if buffer_m is not None:
        targets = shapely.buffer(targets, buffer_m, cap_style="flat")
        measure = shapely.area

    edge_i, poly_i = shapely.STRtree(poly_geoms).query(targets, predicate="intersects")
    amount = measure(shapely.intersection(targets[edge_i], poly_geoms[poly_i]))
    keep = amount > 0
    edge_i, poly_i, amount = edge_i[keep], poly_i[keep], amount[keep]
    total = measure(targets)

    by_pos: dict[str, pd.Series] = {}
    frac_cols: list[str] = []
    if category_column is not None:
        cats = polygons[category_column].iloc[poly_i].reset_index(drop=True)
        frame = pd.DataFrame({"e": edge_i, "c": cats.map(slug), "a": amount})
        sums = frame.pivot_table(index="e", columns="c", values="a", aggfunc="sum", fill_value=0.0)
        for cat in sorted(sums.columns):
            frac_cols.append(f"{prefix}{cat}_frac")
            by_pos[frac_cols[-1]] = sums[cat]
        by_pos[f"{prefix}majority"] = weighted_majority(edge_i, cats, amount)
    for col in value_columns:
        values = polygons[col].iloc[poly_i].reset_index(drop=True)
        by_pos[f"{prefix}{col}"] = weighted_mean(edge_i, values, amount)

    out = to_edge_frame(edges, by_pos)
    safe_total = np.where(total > 0, total, np.nan)
    out[frac_cols] = out[frac_cols].fillna(0.0).div(safe_total, axis=0)
    covered = np.bincount(edge_i, weights=amount, minlength=len(edges))
    out[f"{prefix}coverage"] = covered / safe_total
    return out

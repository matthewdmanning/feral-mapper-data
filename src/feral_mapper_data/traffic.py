"""Average annual daily traffic (AADT) onto edges.

Two source shapes: AADT segments (lines, conflated by spanned length) and count stations
(points, snapped to nearest edge). Segments take precedence; stations fill gaps.
Note: directional segment AADT on divided roads is averaged, not summed.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from feral_mapper_data._util import clean_geometries, require_columns, to_edge_frame
from feral_mapper_data.conflate import conflate_lines
from feral_mapper_data.crs import align_crs


def traffic_attributes(
    edges: gpd.GeoDataFrame,
    *,
    segments: gpd.GeoDataFrame | None = None,
    stations: gpd.GeoDataFrame | None = None,
    segment_column: str = "aadt",
    station_column: str = "aadt",
    buffer_m: float = 10.0,
    min_coverage: float = 0.5,
    max_station_distance_m: float = 30.0,
    prefix: str = "",
) -> pd.DataFrame:
    """Columns ``{prefix}aadt`` and ``{prefix}aadt_source`` ('segment' | 'station' | NaN)."""
    if segments is None and stations is None:
        raise ValueError("Provide segments, stations, or both.")
    aadt = pd.Series(np.nan, index=edges.index, dtype=float)
    source = pd.Series(pd.NA, index=edges.index, dtype="string")

    if segments is not None:
        seg = conflate_lines(
            edges,
            segments,
            [segment_column],
            buffer_m=buffer_m,
            min_coverage=min_coverage,
            aggregations={segment_column: "weighted_mean"},
            name="traffic segments",
        )[segment_column]
        hit = seg.notna()
        aadt[hit], source[hit] = seg[hit], "segment"

    if stations is not None:
        sta = _snap_stations(edges, stations, station_column, max_station_distance_m)
        fill = aadt.isna() & sta.notna()
        aadt[fill], source[fill] = sta[fill], "station"

    return pd.DataFrame({f"{prefix}aadt": aadt, f"{prefix}aadt_source": source})


def _snap_stations(
    edges: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, column: str, max_distance: float
) -> pd.Series:
    """Mean station count per edge for stations within ``max_distance`` of it."""
    require_columns(stations, [column], "traffic stations")
    stations = clean_geometries(align_crs(stations, edges, "traffic stations"))
    pts = stations.geometry.to_numpy()
    if not np.all(shapely.get_type_id(pts) == 0):
        raise TypeError("Traffic stations must be Point geometries.")
    tree = shapely.STRtree(edges.geometry.to_numpy())
    pt_i, edge_i = tree.query_nearest(pts, max_distance=max_distance, all_matches=False)
    values = pd.to_numeric(stations[column], errors="coerce").to_numpy()[pt_i]
    frame = pd.DataFrame({"e": edge_i, "x": values}).dropna()
    by_pos = {"x": frame.groupby("e")["x"].mean()}
    return to_edge_frame(edges, by_pos)["x"]

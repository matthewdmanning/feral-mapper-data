"""Edge table from noded road centerlines."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely

from feral_mapper_data.geo.geometry import clean_geometries, components
from feral_mapper_data.network.schema import EDGE_INDEX


def edges_from_lines(lines: gpd.GeoDataFrame, node_tolerance: float = 0.5) -> gpd.GeoDataFrame:
    """Edge table from noded road centerlines (each line ends at intersections).

    Endpoints within ``node_tolerance`` (CRS units; use a projected CRS) share a node,
    transitively (single linkage: STRtree ``dwithin`` pairs + connected components; unlike
    grid rounding, close points never split across a cell border).
    Lines are not split at interior crossings: node the data upstream if needed.
    """
    if node_tolerance <= 0:
        raise ValueError("node_tolerance must be positive.")
    lines = clean_geometries(lines)
    geoms = lines.geometry.to_numpy()
    if not np.all(shapely.get_type_id(geoms) == 1):
        raise TypeError("Network layer must contain only (Multi)LineString geometries.")
    node_ids = cluster_points(np.vstack(endpoints(geoms)), node_tolerance)
    n = len(geoms)
    out = lines.copy()
    out["u"], out["v"] = node_ids[:n], node_ids[n:]
    out["key"] = out.groupby(["u", "v"]).cumcount()
    return out.set_index(list(EDGE_INDEX))


def cluster_points(xy: np.ndarray, tolerance: float) -> np.ndarray:
    """Cluster label per point; points within ``tolerance`` (transitively) share a label."""
    uniq, inverse = np.unique(xy, axis=0, return_inverse=True)  # exact duplicates first
    pts = shapely.points(uniq)
    a, b = shapely.STRtree(pts).query(pts, predicate="dwithin", distance=tolerance)
    return components(a, b, len(uniq))[inverse.ravel()].astype(np.int64)


def endpoints(geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start, end) coordinate arrays, shape (n, 2), of line geometries."""
    return (
        shapely.get_coordinates(shapely.get_point(geoms, 0)),
        shapely.get_coordinates(shapely.get_point(geoms, -1)),
    )

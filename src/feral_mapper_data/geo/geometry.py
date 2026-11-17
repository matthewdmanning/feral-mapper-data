"""Geometry helpers: cleaning, corridors, connected components."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop null/empty geometries, repair invalid ones, explode multi-part features."""
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf = gdf.copy()
        gdf.loc[invalid, gdf.geometry.name] = gdf.geometry[invalid].make_valid()
    return gdf.explode(index_parts=False, ignore_index=True)


def buffer_flat_corridors(geoms: np.ndarray, buffer_m: float) -> np.ndarray:
    """Flat-capped corridors: stop at line ends, so values never bleed into neighbours."""
    if buffer_m <= 0:
        raise ValueError("buffer_m must be positive.")
    return shapely.buffer(geoms, buffer_m, cap_style="flat")


def label_connected_components(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Connected-component label per vertex ``0..n-1`` for undirected pairs ``(a, b)``."""
    adj = coo_array((np.ones(a.size, dtype=np.int8), (a, b)), shape=(n, n))
    return connected_components(adj, directed=False)[1]

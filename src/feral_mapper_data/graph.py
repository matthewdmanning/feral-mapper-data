"""Road graph <-> edge GeoDataFrame conversion.

Canonical edge table: GeoDataFrame indexed by ``(u, v, key)`` with LineString geometry,
matching the osmnx convention so graphs from osmnx round-trip without translation.
"""

from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString

from feral_mapper_data.geo.geometry import clean_geometries, components

EDGE_INDEX: tuple[str, str, str] = ("u", "v", "key")


def edges_from_graph(graph: nx.MultiDiGraph, crs: Any = None) -> gpd.GeoDataFrame:
    """Edge table from a MultiDiGraph (osmnx-style: nodes carry ``x``/``y``).

    Edges without a ``geometry`` attribute get a straight line between their nodes.
    """
    crs = crs if crs is not None else graph.graph.get("crs")
    if crs is None:
        raise ValueError("Graph has no 'crs' in graph.graph; pass crs explicitly.")
    rows, index = [], []
    for u, v, k, data in graph.edges(keys=True, data=True):
        attrs = dict(data)
        geom = attrs.pop("geometry", None)
        if geom is None:
            nu, nv = graph.nodes[u], graph.nodes[v]
            geom = LineString([(nu["x"], nu["y"]), (nv["x"], nv["y"])])
        attrs["geometry"] = geom
        rows.append(attrs)
        index.append((u, v, k))
    mi = pd.MultiIndex.from_tuples(index, names=list(EDGE_INDEX))
    return gpd.GeoDataFrame(rows, index=mi, geometry="geometry", crs=crs)


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


def node_coordinates(edges: gpd.GeoDataFrame) -> pd.DataFrame:
    """``x``/``y`` per node id (sorted), from the endpoints of an edge table."""
    start, end = endpoints(edges.geometry.to_numpy())
    ids = np.concatenate(
        [edges.index.get_level_values("u").to_numpy(), edges.index.get_level_values("v").to_numpy()]
    )
    return (
        pd.DataFrame(np.vstack([start, end]), columns=["x", "y"], index=ids)
        .groupby(level=0)
        .first()
    )


def graph_from_edges(edges: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    """MultiDiGraph from an edge table; node ``x``/``y`` taken from line endpoints."""
    _check_index(edges)
    graph = nx.MultiDiGraph(crs=edges.crs)
    xy = node_coordinates(edges)
    graph.add_nodes_from(zip(xy.index, _records(xy), strict=True))
    add_edge_records(graph, edges)
    return graph


def add_edge_records(graph: nx.MultiDiGraph, edges: pd.DataFrame) -> None:
    """Add every row of an edge table (indexed u, v, key) as an edge; NaN attributes skipped."""
    graph.add_edges_from(
        (u, v, k, attrs) for (u, v, k), attrs in zip(edges.index, _records(edges), strict=True)
    )


def attach_to_graph(graph: nx.MultiDiGraph, attributes: pd.DataFrame) -> nx.MultiDiGraph:
    """Copy of ``graph`` with ``attributes`` (indexed by u, v, key) set on edges.

    Missing values are not written, so absent attributes stay absent.
    """
    _check_index(attributes)
    missing = [e for e in attributes.index if not graph.has_edge(*e)]
    if missing:
        raise KeyError(f"{len(missing)} attribute rows have no matching edge, e.g. {missing[0]}")
    out = graph.copy()
    for (u, v, k), attrs in zip(attributes.index, _records(attributes), strict=True):
        out.edges[u, v, k].update(attrs)
    return out


def _check_index(frame: pd.DataFrame) -> None:
    if list(frame.index.names) != list(EDGE_INDEX):
        raise ValueError(f"Frame must be indexed by {EDGE_INDEX}, got {frame.index.names}.")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Row dicts with NaN/None dropped and numpy scalars unwrapped."""
    records = []
    for row in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            clean[str(key)] = value.item() if isinstance(value, np.generic) else value
        records.append(clean)
    return records

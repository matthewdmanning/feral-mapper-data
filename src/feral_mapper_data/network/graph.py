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
from shapely.geometry import LineString

from feral_mapper_data.network.edges import line_endpoint_coordinates
from feral_mapper_data.network.schema import EDGE_INDEX, require_edge_index


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


def node_coordinates(edges: gpd.GeoDataFrame) -> pd.DataFrame:
    """``x``/``y`` per node id (sorted), from the endpoints of an edge table."""
    start, end = line_endpoint_coordinates(edges.geometry.to_numpy())
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
    require_edge_index(edges)
    graph = nx.MultiDiGraph(crs=edges.crs)
    xy = node_coordinates(edges)
    graph.add_nodes_from(zip(xy.index, clean_row_dicts(xy), strict=True))
    add_edge_records(graph, edges)
    return graph


def add_edge_records(graph: nx.MultiDiGraph, edges: pd.DataFrame) -> None:
    """Add every row of an edge table (indexed u, v, key) as an edge; NaN attributes skipped."""
    graph.add_edges_from(
        (u, v, k, attrs)
        for (u, v, k), attrs in zip(edges.index, clean_row_dicts(edges), strict=True)
    )


def attach_to_graph(graph: nx.MultiDiGraph, attributes: pd.DataFrame) -> nx.MultiDiGraph:
    """Copy of ``graph`` with ``attributes`` (indexed by u, v, key) set on edges.

    Missing values are not written, so absent attributes stay absent.
    """
    require_edge_index(attributes)
    missing = [e for e in attributes.index if not graph.has_edge(*e)]
    if missing:
        raise KeyError(f"{len(missing)} attribute rows have no matching edge, e.g. {missing[0]}")
    out = graph.copy()
    for (u, v, k), attrs in zip(attributes.index, clean_row_dicts(attributes), strict=True):
        out.edges[u, v, k].update(attrs)
    return out


def clean_row_dicts(frame: pd.DataFrame) -> list[dict[str, Any]]:
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

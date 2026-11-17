"""Attribute-aware graph coarsening: collapse large homogeneous areas into region nodes.

Topology-only simplification (osmnx ``simplify_graph``) removes interstitial vertices but keeps
every junction. Here junctions inside a homogeneous area are also merged, so a uniform forest or
suburb with hundreds of edges becomes one region node, while heterogeneous areas keep full detail.

Algorithm (deterministic, O(E)):

1. Edge class = tuple of quantised attributes: numeric ``floor(x / tol)`` (exact if no tol),
   others exact; missing values form their own class.
2. Interior node = every incident edge has the same class (and, with ``tile_m``, same tile).
3. Connected components of interior nodes, linked by edges, are regions. Regions with fewer than
   ``min_region_nodes`` nodes are left as-is.
   Components: scipy sparse ``connected_components`` (C, no Python graph objects).
4. Edges with both ends in one region are absorbed (summarised on the region); all others are
   re-pointed to region ids. Boundary nodes are never merged, so class borders stay exact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from feral_mapper_data._util import is_numeric, require_columns, weighted_majority, weighted_mean
from feral_mapper_data.graph import EDGE_INDEX, _check_index, _records

REGION_PREFIX = "region_"


@dataclass(frozen=True)
class CoarseGraph:
    """Coarsened network.

    - ``nodes``: GeoDataFrame indexed by coarse node id. ``kind`` is ``node`` (Point at the
      original node) or ``region`` (concave hull of absorbed edges). ``x``/``y`` is a real member
      node (the one nearest the region centroid), so regions stay snapped to the network.
    - ``edges``: surviving edges re-indexed ``(u, v, key)`` on coarse ids; ``orig_u``/``orig_v``/
      ``orig_key`` keep the source edge. Geometry is unchanged, so it may end away from a
      region's ``x``/``y``.
    - ``membership``: original node id -> coarse node id (identity for unmerged nodes).
    """

    nodes: gpd.GeoDataFrame
    edges: gpd.GeoDataFrame
    membership: pd.Series

    def to_networkx(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph(crs=self.edges.crs)
        node_attrs = self.nodes.drop(columns=self.nodes.geometry.name)
        for nid, attrs in zip(node_attrs.index, _records(node_attrs), strict=True):
            graph.add_node(nid, **attrs, geometry=self.nodes.geometry.loc[nid])
        for (u, v, k), attrs in zip(self.edges.index, _records(self.edges), strict=True):
            graph.add_edge(u, v, key=k, **attrs)
        return graph


def edge_classes(
    edges: pd.DataFrame, columns: Sequence[str], tolerances: Mapping[str, float] | None = None
) -> np.ndarray:
    """Integer class per edge from quantised ``columns`` (equal class = homogeneous)."""
    require_columns(edges, columns, "Edge")
    tolerances = dict(tolerances or {})
    unknown = set(tolerances) - set(columns)
    if unknown:
        raise KeyError(f"Tolerances given for columns not in 'columns': {sorted(unknown)}")
    keys = {}
    for col in columns:
        series = edges[col]
        tol = tolerances.get(col)
        if tol is not None:
            if tol <= 0:
                raise ValueError(f"Tolerance for {col!r} must be positive.")
            if not is_numeric(series):
                raise TypeError(f"Tolerance given for non-numeric column {col!r}.")
            series = series.astype(float).floordiv(tol)
        keys[col] = (
            series.astype(object).where(series.notna(), "__missing__").astype(str).to_numpy()
        )
    if not keys:
        return np.zeros(len(edges), dtype=np.int64)
    frame = pd.DataFrame(keys, index=pd.RangeIndex(len(edges)))
    return frame.groupby(list(columns), sort=True).ngroup().to_numpy(dtype=np.int64)


def coarsen_homogeneous(
    edges: gpd.GeoDataFrame,
    columns: Sequence[str],
    *,
    tolerances: Mapping[str, float] | None = None,
    min_region_nodes: int = 3,
    tile_m: float | None = None,
    hull_ratio: float = 0.3,
) -> CoarseGraph:
    """Collapse connected homogeneous areas of ``edges`` into region nodes.

    ``tolerances`` quantise numeric columns (e.g. ``{"ndvi_mean": 0.1, "aadt": 500}``).
    ``tile_m`` caps region extent: regions never cross a square grid of this size (CRS units),
    bounding the spatial resolution lost. ``hull_ratio`` is shapely's concave-hull ratio.
    """
    _check_index(edges)
    if min_region_nodes < 2:
        raise ValueError("min_region_nodes must be >= 2.")
    if tile_m is not None and tile_m <= 0:
        raise ValueError("tile_m must be positive.")
    if edges.crs is not None and not edges.crs.is_projected:
        raise ValueError("Coarsening needs a projected CRS (lengths, tiles, hulls in metres).")

    u = edges.index.get_level_values("u").to_numpy()
    v = edges.index.get_level_values("v").to_numpy()
    geoms = edges.geometry.to_numpy()
    cls = edge_classes(edges, columns, tolerances)

    start = shapely.get_coordinates(shapely.get_point(geoms, 0))
    end = shapely.get_coordinates(shapely.get_point(geoms, -1))
    node_xy = (
        pd.DataFrame(np.vstack([start, end]), columns=["x", "y"], index=np.concatenate([u, v]))
        .groupby(level=0)
        .first()
    )
    node_class = _node_class(u, v, cls)
    if tile_m is not None:
        cells = np.floor(node_xy[["x", "y"]].to_numpy() / tile_m).astype(np.int64)
        tile = pd.Series(np.unique(cells, axis=0, return_inverse=True)[1].ravel(), node_xy.index)
    else:
        tile = pd.Series(0, index=node_xy.index)

    # Positional node ids (node_xy.index is sorted unique) -> sparse adjacency -> components.
    n_nodes = len(node_xy)
    pu, pv = node_xy.index.get_indexer(u), node_xy.index.get_indexer(v)
    interior = node_class.reindex(node_xy.index).notna().to_numpy()
    tiles = tile.to_numpy()
    link = interior[pu] & interior[pv] & (pu != pv) & (tiles[pu] == tiles[pv])
    adj = coo_array(
        (np.ones(int(link.sum()), dtype=np.int8), (pu[link], pv[link])), shape=(n_nodes, n_nodes)
    )
    _, label = connected_components(adj, directed=False)
    size = np.bincount(label)
    grouped = interior & (size[label] >= min_region_nodes)
    # Regions numbered by their first member's position: deterministic.
    region_labels, first = np.unique(label[grouped], return_index=True)
    region_labels = region_labels[np.argsort(np.flatnonzero(grouped)[first])]
    rank = np.full(len(size), -1)
    rank[region_labels] = np.arange(len(region_labels))
    region_of = np.where(grouped, rank[label], -1)

    membership = pd.Series(node_xy.index, index=node_xy.index, dtype=object)
    region_ids = [f"{REGION_PREFIX}{i}" for i in range(len(region_labels))]
    membership[grouped] = np.asarray(region_ids, dtype=object)[region_of[grouped]]
    order = np.argsort(region_of[grouped], kind="stable")
    bounds = np.cumsum(np.bincount(region_of[grouped], minlength=len(region_ids)))[:-1]
    regions: list[Any] = (
        np.split(node_xy.index[grouped].to_numpy()[order], bounds) if region_ids else []
    )

    cu, cv = membership.reindex(u).to_numpy(), membership.reindex(v).to_numpy()
    internal = pd.Series(cu).isin(region_ids).to_numpy() & (cu == cv)

    nodes = _node_table(edges, columns, node_xy, membership, regions, internal, cu, hull_ratio)
    kept = edges.loc[~internal].reset_index(drop=True)
    kept["orig_u"], kept["orig_v"] = u[~internal], v[~internal]
    kept["orig_key"] = edges.index.get_level_values("key").to_numpy()[~internal]
    kept["u"], kept["v"] = cu[~internal], cv[~internal]
    kept["key"] = kept.groupby(["u", "v"]).cumcount()
    kept = kept.set_index(list(EDGE_INDEX))
    return CoarseGraph(nodes=nodes, edges=kept, membership=membership.rename("coarse_node"))


def _node_class(u: np.ndarray, v: np.ndarray, cls: np.ndarray) -> pd.Series:
    """Class per node if all incident edges share one class, else NaN (boundary node)."""
    incident = pd.DataFrame({"n": np.concatenate([u, v]), "c": np.concatenate([cls, cls])})
    grouped = incident.groupby("n")["c"]
    first, uniq = grouped.first(), grouped.nunique()
    return first.where(uniq == 1)


def _node_table(
    edges: gpd.GeoDataFrame,
    columns: Sequence[str],
    node_xy: pd.DataFrame,
    membership: pd.Series,
    regions: list[Any],
    internal: np.ndarray,
    cu: np.ndarray,
    hull_ratio: float,
) -> gpd.GeoDataFrame:
    plain = membership.index[membership.index == membership.to_numpy()]
    plain_nodes = pd.DataFrame(
        {
            "kind": "node",
            "x": node_xy.loc[plain, "x"].to_numpy(),
            "y": node_xy.loc[plain, "y"].to_numpy(),
            "n_members": 1,
            "n_internal_edges": 0,
            "internal_length_m": 0.0,
            "area_m2": 0.0,
        },
        index=pd.Index(plain, dtype=object),
    )
    plain_geom = list(shapely.points(plain_nodes[["x", "y"]].to_numpy()))
    if not regions:
        return gpd.GeoDataFrame(plain_nodes, geometry=plain_geom, crs=edges.crs).rename_axis("node")

    ids = np.array([f"{REGION_PREFIX}{i}" for i in range(len(regions))], dtype=object)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    e_region = pos.reindex(cu[internal]).to_numpy()
    lengths = edges.geometry.length.to_numpy()[internal]
    geoms = edges.geometry.to_numpy()[internal]

    hulls, anchors = [], []
    for i, members in enumerate(regions):
        hull = shapely.concave_hull(
            shapely.multilinestrings(list(geoms[e_region == i])), hull_ratio
        )
        hulls.append(hull)
        xy = node_xy.loc[members, ["x", "y"]].to_numpy()
        anchors.append(xy[np.argmin(((xy - xy.mean(axis=0)) ** 2).sum(axis=1))])
    anchor = np.asarray(anchors)
    table = pd.DataFrame(
        {
            "kind": "region",
            "x": anchor[:, 0],
            "y": anchor[:, 1],
            "n_members": [len(m) for m in regions],
            "n_internal_edges": np.bincount(e_region, minlength=len(ids)),
            "internal_length_m": np.bincount(e_region, weights=lengths, minlength=len(ids)),
            "area_m2": shapely.area(np.asarray(hulls)),
        },
        index=pd.Index(ids, dtype=object),
    )
    sub = edges.iloc[np.flatnonzero(internal)]
    for col in columns:
        agg = weighted_mean if is_numeric(sub[col]) else weighted_majority
        table[col] = agg(e_region, sub[col], lengths).rename(index=dict(enumerate(ids)))
    out = pd.concat([plain_nodes, table])
    return gpd.GeoDataFrame(out, geometry=plain_geom + hulls, crs=edges.crs).rename_axis("node")

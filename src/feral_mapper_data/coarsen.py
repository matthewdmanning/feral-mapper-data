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
   Components: scipy sparse ``connected_components``; all steps positional numpy.
4. Edges with both ends in one region are absorbed (summarised on the region); all others are
   re-pointed to region ids. Boundary nodes are never merged, so class borders stay exact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely

from feral_mapper_data.geo.aggregate import is_numeric, weighted_majority, weighted_mean
from feral_mapper_data.geo.geometry import components
from feral_mapper_data.network.graph import add_edge_records, node_coordinates, records
from feral_mapper_data.network.schema import EDGE_INDEX, check_index, require_columns

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
        attrs = records(self.nodes.drop(columns=self.nodes.geometry.name))
        for record, geom in zip(attrs, self.nodes.geometry.to_numpy(), strict=True):
            record["geometry"] = geom
        graph.add_nodes_from(zip(self.nodes.index, attrs, strict=True))
        add_edge_records(graph, self.edges)
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
    codes = []
    for col in columns:
        series = edges[col]
        tol = tolerances.get(col)
        if tol is not None:
            if tol <= 0:
                raise ValueError(f"Tolerance for {col!r} must be positive.")
            if not is_numeric(series):
                raise TypeError(f"Tolerance given for non-numeric column {col!r}.")
            series = series.astype(float).floordiv(tol)
        codes.append(pd.factorize(series, use_na_sentinel=False)[0])  # missing = own code
    if not codes:
        return np.zeros(len(edges), dtype=np.int64)
    return np.unique(np.column_stack(codes), axis=0, return_inverse=True)[1].ravel()


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
    check_index(edges)
    if min_region_nodes < 2:
        raise ValueError("min_region_nodes must be >= 2.")
    if tile_m is not None and tile_m <= 0:
        raise ValueError("tile_m must be positive.")
    if edges.crs is not None and not edges.crs.is_projected:
        raise ValueError("Coarsening needs a projected CRS (lengths, tiles, hulls in metres).")

    # Positional node ids: node_xy.index is the sorted unique node set.
    node_xy = node_coordinates(edges)
    xy = node_xy[["x", "y"]].to_numpy()
    n_nodes = len(node_xy)
    pu = node_xy.index.get_indexer(edges.index.get_level_values("u"))
    pv = node_xy.index.get_indexer(edges.index.get_level_values("v"))

    # Interior node: min == max class over incident edges.
    cls = edge_classes(edges, columns, tolerances)
    ends, both_cls = np.concatenate([pu, pv]), np.concatenate([cls, cls])
    lo = np.full(n_nodes, np.iinfo(np.int64).max)
    hi = np.full(n_nodes, -1)
    np.minimum.at(lo, ends, both_cls)
    np.maximum.at(hi, ends, both_cls)
    interior = lo == hi

    link = interior[pu] & interior[pv] & (pu != pv)
    if tile_m is not None:
        cell = np.floor(xy / tile_m).astype(np.int64)
        link &= (cell[pu] == cell[pv]).all(axis=1)
    label = components(pu[link], pv[link], n_nodes)
    grouped = interior & (np.bincount(label)[label] >= min_region_nodes)

    # Region number per node (-1 = unmerged), ordered by first member: deterministic.
    uniq, first = np.unique(label[grouped], return_index=True)
    rank = np.full(label.max() + 1, -1)
    rank[uniq[np.argsort(first)]] = np.arange(uniq.size)
    region = np.where(grouped, rank[label], -1)
    region_ids = np.array([f"{REGION_PREFIX}{i}" for i in range(uniq.size)], dtype=object)

    coarse_id = node_xy.index.to_numpy(dtype=object).copy()
    coarse_id[grouped] = region_ids[region[grouped]]
    ru = region[pu]
    internal = (ru >= 0) & (ru == region[pv])

    nodes = _node_table(edges, columns, node_xy, region, region_ids, ru, internal, hull_ratio)
    kept = edges.loc[~internal].reset_index()
    kept[["orig_u", "orig_v", "orig_key"]] = kept[list(EDGE_INDEX)]
    kept["u"], kept["v"] = coarse_id[pu[~internal]], coarse_id[pv[~internal]]
    kept["key"] = kept.groupby(["u", "v"]).cumcount()
    kept = kept.set_index(list(EDGE_INDEX))
    membership = pd.Series(coarse_id, index=node_xy.index, name="coarse_node")
    return CoarseGraph(nodes=nodes, edges=kept, membership=membership)


def _node_table(
    edges: gpd.GeoDataFrame,
    columns: Sequence[str],
    node_xy: pd.DataFrame,
    region: np.ndarray,
    region_ids: np.ndarray,
    ru: np.ndarray,
    internal: np.ndarray,
    hull_ratio: float,
) -> gpd.GeoDataFrame:
    xy = node_xy[["x", "y"]].to_numpy()
    plain = region < 0
    n_reg = region_ids.size
    e_region = ru[internal]
    lengths = edges.geometry.length.to_numpy()[internal]

    # Hulls: one multilinestring per region via shapely's indices= (no per-region loop).
    by_reg = np.argsort(e_region, kind="stable")  # shapely requires sorted indices
    multi = shapely.multilinestrings(
        edges.geometry.to_numpy()[internal][by_reg], indices=e_region[by_reg]
    )
    hulls = shapely.concave_hull(multi, hull_ratio) if n_reg else np.empty(0, dtype=object)

    # Anchor: member node nearest its region centroid (lexsort, first per region).
    members = np.flatnonzero(~plain)
    m_reg = region[members]
    count = np.bincount(m_reg, minlength=n_reg)
    centroid = (
        np.column_stack(
            [np.bincount(m_reg, weights=xy[members, d], minlength=n_reg) for d in (0, 1)]
        )
        / np.maximum(count, 1)[:, None]
    )
    dist = ((xy[members] - centroid[m_reg]) ** 2).sum(axis=1)
    order = np.lexsort((dist, m_reg))
    _, first = np.unique(m_reg[order], return_index=True)
    anchor = xy[members[order[first]]] if n_reg else np.empty((0, 2))

    table = pd.DataFrame(
        {
            "kind": np.r_[np.full(plain.sum(), "node"), np.full(n_reg, "region")],
            "x": np.r_[xy[plain, 0], anchor[:, 0]],
            "y": np.r_[xy[plain, 1], anchor[:, 1]],
            "n_members": np.r_[np.ones(plain.sum(), dtype=np.int64), count],
            "n_internal_edges": np.r_[
                np.zeros(plain.sum(), dtype=np.int64), np.bincount(e_region, minlength=n_reg)
            ],
            "internal_length_m": np.r_[
                np.zeros(plain.sum()), np.bincount(e_region, weights=lengths, minlength=n_reg)
            ],
            "area_m2": np.r_[np.zeros(plain.sum()), shapely.area(hulls)],
        },
        index=pd.Index(
            np.r_[node_xy.index.to_numpy(dtype=object)[plain], region_ids], dtype=object
        ),
    )
    sub = edges.iloc[np.flatnonzero(internal)]
    for col in columns:
        agg = weighted_mean if is_numeric(sub[col]) else weighted_majority
        by_region = agg(e_region, sub[col], lengths).reindex(range(n_reg))
        table[col] = pd.Series(by_region.to_numpy(), index=region_ids)
    geometry = np.r_[shapely.points(xy[plain]), hulls]
    return gpd.GeoDataFrame(table, geometry=geometry, crs=edges.crs).rename_axis("node")

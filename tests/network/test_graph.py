import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from feral_mapper_data import (
    EDGE_INDEX,
    attach_to_graph,
    edges_from_graph,
    edges_from_lines,
    graph_from_edges,
)
from tests.conftest import CRS, by_name, line


def test_edges_from_lines_shares_nodes(edges):
    assert list(edges.index.names) == list(EDGE_INDEX)
    w, e, c = by_name(edges, "main_w"), by_name(edges, "main_e"), by_name(edges, "cross")
    assert w[1] == e[0] == c[1]  # all meet at (100, 0)
    assert len(set(edges.index.get_level_values("u")) | set(edges.index.get_level_values("v"))) == 4


def test_edges_from_lines_tolerance_snaps_near_endpoints():
    lines = gpd.GeoDataFrame(geometry=[line((0, 0), (10, 0)), line((10.1, 0.1), (20, 0))], crs=CRS)
    out = edges_from_lines(lines, node_tolerance=0.5)
    assert out.index[0][1] == out.index[1][0]


def test_parallel_edges_get_distinct_keys():
    lines = gpd.GeoDataFrame(
        geometry=[line((0, 0), (10, 0)), line((0, 0), (5, 3), (10, 0))], crs=CRS
    )
    keys = sorted(edges_from_lines(lines).index.get_level_values("key"))
    assert keys == [0, 1]


def test_rejects_non_lines():
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0], [0]), crs=CRS)
    with pytest.raises(TypeError):
        edges_from_lines(pts)


def test_graph_roundtrip_and_attach(edges):
    graph = graph_from_edges(edges)
    assert graph.number_of_edges() == 3
    back = edges_from_graph(graph)
    assert set(back.index) == set(edges.index)

    attrs = pd.DataFrame({"aadt": [100.0, float("nan"), 5.0]}, index=edges.index)
    out = attach_to_graph(graph, attrs)
    w = by_name(edges, "main_w")
    assert out.edges[w]["aadt"] == 100.0
    assert "aadt" not in out.edges[by_name(edges, "main_e")]  # NaN not written
    assert "aadt" not in graph.edges[w]  # original untouched


def test_attach_rejects_unknown_edges(edges):
    attrs = pd.DataFrame(
        {"x": [1]}, index=pd.MultiIndex.from_tuples([(98, 99, 0)], names=EDGE_INDEX)
    )
    with pytest.raises(KeyError):
        attach_to_graph(graph_from_edges(edges), attrs)


def test_edges_from_graph_builds_missing_geometry():
    g = nx.MultiDiGraph(crs=CRS)
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=3.0, y=4.0)
    g.add_edge(1, 2, speed=30)
    out = edges_from_graph(g)
    assert out.geometry.iloc[0].length == pytest.approx(5.0)
    assert out["speed"].iloc[0] == 30


def test_edges_from_graph_mixed_geometry():
    g = nx.MultiDiGraph(crs=CRS)
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=3.0, y=4.0)
    g.add_edge(1, 2, 0, geometry=line((0, 0), (1, 0), (3, 4)))
    g.add_edge(2, 1, 0, speed=30)
    out = edges_from_graph(g)
    assert list(out.index.names) == ["u", "v", "key"]
    assert out.geometry.iloc[0].length == pytest.approx(1.0 + np.hypot(2, 4))
    assert list(out.geometry.iloc[1].coords) == [(3.0, 4.0), (0.0, 0.0)]
    assert out["speed"].isna().tolist() == [True, False]


def test_edges_from_graph_empty():
    out = edges_from_graph(nx.MultiDiGraph(crs=CRS))
    assert out.empty
    assert list(out.index.names) == ["u", "v", "key"]
    assert out.crs == CRS


def test_snapping_has_no_grid_border_artifacts():
    # 0.24 and 0.26 round to different 0.5 m grid cells but are 0.02 m apart
    lines = gpd.GeoDataFrame(
        geometry=[line((0, 0), (10.24, 0)), line((10.26, 0), (20, 0))], crs=CRS
    )
    out = edges_from_lines(lines, node_tolerance=0.5)
    assert out.index[0][1] == out.index[1][0]


def test_cluster_points_is_transitive():
    from feral_mapper_data.network.edges import cluster_points

    labels = cluster_points(np.array([[0.0, 0], [0.4, 0], [0.8, 0], [5.0, 0]]), 0.5)
    assert labels[0] == labels[1] == labels[2] != labels[3]

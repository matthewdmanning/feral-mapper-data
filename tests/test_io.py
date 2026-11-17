"""Vector and GraphML I/O choose the format from the file extension and round-trip losslessly."""

from __future__ import annotations

import networkx as nx
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import LineString

from feral_mapper_data.io import read_graphml, read_vector, write_graphml, write_vector

# ---------------------------------------------------------------------------
# Helpers / local fixtures
# ---------------------------------------------------------------------------


def _make_graph() -> nx.MultiDiGraph:
    """Two-node graph with one LineString edge and a string CRS."""
    graph = nx.MultiDiGraph(crs="EPSG:32617")
    graph.add_node(0, x=0.0, y=0.0)
    graph.add_node(1, x=10.0, y=0.0)
    graph.add_edge(0, 1, geometry=LineString([(0, 0), (10, 0)]), length=10.0, name="a")
    return graph


# ---------------------------------------------------------------------------
# Vector round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".parquet", ".geoparquet", ".gpkg"])
def test_vector_round_trip_preserves_data_and_crs(tmp_path, network_lines, suffix):
    path = write_vector(network_lines, tmp_path / f"net{suffix}")
    assert_geodataframe_equal(read_vector(path), network_lines)


def test_write_vector_creates_missing_parent_directories(tmp_path, network_lines):
    path = write_vector(network_lines, tmp_path / "a" / "b" / "net.parquet")
    assert path.exists()


def test_read_vector_selects_layer(tmp_path, network_lines):
    path = tmp_path / "multi.gpkg"
    network_lines.iloc[:1].to_file(path, layer="one")
    network_lines.to_file(path, layer="all")
    assert len(read_vector(path, layer="one")) == 1
    assert len(read_vector(path, layer="all")) == len(network_lines)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_read_vector_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent"):
        read_vector(tmp_path / "absent.gpkg")


# ---------------------------------------------------------------------------
# GraphML
# ---------------------------------------------------------------------------


def test_write_graphml_serialises_geometry_as_wkt(tmp_path):
    path = write_graphml(_make_graph(), tmp_path / "out" / "g.graphml")
    (edge,) = (d for *_, d in nx.read_graphml(path).edges(data=True))
    assert edge["geometry"] == "LINESTRING (0 0, 10 0)"


def test_read_graphml_restores_osmnx_geometry(tmp_path):
    ox = pytest.importorskip("osmnx")
    path = tmp_path / "g.graphml"
    ox.save_graphml(_make_graph(), path)
    (edge,) = (d for *_, d in read_graphml(path).edges(data=True))
    assert isinstance(edge["geometry"], LineString)
    assert edge["length"] == pytest.approx(10.0)

"""run_pipeline() writes every configured output and is deterministic."""

from __future__ import annotations

import textwrap

import geopandas as gpd
import networkx as nx
import pytest
from geopandas.testing import assert_geodataframe_equal

from feral_mapper_data import graph_from_edges
from feral_mapper_data.app.config import PipelineConfig
from feral_mapper_data.app.pipeline import run_pipeline
from tests.app.conftest import PIPELINE_TOML

# ---------------------------------------------------------------------------
# Helpers / local fixtures
# ---------------------------------------------------------------------------

COARSEN_TOML = textwrap.dedent(
    """
    coarse_nodes = "out/coarse_nodes.parquet"
    coarse_edges = "out/coarse_edges.parquet"
    coarse_graphml = "out/coarse.graphml"

    [coarsen]
    columns = ["lc_majority"]
    min_region_nodes = 2
    """
)


def _run_with(workspace, toml: str) -> gpd.GeoDataFrame:
    """Write ``toml`` as the workspace config and run the pipeline on it."""
    (workspace / "config.toml").write_text(toml)
    return run_pipeline(PipelineConfig.from_toml(workspace / "config.toml"))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_run_twice_gives_identical_edges(workspace):
    first = _run_with(workspace, PIPELINE_TOML)
    second = _run_with(workspace, PIPELINE_TOML)
    assert_geodataframe_equal(first, second)


# ---------------------------------------------------------------------------
# Coarsened outputs
# ---------------------------------------------------------------------------


def test_coarsen_writes_nodes_edges_and_graphml(workspace):
    _run_with(workspace, PIPELINE_TOML + COARSEN_TOML)
    nodes = gpd.read_parquet(workspace / "out/coarse_nodes.parquet")
    assert {"kind", "n_members", "members"} <= set(nodes.columns)
    assert list(nodes["kind"]) == ["region"]  # all edges are park: one region absorbs them
    assert len(gpd.read_parquet(workspace / "out/coarse_edges.parquet")) == 0
    assert nx.read_graphml(workspace / "out/coarse.graphml").number_of_nodes() == len(nodes)


def test_coarsen_and_geosparql(workspace):
    pytest.importorskip("rdflib")
    toml = PIPELINE_TOML + 'geosparql = "out/network.ttl"\n' + COARSEN_TOML
    _run_with(workspace, toml)
    assert (workspace / "out/network.ttl").stat().st_size > 0


# ---------------------------------------------------------------------------
# GraphML input
# ---------------------------------------------------------------------------


def test_graphml_input_keeps_attributes_on_output_graph(workspace, edges):
    ox = pytest.importorskip("osmnx")
    ox.save_graphml(graph_from_edges(edges), workspace / "network.graphml")
    _run_with(workspace, PIPELINE_TOML.replace("network.gpkg", "network.graphml"))
    graph = nx.read_graphml(workspace / "out/graph.graphml")
    assert graph.number_of_edges() == len(edges)
    assert any("ndvi_mean" in d for *_, d in graph.edges(data=True))

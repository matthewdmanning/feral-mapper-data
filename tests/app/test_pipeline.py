"""run_pipeline() writes every configured output and is deterministic."""

from __future__ import annotations

import textwrap
from itertools import pairwise

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import box

from feral_mapper_data import graph_from_edges
from feral_mapper_data.app.config import PipelineConfig
from feral_mapper_data.app.pipeline import run_pipeline
from tests.app.conftest import PIPELINE_TOML
from tests.conftest import CRS, X0, Y0, line

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


def test_coarse_members_mix_nodes_and_regions(tmp_path):
    """Path A-B-C-D-E: park west of C, urban east. Two regions plus boundary node C survive."""
    xs = [0, 100, 200, 300, 400]
    gpd.GeoDataFrame(
        {"name": ["ab", "bc", "cd", "de"]},
        geometry=[line((a, 0), (b, 0)) for a, b in pairwise(xs)],
        crs=CRS,
    ).to_file(tmp_path / "network.gpkg")
    gpd.GeoDataFrame(
        {"class": ["park", "urban"]},
        geometry=[
            box(X0 - 10, Y0 - 50, X0 + 200, Y0 + 50),
            box(X0 + 200, Y0 - 50, X0 + 410, Y0 + 50),
        ],
        crs=CRS,
    ).to_file(tmp_path / "landcover.gpkg")
    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            """
            [network]
            path = "network.gpkg"

            [[polygons]]
            path = "landcover.gpkg"
            category_column = "class"
            prefix = "lc_"

            [coarsen]
            columns = ["lc_majority"]
            min_region_nodes = 2

            [output]
            edges = "out/edges.parquet"
            coarse_nodes = "out/coarse_nodes.parquet"
            coarse_edges = "out/coarse_edges.parquet"
            """
        )
    )
    run_pipeline(PipelineConfig.from_toml(tmp_path / "config.toml"))
    nodes = gpd.read_parquet(tmp_path / "out/coarse_nodes.parquet")
    assert nodes.index.dtype == np.int64
    members = {kind: sorted(map(sorted, grp["members"])) for kind, grp in nodes.groupby("kind")}
    # edges_from_lines numbers nodes west to east: A=0 .. E=4; C=2 is the class border
    assert members == {"node": [[2]], "region": [[0, 1], [3, 4]]}
    assert set(nodes.loc[nodes["kind"] == "region", "lc_majority"]) == {"park", "urban"}
    coarse_edges = gpd.read_parquet(tmp_path / "out/coarse_edges.parquet")
    assert len(coarse_edges) == 2  # bc and cd, re-pointed from regions to C
    assert coarse_edges["u"].dtype == np.int64
    assert set(coarse_edges["u"]) | set(coarse_edges["v"]) <= set(nodes.index)

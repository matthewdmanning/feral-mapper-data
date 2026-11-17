import textwrap

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import Point, box

from feral_mapper_data.cli import main
from feral_mapper_data.config import ConfigError, PipelineConfig
from tests.conftest import CRS, X0, Y0, line
from tests.test_vegetation import make_raster

CONFIG = """
[network]
path = "network.gpkg"

[roadway]
path = "roads.parquet"
columns = ["lanes"]

[traffic]
segments = "aadt.gpkg"
stations = "stations.gpkg"

[[vegetation]]
path = "ndvi.tif"
buffer_m = 15
stats = ["mean"]

[[polygons]]
path = "landcover.gpkg"
category_column = "class"
prefix = "lc_"

[output]
edges = "out/edges.parquet"
graphml = "out/graph.graphml"
"""


@pytest.fixture
def workspace(tmp_path, network_lines):
    network_lines.to_crs("EPSG:4326").to_file(tmp_path / "network.gpkg")  # geographic input
    gpd.GeoDataFrame({"lanes": [2]}, geometry=[line((0, 2), (100, 2))], crs=CRS).to_parquet(
        tmp_path / "roads.parquet"
    )
    gpd.GeoDataFrame({"aadt": [5000]}, geometry=[line((0, 1), (100, 1))], crs=CRS).to_file(
        tmp_path / "aadt.gpkg"
    )
    gpd.GeoDataFrame({"aadt": [700]}, geometry=[Point(X0 + 150, Y0 + 3)], crs=CRS).to_file(
        tmp_path / "stations.gpkg"
    )
    make_raster(tmp_path / "ndvi.tif", np.full((20, 30), 0.4, dtype="float32"))
    gpd.GeoDataFrame(
        {"class": ["park"]}, geometry=[box(X0 - 10, Y0 - 50, X0 + 250, Y0 + 50)], crs=CRS
    ).to_file(tmp_path / "landcover.gpkg")
    (tmp_path / "config.toml").write_text(CONFIG)
    return tmp_path


def test_cli_end_to_end(workspace):
    assert main([str(workspace / "config.toml")]) == 0
    edges = gpd.read_parquet(workspace / "out" / "edges.parquet")
    assert edges.crs.is_projected
    assert len(edges) == 3
    main_w = edges[edges["name"] == "main_w"].iloc[0]
    main_e = edges[edges["name"] == "main_e"].iloc[0]
    assert main_w["road_lanes"] == pytest.approx(2)
    assert main_w["aadt"] == pytest.approx(5000)
    assert main_e["aadt_source"] == "station"
    assert main_w["ndvi_mean"] == pytest.approx(0.4, abs=1e-6)
    assert main_w["lc_park_frac"] == pytest.approx(1.0, abs=1e-6)

    graph = nx.read_graphml(workspace / "out" / "graph.graphml")
    assert graph.number_of_edges() == 3
    assert any("ndvi_mean" in d for *_, d in graph.edges(data=True))


def test_prefix_clash_is_an_error(workspace):
    cfg = CONFIG.replace('prefix = "lc_"', 'prefix = "ndvi_"').replace(
        'category_column = "class"', 'value_columns = ["mean"]'
    )
    gpd.GeoDataFrame({"mean": [1.0]}, geometry=[box(X0, Y0 - 5, X0 + 5, Y0 + 5)], crs=CRS).to_file(
        workspace / "landcover.gpkg"
    )
    (workspace / "config.toml").write_text(cfg)
    with pytest.raises(ValueError, match="clash"):
        main([str(workspace / "config.toml")])


@pytest.mark.parametrize(
    ("toml", "match"),
    [
        ('[network]\npath="a"\n', "output"),
        ('[network]\npath="a"\nbogus=1\n[output]\nedges="e"\n', "Unknown keys"),
        ('[network]\npath="a"\n[output]\nedges="e"\n[traffic]\nbuffer_m=5\n', "segments"),
        ('[network]\npath="a"\n[output]\nedges="e"\n[polygons]\npath="p"\nprefix="x"\n', "array"),
    ],
)
def test_config_validation(tmp_path, toml, match):
    path = tmp_path / "c.toml"
    path.write_text(textwrap.dedent(toml))
    with pytest.raises(ConfigError, match=match):
        PipelineConfig.from_toml(path)


def test_config_paths_resolve_relative_to_file(tmp_path):
    path = tmp_path / "sub" / "c.toml"
    path.parent.mkdir()
    path.write_text('[network]\npath="net.gpkg"\n[output]\nedges="o.parquet"\n')
    cfg = PipelineConfig.from_toml(path)
    assert cfg.network.path == tmp_path / "sub" / "net.gpkg"


def test_cli_config_error_exit_code(tmp_path, capsys):
    path = tmp_path / "c.toml"
    path.write_text("[network]\n")
    assert main([str(path)]) == 2
    assert "config error" in capsys.readouterr().err

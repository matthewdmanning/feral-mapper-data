"""The CLI runs the pipeline and maps config problems to exit code 2."""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import box

from feral_mapper_data.app.cli import main
from tests.app.conftest import PIPELINE_TOML
from tests.conftest import CRS, X0, Y0

# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


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
    cfg = PIPELINE_TOML.replace('prefix = "lc_"', 'prefix = "ndvi_"').replace(
        'category_column = "class"', 'value_columns = ["mean"]'
    )
    gpd.GeoDataFrame({"mean": [1.0]}, geometry=[box(X0, Y0 - 5, X0 + 5, Y0 + 5)], crs=CRS).to_file(
        workspace / "landcover.gpkg"
    )
    (workspace / "config.toml").write_text(cfg)
    with pytest.raises(ValueError, match="clash"):
        main([str(workspace / "config.toml")])


# ---------------------------------------------------------------------------
# Error exit codes
# ---------------------------------------------------------------------------


def test_cli_config_error_exit_code(tmp_path, capsys):
    path = tmp_path / "c.toml"
    path.write_text("[network]\n")
    assert main([str(path)]) == 2
    assert "config error" in capsys.readouterr().err


def test_cli_missing_config_file_exit_code(tmp_path, capsys):
    assert main([str(tmp_path / "absent.toml")]) == 2
    assert "config error" in capsys.readouterr().err

"""Fixtures for end-to-end runs: every input layer plus a pipeline config."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from tests.conftest import CRS, X0, Y0, line, make_raster

PIPELINE_TOML = """
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
    """Directory with every input layer plus ``config.toml`` (see ``PIPELINE_TOML``)."""
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
    (tmp_path / "config.toml").write_text(PIPELINE_TOML)
    return tmp_path

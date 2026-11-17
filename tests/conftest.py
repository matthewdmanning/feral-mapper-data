"""Synthetic fixtures in UTM 17N (metres). Layout: a 100 m east-west edge plus a cross street."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

CRS = "EPSG:32617"
X0, Y0 = 500_000.0, 3_800_000.0
NODATA = -9999


def line(*pts: tuple[float, float]) -> LineString:
    return LineString([(X0 + x, Y0 + y) for x, y in pts])


@pytest.fixture
def network_lines() -> gpd.GeoDataFrame:
    """Main street split at x=100 plus a north-south cross street at x=100."""
    return gpd.GeoDataFrame(
        {"name": ["main_w", "main_e", "cross"]},
        geometry=[line((0, 0), (100, 0)), line((100, 0), (200, 0)), line((100, -100), (100, 0))],
        crs=CRS,
    )


@pytest.fixture
def edges(network_lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    from feral_mapper_data import edges_from_lines

    return edges_from_lines(network_lines)


def make_raster(path, data, crs=CRS, nodata=NODATA, res=10.0):
    """Raster covering x in [-50, 250), y in (-150, 50] relative to the fixture origin."""
    transform = from_origin(X0 - 50, Y0 + 50, res, res)
    count = 1 if data.ndim == 2 else data.shape[0]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[-1],
        height=data.shape[-2],
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data if data.ndim == 3 else data[None])
    return path


def by_name(edges: gpd.GeoDataFrame, name: str):
    return edges.index[edges["name"] == name][0]


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

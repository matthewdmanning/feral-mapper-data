"""clean_geometries drops empty features and repairs invalid ones."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Polygon

from feral_mapper_data.geo.geometry import clean_geometries
from tests.conftest import CRS, line


def test_clean_geometries_repairs_and_drops():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    gdf = gpd.GeoDataFrame(geometry=[bowtie, None, line((0, 0), (1, 0))], crs=CRS)
    out = clean_geometries(gdf)
    assert out.geometry.is_valid.all()
    assert out.geometry.notna().all()

"""Synthetic fixtures in UTM 17N (metres). Layout: a 100 m east-west edge plus a cross street."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString

CRS = "EPSG:32617"
X0, Y0 = 500_000.0, 3_800_000.0


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


def by_name(edges: gpd.GeoDataFrame, name: str):
    return edges.index[edges["name"] == name][0]

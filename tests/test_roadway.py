import geopandas as gpd
import numpy as np
import pytest

from feral_mapper_data import roadway_attributes
from feral_mapper_data.conflate import conflate_lines
from tests.conftest import CRS, by_name, line


@pytest.fixture
def roads() -> gpd.GeoDataFrame:
    """Inventory offset 3 m from the main street; cross street has very different values."""
    return gpd.GeoDataFrame(
        {
            "lanes": [2, 4, 6],
            "func_class": ["local", "arterial", "highway"],
        },
        geometry=[
            line((0, 3), (100, 3)),  # main_w
            line((100, 3), (150, 3)),  # first half of main_e only
            line((100, -100), (100, 0)),  # cross street
        ],
        crs=CRS,
    )


def test_parallel_match_and_crossing_ignored(edges, roads):
    out = roadway_attributes(edges, roads, ["lanes", "func_class"])
    w = by_name(edges, "main_w")
    assert out.loc[w, "road_lanes"] == pytest.approx(2.0)
    assert out.loc[w, "road_func_class"] == "local"
    assert out.loc[w, "road_coverage"] == pytest.approx(1.0, abs=0.01)


def test_partial_coverage_threshold(edges, roads):
    e = by_name(edges, "main_e")
    low = roadway_attributes(edges, roads, ["lanes"], min_coverage=0.4)
    high = roadway_attributes(edges, roads, ["lanes"], min_coverage=0.6)
    assert low.loc[e, "road_coverage"] == pytest.approx(0.5, abs=0.01)
    assert low.loc[e, "road_lanes"] == pytest.approx(4.0)
    assert np.isnan(high.loc[e, "road_lanes"])


def test_length_weighted_mean_and_majority(edges):
    roads = gpd.GeoDataFrame(
        {"speed": [30.0, 50.0], "surface": ["paved", "gravel"]},
        geometry=[line((0, 2), (75, 2)), line((75, 2), (100, 2))],
        crs=CRS,
    )
    out = conflate_lines(edges, roads, ["speed", "surface"])
    w = by_name(edges, "main_w")
    assert out.loc[w, "speed"] == pytest.approx(30 * 0.75 + 50 * 0.25)
    assert out.loc[w, "surface"] == "paved"


def test_explicit_max_aggregation(edges):
    roads = gpd.GeoDataFrame(
        {"speed": [30.0, 50.0]},
        geometry=[line((0, 2), (75, 2)), line((75, 2), (100, 2))],
        crs=CRS,
    )
    out = conflate_lines(edges, roads, ["speed"], aggregations={"speed": "max"})
    assert out.loc[by_name(edges, "main_w"), "speed"] == 50.0


def test_reprojects_source(edges, roads):
    out = roadway_attributes(edges, roads.to_crs("EPSG:4326"), ["lanes"])
    assert out.loc[by_name(edges, "main_w"), "road_lanes"] == pytest.approx(2.0)


def test_errors(edges, roads):
    with pytest.raises(KeyError):
        roadway_attributes(edges, roads, ["missing"])
    with pytest.raises(ValueError):
        roadway_attributes(edges, roads, ["lanes"], buffer_m=0)
    with pytest.raises(ValueError):
        roadway_attributes(edges, roads.set_crs(None, allow_override=True), ["lanes"])

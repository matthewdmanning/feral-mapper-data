import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from feral_mapper_data import traffic_attributes
from tests.conftest import CRS, X0, Y0, by_name, line


@pytest.fixture
def segments():
    return gpd.GeoDataFrame({"aadt": [12000]}, geometry=[line((0, 1), (100, 1))], crs=CRS)


@pytest.fixture
def stations():
    return gpd.GeoDataFrame(
        {"aadt": [800, 1200, 99999]},
        geometry=[
            Point(X0 + 150, Y0 + 5),  # main_e
            Point(X0 + 170, Y0 - 5),  # main_e
            Point(X0 + 50, Y0 + 500),  # too far from anything
        ],
        crs=CRS,
    )


def test_segments_then_station_fill(edges, segments, stations):
    out = traffic_attributes(edges, segments=segments, stations=stations)
    w, e, c = by_name(edges, "main_w"), by_name(edges, "main_e"), by_name(edges, "cross")
    assert out.loc[w, "aadt"] == pytest.approx(12000)
    assert out.loc[w, "aadt_source"] == "segment"
    assert out.loc[e, "aadt"] == pytest.approx(1000)  # mean of two stations
    assert out.loc[e, "aadt_source"] == "station"
    assert np.isnan(out.loc[c, "aadt"])
    assert pd.isna(out.loc[c, "aadt_source"])


def test_station_distance_limit(edges, stations):
    out = traffic_attributes(edges, stations=stations, max_station_distance_m=2)
    assert out["aadt"].isna().all()


def test_requires_a_source(edges):
    with pytest.raises(ValueError):
        traffic_attributes(edges)

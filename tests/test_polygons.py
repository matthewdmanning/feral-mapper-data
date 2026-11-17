import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from feral_mapper_data import polygon_attributes
from tests.conftest import CRS, X0, Y0, by_name


@pytest.fixture
def landcover():
    """Park on x in [0, 30], urban on [30, 100] around the main street; nothing east."""
    return gpd.GeoDataFrame(
        {"class": ["Park", "Urban Core"], "housing_density": [0.0, 10.0]},
        geometry=[
            box(X0 - 10, Y0 - 50, X0 + 30, Y0 + 50),
            box(X0 + 30, Y0 - 50, X0 + 100, Y0 + 50),
        ],
        crs=CRS,
    )


def test_length_fractions(edges, landcover):
    out = polygon_attributes(
        edges,
        landcover,
        category_column="class",
        value_columns=["housing_density"],
        prefix="lc_",
    )
    w, e = by_name(edges, "main_w"), by_name(edges, "main_e")
    assert out.loc[w, "lc_park_frac"] == pytest.approx(0.3)
    assert out.loc[w, "lc_urban_core_frac"] == pytest.approx(0.7)
    assert out.loc[w, "lc_majority"] == "Urban Core"
    assert out.loc[w, "lc_housing_density"] == pytest.approx(7.0)
    assert out.loc[w, "lc_coverage"] == pytest.approx(1.0)
    assert out.loc[e, "lc_park_frac"] == 0.0
    assert out.loc[e, "lc_coverage"] == 0.0
    assert np.isnan(out.loc[e, "lc_housing_density"])


def test_corridor_area_mode(edges, landcover):
    out = polygon_attributes(edges, landcover, category_column="class", buffer_m=10, prefix="lc_")
    w, e = by_name(edges, "main_w"), by_name(edges, "main_e")
    assert out.loc[w, "lc_park_frac"] == pytest.approx(0.3)
    assert out.loc[e, "lc_urban_core_frac"] == 0.0  # flat caps: no bleed past x=100


def test_validation(edges, landcover):
    with pytest.raises(ValueError):
        polygon_attributes(edges, landcover)
    with pytest.raises(KeyError):
        polygon_attributes(edges, landcover, category_column="nope")
    lines = gpd.GeoDataFrame({"class": ["x"]}, geometry=[edges.geometry.iloc[0]], crs=CRS)
    with pytest.raises(TypeError):
        polygon_attributes(edges, lines, category_column="class")

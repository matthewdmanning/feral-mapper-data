import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from feral_mapper_data._util import clean_geometries, slug
from feral_mapper_data.crs import CRSError, ensure_projected
from tests.conftest import CRS, line


def test_ensure_projected(network_lines):
    geographic = network_lines.to_crs("EPSG:4326")
    assert ensure_projected(geographic).crs.is_projected
    assert ensure_projected(network_lines).crs == network_lines.crs
    assert ensure_projected(network_lines, "EPSG:3857").crs.to_epsg() == 3857
    with pytest.raises(CRSError):
        ensure_projected(network_lines, "EPSG:4326")
    with pytest.raises(CRSError):
        ensure_projected(network_lines.set_crs(None, allow_override=True))


def test_clean_geometries_repairs_and_drops():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    gdf = gpd.GeoDataFrame(geometry=[bowtie, None, line((0, 0), (1, 0))], crs=CRS)
    out = clean_geometries(gdf)
    assert out.geometry.is_valid.all()
    assert out.geometry.notna().all()


def test_slug():
    assert slug("Urban Core!") == "urban_core"
    assert slug("") == "blank"

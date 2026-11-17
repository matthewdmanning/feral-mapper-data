import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from feral_mapper_data import raster_corridor_stats, write_ndvi
from tests.conftest import CRS, NODATA, X0, Y0, by_name, make_raster


@pytest.fixture
def ndvi(tmp_path):
    """0.2 west of x=100, 0.8 east of it."""
    data = np.full((20, 30), 0.2, dtype="float32")
    data[:, 15:] = 0.8
    return make_raster(tmp_path / "ndvi.tif", data)


def test_corridor_mean(edges, ndvi):
    out = raster_corridor_stats(edges, ndvi, buffer_m=15, stats=("mean", "std", "count"))
    w, e = by_name(edges, "main_w"), by_name(edges, "main_e")
    assert out.loc[w, "ndvi_mean"] == pytest.approx(0.2, abs=1e-6)
    assert out.loc[e, "ndvi_mean"] == pytest.approx(0.8, abs=1e-6)
    assert out.loc[w, "ndvi_std"] == pytest.approx(0.0, abs=1e-6)
    assert out.loc[w, "ndvi_valid_fraction"] == pytest.approx(1.0)
    assert out.loc[w, "ndvi_count"] > 0


def test_nodata_scale_and_valid_range(edges, tmp_path):
    data = np.full((20, 30), 5000, dtype="int16")  # MODIS-style: NDVI * 10000
    data[:, :10] = NODATA  # x < 50 is nodata
    data[0, :] = 32000  # out of range after scaling; outside corridor anyway
    path = make_raster(tmp_path / "modis.tif", data)
    out = raster_corridor_stats(
        edges, path, buffer_m=15, scale=1e-4, valid_range=(-1.0, 1.0), stats=("mean",)
    )
    w = by_name(edges, "main_w")
    assert out.loc[w, "ndvi_mean"] == pytest.approx(0.5)
    assert 0.3 < out.loc[w, "ndvi_valid_fraction"] < 0.7


def test_raster_in_other_crs(edges, ndvi, tmp_path):
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    dst_path = tmp_path / "ndvi_3857.tif"
    with rasterio.open(ndvi) as src:
        transform, w, h = calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds
        )
        profile = src.profile | {
            "crs": "EPSG:3857",
            "transform": transform,
            "width": w,
            "height": h,
        }
        with rasterio.open(dst_path, "w", **profile) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1), resampling=Resampling.nearest)
    out = raster_corridor_stats(edges, dst_path, buffer_m=15, stats=("mean",))
    assert out.loc[by_name(edges, "main_w"), "ndvi_mean"] == pytest.approx(0.2, abs=1e-6)


def test_disjoint_raster_gives_nan(edges, tmp_path):
    data = np.ones((5, 5), dtype="float32")
    path = tmp_path / "far.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=5,
        height=5,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=from_origin(X0 + 10_000, Y0, 10, 10),
    ) as dst:
        dst.write(data, 1)
    out = raster_corridor_stats(edges, path, stats=("mean",))
    assert out["ndvi_mean"].isna().all()


def test_rejects_unknown_stat(edges, ndvi):
    with pytest.raises(ValueError):
        raster_corridor_stats(edges, ndvi, stats=("mode",))


def test_write_ndvi(tmp_path):
    red = np.full((4, 4), 100, dtype="uint16")
    nir = np.full((4, 4), 300, dtype="uint16")
    nir[0, 0] = red[0, 0] = 0  # zero denominator -> NaN
    src = make_raster(tmp_path / "bands.tif", np.stack([red, nir]), nodata=None)
    out = write_ndvi(src, tmp_path / "ndvi.tif", red_band=1, nir_band=2)
    with rasterio.open(out) as ds:
        arr = ds.read(1)
    assert arr[1, 1] == pytest.approx(0.5)
    assert np.isnan(arr[0, 0])


def test_median_min_max_and_all_touched(edges, ndvi):
    out = raster_corridor_stats(edges, ndvi, buffer_m=15, stats=("median", "min", "max", "count"))
    touched = raster_corridor_stats(edges, ndvi, buffer_m=15, stats=("count",), all_touched=True)
    w = by_name(edges, "main_w")
    assert out.loc[w, "ndvi_median"] == pytest.approx(0.2, abs=1e-6)
    assert out.loc[w, "ndvi_min"] == pytest.approx(0.2, abs=1e-6)
    assert out.loc[w, "ndvi_max"] == pytest.approx(0.2, abs=1e-6)
    assert out.loc[w, "ndvi_count"] == pytest.approx(30.0)  # 100 m x 30 m / 100 m2 cells
    assert touched.loc[w, "ndvi_count"] >= out.loc[w, "ndvi_count"]


def test_weighted_median():
    from feral_mapper_data.attributes.vegetation import _weighted_median

    feat = np.array([0, 0, 0, 2, 2])
    vals = np.array([3.0, 1.0, 2.0, 9.0, 5.0])
    w = np.array([1.0, 1.0, 5.0, 1.0, 3.0])
    med = _weighted_median(feat, vals, w, 3)
    assert med[0] == 2.0 and np.isnan(med[1]) and med[2] == 5.0

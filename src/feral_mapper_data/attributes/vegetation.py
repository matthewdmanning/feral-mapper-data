"""Satellite vegetation / foliage rasters (e.g. NDVI, canopy cover) onto edges.

Stats are taken over a flat-capped ``buffer_m`` corridor alongside each edge (no bleed
past the edge ends into neighbouring edges). Corridors are reprojected to the raster CRS
(vector reprojection), so source pixels are never resampled.

Extraction: exactextract (C++, raster-sequential): one pass over raster blocks, exact
fraction of each cell covered by each corridor. Stats are coverage-weighted; with
``all_touched`` every touched cell has weight 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from exactextract import exact_extract
from exactextract.raster import RasterioRasterSource

from feral_mapper_data.geo.geometry import corridors

SUPPORTED_STATS = ("mean", "std", "min", "max", "median", "count")


def raster_corridor_stats(
    edges: gpd.GeoDataFrame,
    raster_path: str | Path,
    *,
    buffer_m: float = 30.0,
    band: int = 1,
    stats: Sequence[str] = ("mean", "std"),
    prefix: str = "ndvi_",
    scale: float = 1.0,
    offset: float = 0.0,
    valid_range: tuple[float, float] | None = None,
    all_touched: bool = False,
) -> pd.DataFrame:
    """Corridor statistics per edge plus ``{prefix}valid_fraction``.

    Values are ``raw * scale + offset`` (e.g. MODIS NDVI: scale=0.0001). Nodata and values
    outside ``valid_range`` (after scaling) are excluded. Stats are NaN with no valid cells.
    ``count`` is the covered cell count (fractional). ``valid_fraction`` is the share of the
    corridor area covered by valid cells (area off the raster counts as invalid).
    """
    bad = set(stats) - set(SUPPORTED_STATS)
    if bad:
        raise ValueError(f"Unsupported stats {sorted(bad)}; choose from {SUPPORTED_STATS}.")
    n = len(edges)
    zones = gpd.GeoSeries(corridors(edges.geometry.to_numpy(), buffer_m), crs=edges.crs)

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster {raster_path} has no CRS.")
        vec = gpd.GeoDataFrame(geometry=zones.to_crs(src.crs))
        cell_area = abs(src.transform.a * src.transform.e - src.transform.b * src.transform.d)
        extracted = exact_extract(
            RasterioRasterSource(src, band),
            vec,
            ["values", "coverage"],
            output="pandas",
            strategy="raster-sequential",
        )

    sizes = extracted["values"].map(len).to_numpy()
    feat = np.repeat(np.arange(n), sizes)
    vals = np.concatenate([*extracted["values"], np.empty(0)]) * scale + offset
    cov = np.concatenate([*extracted["coverage"], np.empty(0)])
    ok = np.isfinite(vals) & (cov > 0)
    if valid_range is not None:
        ok &= (vals >= valid_range[0]) & (vals <= valid_range[1])
    feat, vals, cov = feat[ok], vals[ok], cov[ok]
    w = np.ones_like(cov) if all_touched else cov

    sw = np.bincount(feat, weights=w, minlength=n)
    has = sw > 0
    mean = np.divide(
        np.bincount(feat, weights=w * vals, minlength=n), sw, where=has, out=np.full(n, np.nan)
    )
    columns: dict[str, np.ndarray] = {}
    for stat in stats:
        if stat == "mean":
            columns[stat] = mean
        elif stat == "std":
            dev = np.bincount(feat, weights=w * (vals - mean[feat]) ** 2, minlength=n)
            columns[stat] = np.sqrt(np.divide(dev, sw, where=has, out=np.full(n, np.nan)))
        elif stat == "count":
            columns[stat] = np.where(has, sw, np.nan)
        elif stat in ("min", "max"):
            grouped = pd.Series(vals).groupby(feat).agg(stat)
            columns[stat] = grouped.reindex(range(n)).to_numpy()
        else:
            columns[stat] = _weighted_median(feat, vals, w, n)
    area = vec.geometry.area.to_numpy()
    covered = np.bincount(feat, weights=cov, minlength=n) * cell_area
    valid = np.clip(np.divide(covered, area, where=area > 0, out=np.zeros(n)), 0.0, 1.0)
    out = pd.DataFrame({f"{prefix}{k}": v for k, v in columns.items()}, index=edges.index)
    out[f"{prefix}valid_fraction"] = valid
    return out


def _weighted_median(feat: np.ndarray, vals: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Per-group weighted median (lower median), vectorised via one lexsort."""
    out = np.full(n, np.nan)
    if feat.size == 0:
        return out
    order = np.lexsort((vals, feat))
    f, v, ww = feat[order], vals[order], w[order]
    within = pd.Series(ww).groupby(f).cumsum().to_numpy()  # cumulative weight per group
    hit = np.flatnonzero(within >= np.bincount(f, weights=ww, minlength=n)[f] / 2 - 1e-12)
    first = np.r_[True, f[hit][1:] != f[hit][:-1]]  # first crossing per group
    out[f[hit][first]] = v[hit][first]
    return out


def write_ndvi(src_path: str | Path, out_path: str | Path, *, red_band: int, nir_band: int) -> Path:
    """Write NDVI = (NIR - Red) / (NIR + Red) as float32 GeoTIFF, block by block."""
    out_path = Path(out_path)
    with rasterio.open(src_path) as src:
        profile = src.profile | {"count": 1, "dtype": "float32", "nodata": np.nan}
        profile.pop("photometric", None)
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                red = src.read(red_band, window=window, masked=True).astype("float32")
                nir = src.read(nir_band, window=window, masked=True).astype("float32")
                denom = nir + red
                with np.errstate(divide="ignore", invalid="ignore"):
                    ndvi = np.ma.where(denom == 0, np.ma.masked, (nir - red) / denom)
                dst.write(np.ma.filled(ndvi, np.nan).astype("float32"), 1, window=window)
    return out_path

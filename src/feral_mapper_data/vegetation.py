"""Satellite vegetation / foliage rasters (e.g. NDVI, canopy cover) onto edges.

Stats are taken over a flat-capped ``buffer_m`` corridor alongside each edge (no bleed
past the edge ends into neighbouring edges). Corridors are reprojected
to the raster CRS (vector reprojection), so source pixels are never resampled.
A pixel counts when its center falls in the corridor (``all_touched=False``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window

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
    outside ``valid_range`` (after scaling) are excluded. Stats are NaN with no valid pixels.
    """
    bad = set(stats) - set(SUPPORTED_STATS)
    if bad:
        raise ValueError(f"Unsupported stats {sorted(bad)}; choose from {SUPPORTED_STATS}.")
    if buffer_m <= 0:
        raise ValueError("buffer_m must be positive.")
    corridors = edges.geometry.buffer(buffer_m, cap_style="flat")
    results = np.full((len(edges), len(stats) + 1), np.nan)

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster {raster_path} has no CRS.")
        corridors = corridors.to_crs(src.crs)
        for i, geom in enumerate(corridors.to_numpy()):
            window = _window(src, geom.bounds)
            if window is None:
                continue
            data = src.read(band, window=window, masked=True)
            inside = geometry_mask(
                [geom],
                out_shape=data.shape,
                transform=src.window_transform(window),
                invert=True,
                all_touched=all_touched,
            )
            n_inside = int(inside.sum())
            if n_inside == 0:
                continue
            values = data.data[inside & ~np.ma.getmaskarray(data)].astype(float) * scale + offset
            values = values[np.isfinite(values)]
            if valid_range is not None:
                values = values[(values >= valid_range[0]) & (values <= valid_range[1])]
            results[i, -1] = values.size / n_inside
            if values.size:
                results[i, :-1] = [_stat(s, values) for s in stats]

    columns = [f"{prefix}{s}" for s in stats] + [f"{prefix}valid_fraction"]
    return pd.DataFrame(results, index=edges.index, columns=columns)


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


def _window(src: rasterio.DatasetReader, bounds: tuple[float, ...]) -> Window | None:
    """Pixel window covering ``bounds``, clipped to the raster; None if disjoint."""
    inv = ~src.transform
    xs, ys = (bounds[0], bounds[2]), (bounds[1], bounds[3])
    cols, rows = zip(*(inv @ (x, y) for x in xs for y in ys), strict=True)
    c0, c1 = max(0, math.floor(min(cols))), min(src.width, math.ceil(max(cols)))
    r0, r1 = max(0, math.floor(min(rows))), min(src.height, math.ceil(max(rows)))
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def _stat(name: str, values: np.ndarray) -> float:
    if name == "count":
        return float(values.size)
    return float(getattr(np, name)(values))

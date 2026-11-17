"""CRS handling. All metric work (buffers, lengths) happens in a projected CRS."""

from __future__ import annotations

import geopandas as gpd
from pyproj import CRS


class CRSError(ValueError):
    """Raised when a layer has no CRS or an unusable one."""


def ensure_projected(
    gdf: gpd.GeoDataFrame, target_crs: str | CRS | None = None
) -> gpd.GeoDataFrame:
    """Return ``gdf`` in a projected (metric) CRS.

    Uses ``target_crs`` if given, keeps an existing projected CRS, else estimates UTM.
    """
    if gdf.crs is None:
        raise CRSError("Layer has no CRS; set one before conversion.")
    if target_crs is not None:
        out = gdf.to_crs(target_crs)
        if not out.crs.is_projected:
            raise CRSError(f"target_crs {target_crs!r} is not projected.")
        return out
    if gdf.crs.is_projected:
        return gdf
    return gdf.to_crs(gdf.estimate_utm_crs())


def align_crs(layer: gpd.GeoDataFrame, reference: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    """Reproject ``layer`` to ``reference.crs``; refuse layers without a CRS."""
    if layer.crs is None:
        raise CRSError(f"{name} layer has no CRS.")
    if reference.crs is None:
        raise CRSError("Edge layer has no CRS.")
    return layer if layer.crs == reference.crs else layer.to_crs(reference.crs)

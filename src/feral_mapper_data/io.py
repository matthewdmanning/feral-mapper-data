"""Vector/graph I/O with format detection by file extension."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np

PARQUET = {".parquet", ".geoparquet"}


def read_vector(path: str | Path, layer: str | None = None) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in PARQUET:
        return gpd.read_parquet(path)
    return gpd.read_file(path, layer=layer)


def write_vector(gdf: gpd.GeoDataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in PARQUET:
        gdf.to_parquet(path)
    else:
        gdf.to_file(path)
    return path


def read_graphml(path: str | Path) -> nx.MultiDiGraph:
    """Load an osmnx GraphML file (needs the ``osm`` extra for typed attributes)."""
    try:
        import osmnx as ox
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Reading GraphML needs osmnx: install 'feral-mapper-data[osm]'.") from exc
    return ox.load_graphml(path)


def write_graphml(graph: nx.MultiDiGraph, path: str | Path) -> Path:
    """Write GraphML; geometries as WKT, CRS as a string, numpy scalars unwrapped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = graph.copy()
    out.graph = {k: str(v) for k, v in out.graph.items()}
    for _, _, data in out.edges(data=True):
        for key, value in list(data.items()):
            if hasattr(value, "wkt"):
                data[key] = value.wkt
            elif isinstance(value, np.generic):
                data[key] = value.item()
            elif not isinstance(value, str | int | float | bool):
                data[key] = str(value)
    nx.write_graphml(out, path)
    return path

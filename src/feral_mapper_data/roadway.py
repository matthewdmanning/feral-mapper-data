"""Roadway inventory attributes (lanes, speed, functional class, surface...) onto edges."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import geopandas as gpd
import pandas as pd

from feral_mapper_data.conflate import Aggregation, conflate_lines


def roadway_attributes(
    edges: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    columns: Sequence[str],
    *,
    buffer_m: float = 10.0,
    min_coverage: float = 0.5,
    aggregations: Mapping[str, Aggregation] | None = None,
    prefix: str = "road_",
) -> pd.DataFrame:
    """Conflate roadway-inventory line attributes onto edges. See ``conflate_lines``."""
    return conflate_lines(
        edges,
        roads,
        columns,
        buffer_m=buffer_m,
        min_coverage=min_coverage,
        aggregations=aggregations,
        prefix=prefix,
        name="roadway",
    )

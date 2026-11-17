"""Pipeline configuration loaded from TOML. Unknown keys are errors, not silently ignored.

Relative paths resolve against the config file's directory.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from feral_mapper_data.attributes._conflate import Aggregation
from feral_mapper_data.attributes.vegetation import Stat


class ConfigError(ValueError):
    """Invalid pipeline configuration."""


class _Section(BaseModel):
    """Frozen table that rejects unknown keys and resolves paths against ``base_dir``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def _resolve_path(cls, value: object, info: ValidationInfo) -> object:
        base_dir = (info.context or {}).get("base_dir", Path())
        return base_dir / value if isinstance(value, Path) else value


class NetworkConfig(_Section):
    path: Path
    layer: str | None = None
    crs: str | None = None
    node_tolerance: float = 0.5


class RoadwayConfig(_Section):
    path: Path
    columns: list[str]
    layer: str | None = None
    buffer_m: float = 10.0
    min_coverage: float = 0.5
    aggregations: dict[str, Aggregation] = Field(default_factory=dict)
    prefix: str = "road_"


class TrafficConfig(_Section):
    segments: Path | None = None
    segments_layer: str | None = None
    segment_column: str = "aadt"
    stations: Path | None = None
    stations_layer: str | None = None
    station_column: str = "aadt"
    buffer_m: float = 10.0
    min_coverage: float = 0.5
    max_station_distance_m: float = 30.0
    prefix: str = ""

    @model_validator(mode="after")
    def _needs_a_source(self) -> Self:
        if self.segments is None and self.stations is None:
            raise ValueError("[traffic] needs 'segments' and/or 'stations'.")
        return self


class VegetationConfig(_Section):
    path: Path
    prefix: str = "ndvi_"
    buffer_m: float = 30.0
    band: int = 1
    stats: list[Stat] = Field(default=["mean", "std"])  # pydantic copies defaults
    scale: float = 1.0
    offset: float = 0.0
    valid_range: tuple[float, float] | None = None
    all_touched: bool = False


class PolygonConfig(_Section):
    path: Path
    prefix: str
    layer: str | None = None
    category_column: str | None = None
    value_columns: list[str] = Field(default_factory=list)
    buffer_m: float | None = None


class CoarsenConfig(_Section):
    columns: list[str]
    tolerances: dict[str, float] = Field(default_factory=dict)
    min_region_nodes: int = 3
    tile_m: float | None = None
    hull_ratio: float = 0.3


class OutputConfig(_Section):
    edges: Path
    graphml: Path | None = None
    coarse_nodes: Path | None = None
    coarse_edges: Path | None = None
    coarse_graphml: Path | None = None
    geosparql: Path | None = None
    geosparql_crs: Literal["crs84", "native"] = "crs84"


class PipelineConfig(_Section):
    network: NetworkConfig
    output: OutputConfig
    roadway: RoadwayConfig | None = None
    traffic: TrafficConfig | None = None
    vegetation: list[VegetationConfig] = Field(default_factory=list)
    polygons: list[PolygonConfig] = Field(default_factory=list)
    coarsen: CoarsenConfig | None = None

    @model_validator(mode="after")
    def _coarsen_matches_outputs(self) -> Self:
        out = self.output
        has_outputs = any((out.coarse_nodes, out.coarse_edges, out.coarse_graphml))
        if self.coarsen is not None and not has_outputs:
            raise ValueError(
                "[coarsen] needs an [output] coarse_nodes/coarse_edges/coarse_graphml."
            )
        if self.coarsen is None and has_outputs:
            raise ValueError("[output] coarse_* paths need a [coarsen] section.")
        return self

    @classmethod
    def from_toml(cls, path: str | Path) -> PipelineConfig:
        path = Path(path)
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        return cls.from_dict(raw, base_dir=path.parent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_dir: Path = Path()) -> PipelineConfig:
        try:
            return cls.model_validate(raw, context={"base_dir": base_dir})
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc

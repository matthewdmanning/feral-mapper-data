"""Pipeline configuration loaded from TOML. Unknown keys are errors, not silently ignored.

Relative paths resolve against the config file's directory.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class ConfigError(ValueError):
    """Invalid pipeline configuration."""


@dataclass(frozen=True)
class NetworkConfig:
    path: Path
    layer: str | None = None
    crs: str | None = None
    node_tolerance: float = 0.5


@dataclass(frozen=True)
class RoadwayConfig:
    path: Path
    columns: list[str]
    layer: str | None = None
    buffer_m: float = 10.0
    min_coverage: float = 0.5
    aggregations: dict[str, str] = field(default_factory=dict)
    prefix: str = "road_"


@dataclass(frozen=True)
class TrafficConfig:
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


@dataclass(frozen=True)
class VegetationConfig:
    path: Path
    prefix: str = "ndvi_"
    buffer_m: float = 30.0
    band: int = 1
    stats: list[str] = field(default_factory=lambda: ["mean", "std"])
    scale: float = 1.0
    offset: float = 0.0
    valid_range: list[float] | None = None
    all_touched: bool = False


@dataclass(frozen=True)
class PolygonConfig:
    path: Path
    prefix: str
    layer: str | None = None
    category_column: str | None = None
    value_columns: list[str] = field(default_factory=list)
    buffer_m: float | None = None


@dataclass(frozen=True)
class CoarsenConfig:
    columns: list[str]
    tolerances: dict[str, float] = field(default_factory=dict)
    min_region_nodes: int = 3
    tile_m: float | None = None
    hull_ratio: float = 0.3


@dataclass(frozen=True)
class OutputConfig:
    edges: Path
    graphml: Path | None = None
    coarse_nodes: Path | None = None
    coarse_edges: Path | None = None
    coarse_graphml: Path | None = None
    geosparql: Path | None = None
    geosparql_crs: str = "crs84"


@dataclass(frozen=True)
class PipelineConfig:
    network: NetworkConfig
    output: OutputConfig
    roadway: RoadwayConfig | None = None
    traffic: TrafficConfig | None = None
    vegetation: list[VegetationConfig] = field(default_factory=list)
    polygons: list[PolygonConfig] = field(default_factory=list)
    coarsen: CoarsenConfig | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> PipelineConfig:
        path = Path(path)
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        return cls.from_dict(raw, base_dir=path.parent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_dir: Path = Path()) -> PipelineConfig:
        _reject_unknown(cls, raw, "top level")
        for required in ("network", "output"):
            if required not in raw:
                raise ConfigError(f"Missing [{required}] section.")

        def one(kind: type[T], key: str) -> T | None:
            return _build(kind, raw[key], key, base_dir) if key in raw else None

        def many(kind: type[T], key: str) -> list[T]:
            items = raw.get(key, [])
            if not isinstance(items, list):
                raise ConfigError(f"[{key}] must be an array of tables: use [[{key}]].")
            return [_build(kind, item, f"{key}[{i}]", base_dir) for i, item in enumerate(items)]

        traffic = one(TrafficConfig, "traffic")
        if traffic is not None and traffic.segments is None and traffic.stations is None:
            raise ConfigError("[traffic] needs 'segments' and/or 'stations'.")
        output = _build(OutputConfig, raw["output"], "output", base_dir)
        coarsen = one(CoarsenConfig, "coarsen")
        coarse_outputs = (output.coarse_nodes, output.coarse_edges, output.coarse_graphml)
        if coarsen is not None and all(p is None for p in coarse_outputs):
            raise ConfigError(
                "[coarsen] needs an [output] coarse_nodes/coarse_edges/coarse_graphml."
            )
        if coarsen is None and any(p is not None for p in coarse_outputs):
            raise ConfigError("[output] coarse_* paths need a [coarsen] section.")
        if output.geosparql_crs not in ("crs84", "native"):
            raise ConfigError("[output] geosparql_crs must be 'crs84' or 'native'.")
        return cls(
            network=_build(NetworkConfig, raw["network"], "network", base_dir),
            output=output,
            coarsen=coarsen,
            roadway=one(RoadwayConfig, "roadway"),
            traffic=traffic,
            vegetation=many(VegetationConfig, "vegetation"),
            polygons=many(PolygonConfig, "polygons"),
        )


_PATH_KEYS = {
    "path",
    "segments",
    "stations",
    "edges",
    "graphml",
    "coarse_nodes",
    "coarse_edges",
    "coarse_graphml",
    "geosparql",
}


def _reject_unknown(kind: type, raw: dict[str, Any], where: str) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(f"[{where}] must be a table.")
    unknown = set(raw) - {f.name for f in fields(kind)}
    if unknown:
        raise ConfigError(f"Unknown keys in [{where}]: {sorted(unknown)}")


def _build[T](kind: type[T], raw: dict[str, Any], where: str, base_dir: Path) -> T:
    _reject_unknown(kind, raw, where)
    values = {k: (base_dir / v if k in _PATH_KEYS else v) for k, v in raw.items()}
    try:
        return kind(**values)
    except TypeError as exc:
        raise ConfigError(f"[{where}]: {exc}") from exc

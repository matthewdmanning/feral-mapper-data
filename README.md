# feral-mapper-data
Data backend and configuration interaction for training feral cat population models.

## Edge attribute conversion

Puts roadway, average traffic (AADT), satellite foliage (NDVI), and polygon data onto road-graph edges.
Details: [docs/edge-attributes.md](docs/edge-attributes.md).

```bash
uv sync                                   # add --extra osm for GraphML/osmnx networks
uv run feral-edges examples/pipeline.toml
```

Library use:

```python
from feral_mapper_data import edges_from_lines, roadway_attributes, raster_corridor_stats

edges = edges_from_lines(roads_utm)
attrs = roadway_attributes(edges, inventory, ["lanes", "speed_limit"])
```

## Development

```bash
uv run pytest            # tests (synthetic fixtures, no downloads)
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

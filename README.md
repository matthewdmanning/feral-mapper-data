# feral-mapper-data
Data backend and configuration interaction for training feral cat population models.

## Edge attribute conversion

Puts roadway, average traffic (AADT), satellite foliage (NDVI), and polygon data onto road-graph edges.
Collapses large homogeneous areas into region nodes; exports GeoSPARQL 1.1 RDF.
Details: [docs/edge-attributes.md](docs/edge-attributes.md).

```bash
uv sync               # extras: --extra osm (GraphML/osmnx), --extra rdf (GeoSPARQL)
uv run feral-edges examples/pipeline.toml
```

Library use:

```python
from feral_mapper_data import coarsen_homogeneous, edges_from_lines, roadway_attributes

edges = edges_from_lines(roads_utm)
attrs = roadway_attributes(edges, inventory, ["lanes", "speed_limit"])
coarse = coarsen_homogeneous(edges.join(attrs), ["road_lanes"])  # CoarseGraph
```

## Development

```bash
uv run --extra rdf pytest   # tests (synthetic fixtures, no downloads)
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

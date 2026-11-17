# Edge attribute conversion

Converts roadway, traffic (AADT), satellite vegetation, and polygon layers onto road-graph edges.

## Data model

- Edge table: GeoDataFrame, index `(u, v, key)` (osmnx convention), LineString geometry, projected CRS.
- Every converter is pure: `f(edges, source, ...) -> DataFrame` indexed like `edges`. No mutation.
- Sources are reprojected to the edge CRS; rasters are never resampled (corridors go to raster CRS).
- Invalid geometries repaired (`make_valid`); null/empty dropped; multi-parts exploded.

## Module layout

Layers import only from layers above them.

| Package | Role |
|---|---|
| `geo/` | CRS handling, geometry cleaning and corridors, per-edge aggregation |
| `network/` | `(u, v, key)` schema, edges from lines, graph <-> edge table |
| `attributes/` | Converters: roadway, traffic, vegetation, polygons (shared `_conflate.py`) |
| `coarsen.py` | Homogeneous-region coarsening |
| `io/` | Vector and GraphML files (`files.py`), GeoSPARQL (`geosparql.py`) |
| `app/` | TOML config, `run_pipeline`, CLI |

The public API stays importable from `feral_mapper_data` directly.

## Converters

| Source | Function | Method | Output columns |
|---|---|---|---|
| Roadway lines | `roadway_attributes` | line conflation | `road_<col>`, `road_coverage` |
| AADT segments | `traffic_attributes` | line conflation, weighted mean | `aadt`, `aadt_source=segment` |
| AADT stations | `traffic_attributes` | nearest edge ≤ `max_station_distance_m`, mean | `aadt`, `aadt_source=station` |
| Raster (NDVI) | `raster_corridor_stats` | exactextract, coverage-weighted stats in corridor | `ndvi_<stat>`, `ndvi_valid_fraction` |
| Polygons | `polygon_attributes` | overlap length, or corridor area with `buffer_m` | `<p><cat>_frac`, `<p>majority`, `<p><col>`, `<p>coverage` |

Bands → NDVI: `write_ndvi(src, out, red_band=, nir_band=)`.

### Line conflation (`attributes/_conflate.py`)

1. Flat-capped corridor of `buffer_m` per edge.
2. Clip source lines to corridor.
3. Weight = edge distance spanned (|Δ linear-reference| of piece endpoints). Crossing streets → ~0 weight; no angle filter needed.
4. `coverage` = Σweight / edge length, clipped to [0, 1]. Attributes NaN below `min_coverage`.
5. Aggregation per column: `weighted_mean` (numeric default), `majority` (other default), `max`, `min`.

### Raster stats (`attributes/vegetation.py`)

- exactextract (C++), `raster-sequential`: one pass over raster blocks, exact cell-coverage fractions. No per-edge reads/rasterisation.
- Stats coverage-weighted (`all_touched=True` → weight 1 per touched cell). `count` = covered cells (fractional). `median` weighted.
- `valid_fraction` = valid covered area / corridor area; off-raster area counts invalid (disjoint → 0).

### Polygon overlay (`attributes/polygons.py`)

Polygons `shapely.prepare`d; pairs where polygon `contains_properly` target take full target measure. Overlay (`intersection`) only on boundary-crossing pairs: big land-cover polygons cost ~O(boundary edges), not O(all edges).

### Corridors

Flat caps everywhere: corridors stop at edge ends, so values do not bleed into neighbouring edges.

## Caveats

- `edges_from_lines` needs noded input (lines break at intersections); it snaps endpoints, does not split crossings.
- Endpoint snapping: STRtree `dwithin` pairs + scipy `connected_components`. Single linkage: a chain of points each ≤ tolerance apart merges even if ends are farther.
- Directional AADT on divided roads is averaged onto a shared centerline, not summed.
- Overlapping polygons within one layer can push `_frac`/`coverage` above 1.
- Category columns use slugged names (`"Urban Core"` → `urban_core`); slug collisions merge.
- Raster stats hold per-cell values for all corridors (≈ 16 B × covered cells); chunk edges for continental runs.
- Prefix clashes across blocks fail fast in the pipeline.

## Coarsening (`coarsen.py`)

`coarsen_homogeneous(edges, columns, tolerances=, min_region_nodes=3, tile_m=None)` → `CoarseGraph(nodes, edges, membership)`.

Goal: collapse large homogeneous areas into region nodes; heterogeneous areas keep full detail. Goes past osmnx `simplify_graph(edge_attrs_differ=...)`, which keeps every junction.

1. Edge class = quantised attribute tuple: numeric `floor(x/tol)` (exact if no tol), others exact, missing = own class.
2. Interior node: all incident edges one class (and one `tile_m` grid cell).
3. Regions = connected components (scipy sparse) of interior nodes; size < `min_region_nodes` → untouched.
4. Edges inside a region absorbed; others re-pointed to region ids. Boundary nodes never merge → class borders exact.

Region node: `kind=region`, concave-hull geometry, `n_members`, `n_internal_edges`, `internal_length_m`, `area_m2`, length-weighted mean/majority of `columns`; `x`/`y` = member node nearest centroid. `CoarseGraph.to_networkx()` → MultiDiGraph.

Caveats: fixed bins split values straddling a bin edge (9.9 / 10.1 with tol 10); kept edge geometry is original (may not end at region `x`/`y`); `orig_u`/`orig_v`/`orig_key` trace sources.

## GeoSPARQL export (`io/geosparql.py`, `[rdf]` extra)

GeoSPARQL 1.1 RDF for knowledge graphs / triplestores. GeoParquet stays primary (columnar, training input).

- Row → `geo:Feature` + `fm:Edge|Node|Region`; `geo:hasDefaultGeometry` → `<feature>/geometry` (`geo:Geometry`, `geo:asWKT`); `geo:hasMetricLength` (lines) / `geo:hasMetricArea` (regions), from projected CRS.
- Edges: `fm:from`/`fm:to`. Original nodes `geo:sfWithin` their region. All in one `geo:FeatureCollection` (`rdfs:member`).
- Attributes → `fm:<column>` xsd-typed literals; NaN omitted.
- WKT: `crs84` (default) lon/lat, no CRS prefix = GeoSPARQL default; some stores (RDF4J) support nothing else. `native` → `<http://www.opengis.net/def/crs/EPSG/0/{code}>` prefix.
- rdflib builds in memory: fine regionally; for very large graphs prefer `.nt`.

## Storage

GeoParquet 1.1 with bbox covering column (`write_covering_bbox`): row-group spatial filter pushdown in DuckDB/GDAL/pyarrow.

## Pipeline

`feral-edges CONFIG.toml` — see `examples/pipeline.toml`. Unknown config keys are errors.
Outputs: GeoParquet edge table; optional GraphML (geometry as WKT, NaN attributes omitted).
`[coarsen]` (columns refer to output attribute names, e.g. `lc_majority`) needs ≥1 of `coarse_nodes` / `coarse_edges` / `coarse_graphml`. `geosparql` exports coarse graph when `[coarsen]` set, else edges.

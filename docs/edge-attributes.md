# Edge attribute conversion

Converts roadway, traffic (AADT), satellite vegetation, and polygon layers onto road-graph edges.

## Data model

- Edge table: GeoDataFrame, index `(u, v, key)` (osmnx convention), LineString geometry, projected CRS.
- Every converter is pure: `f(edges, source, ...) -> DataFrame` indexed like `edges`. No mutation.
- Sources are reprojected to the edge CRS; rasters are never resampled (corridors go to raster CRS).
- Invalid geometries repaired (`make_valid`); null/empty dropped; multi-parts exploded.

## Converters

| Source | Function | Method | Output columns |
|---|---|---|---|
| Roadway lines | `roadway_attributes` | line conflation | `road_<col>`, `road_coverage` |
| AADT segments | `traffic_attributes` | line conflation, weighted mean | `aadt`, `aadt_source=segment` |
| AADT stations | `traffic_attributes` | nearest edge ≤ `max_station_distance_m`, mean | `aadt`, `aadt_source=station` |
| Raster (NDVI) | `raster_corridor_stats` | pixel-centre stats in corridor | `ndvi_<stat>`, `ndvi_valid_fraction` |
| Polygons | `polygon_attributes` | overlap length, or corridor area with `buffer_m` | `<p><cat>_frac`, `<p>majority`, `<p><col>`, `<p>coverage` |

Bands → NDVI: `write_ndvi(src, out, red_band=, nir_band=)`.

### Line conflation (`conflate.py`)

1. Flat-capped corridor of `buffer_m` per edge.
2. Clip source lines to corridor.
3. Weight = edge distance spanned (|Δ linear-reference| of piece endpoints). Crossing streets → ~0 weight; no angle filter needed.
4. `coverage` = Σweight / edge length, clipped to [0, 1]. Attributes NaN below `min_coverage`.
5. Aggregation per column: `weighted_mean` (numeric default), `majority` (other default), `max`, `min`.

### Corridors

Flat caps everywhere: corridors stop at edge ends, so values do not bleed into neighbouring edges.

## Caveats

- `edges_from_lines` needs noded input (lines break at intersections); it snaps endpoints, does not split crossings.
- Directional AADT on divided roads is averaged onto a shared centerline, not summed.
- Overlapping polygons within one layer can push `_frac`/`coverage` above 1.
- Category columns use slugged names (`"Urban Core"` → `urban_core`); slug collisions merge.
- Raster stats loop per edge with windowed reads: memory-flat, CPU-linear in edge count.
- Prefix clashes across blocks fail fast in the pipeline.

## Pipeline

`feral-edges CONFIG.toml` — see `examples/pipeline.toml`. Unknown config keys are errors.
Outputs: GeoParquet edge table; optional GraphML (geometry as WKT, NaN attributes omitted).

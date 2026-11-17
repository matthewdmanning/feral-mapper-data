"""Convert geospatial layers (roadway, traffic, vegetation, polygons) onto road-graph edges."""

from feral_mapper_data.graph import (
    EDGE_INDEX,
    attach_to_graph,
    edges_from_graph,
    edges_from_lines,
    graph_from_edges,
)
from feral_mapper_data.polygons import polygon_attributes
from feral_mapper_data.roadway import roadway_attributes
from feral_mapper_data.traffic import traffic_attributes
from feral_mapper_data.vegetation import raster_corridor_stats, write_ndvi

__all__ = [
    "EDGE_INDEX",
    "attach_to_graph",
    "edges_from_graph",
    "edges_from_lines",
    "graph_from_edges",
    "polygon_attributes",
    "raster_corridor_stats",
    "roadway_attributes",
    "traffic_attributes",
    "write_ndvi",
]

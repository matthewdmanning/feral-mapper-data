"""End-to-end conversion: network + source layers -> attributed edge table / graph."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from feral_mapper_data.app.config import PipelineConfig
from feral_mapper_data.attributes.polygons import polygon_attributes
from feral_mapper_data.attributes.roadway import roadway_attributes
from feral_mapper_data.attributes.traffic import traffic_attributes
from feral_mapper_data.attributes.vegetation import raster_corridor_stats
from feral_mapper_data.coarsen import CoarseGraph, coarsen_homogeneous
from feral_mapper_data.geo.crs import ensure_projected
from feral_mapper_data.io.files import read_graphml, read_vector, write_graphml, write_vector
from feral_mapper_data.network.edges import edges_from_lines
from feral_mapper_data.network.graph import attach_to_graph, edges_from_graph, graph_from_edges

log = logging.getLogger(__name__)

GRAPH_SUFFIXES = {".graphml"}


def load_edges(config: PipelineConfig) -> gpd.GeoDataFrame:
    net = config.network
    if net.path.suffix.lower() in GRAPH_SUFFIXES:
        edges = edges_from_graph(read_graphml(net.path))
        return ensure_projected(edges, net.crs)
    lines = ensure_projected(read_vector(net.path, net.layer), net.crs)
    return edges_from_lines(lines, node_tolerance=net.node_tolerance)


def build_edge_attributes(config: PipelineConfig, edges: gpd.GeoDataFrame) -> pd.DataFrame:
    """All configured attribute blocks, indexed like ``edges``."""
    blocks: list[pd.DataFrame] = []
    if (rc := config.roadway) is not None:
        log.info("Roadway: %s", rc.path)
        blocks.append(
            roadway_attributes(
                edges,
                read_vector(rc.path, rc.layer),
                rc.columns,
                buffer_m=rc.buffer_m,
                min_coverage=rc.min_coverage,
                aggregations=rc.aggregations,
                prefix=rc.prefix,
            )
        )
    if (tc := config.traffic) is not None:
        log.info("Traffic: segments=%s stations=%s", tc.segments, tc.stations)
        blocks.append(
            traffic_attributes(
                edges,
                segments=read_vector(tc.segments, tc.segments_layer) if tc.segments else None,
                stations=read_vector(tc.stations, tc.stations_layer) if tc.stations else None,
                segment_column=tc.segment_column,
                station_column=tc.station_column,
                buffer_m=tc.buffer_m,
                min_coverage=tc.min_coverage,
                max_station_distance_m=tc.max_station_distance_m,
                prefix=tc.prefix,
            )
        )
    for vc in config.vegetation:
        log.info("Vegetation: %s", vc.path)
        blocks.append(
            raster_corridor_stats(
                edges,
                vc.path,
                buffer_m=vc.buffer_m,
                band=vc.band,
                stats=vc.stats,
                prefix=vc.prefix,
                scale=vc.scale,
                offset=vc.offset,
                valid_range=vc.valid_range,
                all_touched=vc.all_touched,
            )
        )
    for pc in config.polygons:
        log.info("Polygons: %s", pc.path)
        blocks.append(
            polygon_attributes(
                edges,
                read_vector(pc.path, pc.layer),
                category_column=pc.category_column,
                value_columns=pc.value_columns,
                buffer_m=pc.buffer_m,
                prefix=pc.prefix,
            )
        )
    attrs = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=edges.index)
    clashes = attrs.columns[attrs.columns.duplicated()].tolist()
    clashes += [c for c in attrs.columns if c in edges.columns]
    if clashes:
        raise ValueError(f"Attribute column name clashes; adjust prefixes: {sorted(set(clashes))}")
    return attrs


def run_pipeline(config: PipelineConfig) -> gpd.GeoDataFrame:
    edges = load_edges(config)
    log.info("Loaded %d edges (CRS %s)", len(edges), edges.crs)
    attrs = build_edge_attributes(config, edges)
    result = gpd.GeoDataFrame(pd.concat([edges, attrs], axis=1), geometry="geometry", crs=edges.crs)
    write_vector(result, config.output.edges)
    log.info("Wrote %s", config.output.edges)
    if config.output.graphml is not None:
        _write_graph(config, result, attrs, config.output.graphml)
    coarse = _coarsen_and_write(config, result)
    if config.output.geosparql is not None:
        from feral_mapper_data.io.geosparql import to_geosparql, write_geosparql

        crs = config.output.geosparql_crs
        if coarse is not None:
            rdf = to_geosparql(coarse.edges, coarse.nodes, coarse.membership, wkt_crs=crs)
        else:
            rdf = to_geosparql(result, wkt_crs=crs)
        write_geosparql(rdf, config.output.geosparql)
        log.info("Wrote %s (%d triples)", config.output.geosparql, len(rdf))
    return result


def _coarsen_and_write(config: PipelineConfig, result: gpd.GeoDataFrame) -> CoarseGraph | None:
    if (cc := config.coarsen) is None:
        return None
    coarse = coarsen_homogeneous(result, **cc.model_dump())  # config fields mirror the signature
    n_in = len(coarse.membership)
    log.info("Coarsened %d nodes -> %d (%d edges kept)", n_in, len(coarse.nodes), len(coarse.edges))
    out = config.output
    if out.coarse_nodes is not None:
        nodes = coarse.nodes.assign(
            members=coarse.membership.index.to_series()
            .groupby(coarse.membership.to_numpy())
            .agg(list)
            .reindex(coarse.nodes.index)
        )
        write_vector(nodes, out.coarse_nodes)
    if out.coarse_edges is not None:
        write_vector(coarse.edges.reset_index(), out.coarse_edges)
    if out.coarse_graphml is not None:
        write_graphml(coarse.to_networkx(), out.coarse_graphml)
    return coarse


def _write_graph(
    config: PipelineConfig, result: gpd.GeoDataFrame, attrs: pd.DataFrame, path: Path
) -> None:
    if config.network.path.suffix.lower() in GRAPH_SUFFIXES:
        graph = attach_to_graph(read_graphml(config.network.path), attrs)
    else:
        graph = graph_from_edges(result)
    write_graphml(graph, path)
    log.info("Wrote %s", path)

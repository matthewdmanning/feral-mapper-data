"""GeoSPARQL 1.1 (OGC 22-047) export of edge / coarse-node tables as RDF.

Interchange format for knowledge graphs and triplestores; GeoParquet stays the primary store
(columnar, fast, what training reads). Needs the ``rdf`` extra (rdflib).

Modelling:

- Each row -> ``geo:Feature`` + ``fm:Edge``/``fm:Node``/``fm:Region``, one ``geo:Geometry``
  (IRI ``<feature>/geometry``) via ``geo:hasDefaultGeometry``; ``geo:asWKT``;
  ``geo:hasMetricLength`` (lines) or
  ``geo:hasMetricArea`` (regions), measured in the projected source CRS.
- Edges link endpoints with ``fm:from``/``fm:to``; members link to regions with ``geo:sfWithin``.
- Attributes -> ``fm:<column>`` typed literals (xsd from dtype); NaN omitted.
- WKT CRS: ``crs84`` (default) reprojects to lon/lat with no CRS prefix: GeoSPARQL's default CRS
  and the only one some stores (e.g. RDF4J) support. ``native`` keeps source coordinates with an
  ``<http://www.opengis.net/def/crs/EPSG/0/{code}>`` prefix.
- All features go in one ``geo:FeatureCollection``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd

from feral_mapper_data.graph import _check_index

if TYPE_CHECKING:
    from rdflib import Graph

DEFAULT_BASE = "urn:feral-mapper:"
_FORMATS = {".ttl": "turtle", ".nt": "nt", ".jsonld": "json-ld", ".rdf": "xml", ".xml": "xml"}


def to_geosparql(
    edges: gpd.GeoDataFrame | None = None,
    nodes: gpd.GeoDataFrame | None = None,
    membership: pd.Series | None = None,
    *,
    base_iri: str = DEFAULT_BASE,
    wkt_crs: Literal["crs84", "native"] = "crs84",
    collection: str = "network",
) -> Graph:
    """RDF graph of ``edges`` (indexed u, v, key) and/or ``nodes`` (e.g. ``CoarseGraph.nodes``).

    ``membership`` (original node -> coarse id, from coarsening) adds ``geo:sfWithin`` links.
    """
    try:
        from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef
        from rdflib.namespace import GEO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "GeoSPARQL export needs rdflib: install 'feral-mapper-data[rdf]'."
        ) from exc

    fm = Namespace(f"{base_iri}vocab#")
    graph = Graph()
    graph.bind("geo", GEO)
    graph.bind("fm", fm)
    coll = URIRef(f"{base_iri}collection/{collection}")
    graph.add((coll, RDF.type, GEO.FeatureCollection))

    def node_iri(nid: Any) -> URIRef:
        return URIRef(f"{base_iri}node/{nid}")

    def datatype(series: pd.Series) -> Any:
        """xsd type chosen once per column from its dtype (None -> plain string)."""
        if pd.api.types.is_bool_dtype(series):
            return XSD.boolean
        if pd.api.types.is_integer_dtype(series):
            return XSD.integer
        if pd.api.types.is_float_dtype(series):
            return XSD.double
        return None

    def add_features(
        frame: gpd.GeoDataFrame, iris: list[URIRef], kinds: list[URIRef], skip: set[str]
    ) -> None:
        wkts = _wkts(frame, wkt_crs)
        metric_len = frame.geometry.length.to_numpy()
        metric_area = frame.geometry.area.to_numpy()
        attrs = frame.drop(columns=[frame.geometry.name, *skip], errors="ignore")
        for i, (iri, kind) in enumerate(zip(iris, kinds, strict=True)):
            geom = URIRef(f"{iri}/geometry")
            graph.add((iri, RDF.type, GEO.Feature))
            graph.add((iri, RDF.type, kind))
            graph.add((coll, RDFS.member, iri))
            graph.add((iri, GEO.hasDefaultGeometry, geom))
            graph.add((geom, RDF.type, GEO.Geometry))
            graph.add((geom, GEO.asWKT, Literal(wkts[i], datatype=GEO.wktLiteral)))
            if metric_area[i] > 0:
                graph.add(
                    (iri, GEO.hasMetricArea, Literal(float(metric_area[i]), datatype=XSD.double))
                )
            elif metric_len[i] > 0:
                graph.add(
                    (iri, GEO.hasMetricLength, Literal(float(metric_len[i]), datatype=XSD.double))
                )
        # Attributes column by column: one dtype decision and one NaN mask per column.
        iri_arr = np.asarray(iris, dtype=object)
        for col in attrs.columns:
            pred, series, dt = fm[_local(str(col))], attrs[col], datatype(attrs[col])
            mask = series.notna().to_numpy()
            values = series.to_numpy()[mask].tolist()  # numpy scalars -> Python
            for iri, value in zip(iri_arr[mask], values, strict=True):
                lit = Literal(value, datatype=dt) if dt is not None else Literal(str(value))
                graph.add((iri, pred, lit))

    if edges is not None:
        _check_index(edges)
        iris = [URIRef(f"{base_iri}edge/{u}_{v}_{k}") for u, v, k in edges.index]
        add_features(edges, iris, [fm.Edge] * len(iris), set())
        for iri, (u, v, _) in zip(iris, edges.index, strict=True):
            graph.add((iri, fm["from"], node_iri(u)))
            graph.add((iri, fm["to"], node_iri(v)))
    if nodes is not None:
        iris = [node_iri(n) for n in nodes.index]
        kinds = [
            fm.Region if k == "region" else fm.Node
            for k in nodes.get("kind", ["node"] * len(nodes))
        ]
        add_features(nodes, iris, kinds, {"kind"})
    if membership is not None:
        for orig, coarse in membership.items():
            if orig != coarse:
                graph.add((node_iri(orig), GEO.sfWithin, node_iri(coarse)))
    return graph


def write_geosparql(graph: Graph, path: str | Path) -> Path:
    """Serialise; format from suffix (.ttl, .nt, .jsonld, .rdf/.xml)."""
    path = Path(path)
    fmt = _FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"Unknown RDF suffix {path.suffix!r}; use one of {sorted(_FORMATS)}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=path, format=fmt)
    return path


def _wkts(frame: gpd.GeoDataFrame, wkt_crs: str) -> list[str]:
    if frame.crs is None:
        raise ValueError("Frame needs a CRS for GeoSPARQL export.")
    if wkt_crs == "crs84":
        return list(frame.geometry.to_crs("OGC:CRS84").to_wkt(rounding_precision=8))  # ~1 mm
    if wkt_crs == "native":
        code = frame.crs.to_epsg()
        if code == 4326:  # EPSG:4326 is lat/lon; shapely WKT is lon/lat, i.e. CRS84
            return list(frame.geometry.to_wkt())
        if code is None:
            raise ValueError("Native WKT needs an EPSG-coded CRS; use wkt_crs='crs84'.")
        prefix = f"<http://www.opengis.net/def/crs/EPSG/0/{code}> "
        return [prefix + w for w in frame.geometry.to_wkt()]
    raise ValueError(f"wkt_crs must be 'crs84' or 'native', got {wkt_crs!r}.")


def _local(name: str) -> str:
    """IRI-safe local name for a column."""
    out = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return out if out and not out[0].isdigit() else f"c_{out}"

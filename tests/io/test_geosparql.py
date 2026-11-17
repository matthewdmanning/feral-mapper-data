import pytest

rdflib = pytest.importorskip("rdflib")
from rdflib.namespace import GEO, RDF  # noqa: E402

from feral_mapper_data import coarsen_homogeneous, to_geosparql, write_geosparql  # noqa: E402
from tests.test_coarsen import lattice  # noqa: E402, F401


def test_edges_export_crs84(edges):
    edges = edges.assign(aadt=[100.0, float("nan"), 7.0])
    g = to_geosparql(edges)
    feats = set(g.subjects(RDF.type, GEO.Feature))
    assert len(feats) == 3
    wkts = [str(o) for o in g.objects(None, GEO.asWKT)]
    assert all(
        w.startswith(("LINESTRING (-81", "LINESTRING (-80.9")) for w in wkts
    )  # UTM 17N -> lon/lat, no prefix
    lengths = sorted(float(o) for o in g.objects(None, GEO.hasMetricLength))
    assert lengths == pytest.approx([100.0, 100.0, 100.0])
    fm = rdflib.Namespace("urn:feral-mapper:vocab#")
    assert len(list(g.objects(None, fm.aadt))) == 2  # NaN omitted


def test_native_crs_prefix(edges):
    g = to_geosparql(edges, wkt_crs="native")
    wkt = str(next(g.objects(None, GEO.asWKT)))
    assert wkt.startswith("<http://www.opengis.net/def/crs/EPSG/0/32617> LINESTRING")


def test_regions_and_membership(lattice, tmp_path):  # noqa: F811
    cg = coarsen_homogeneous(lattice, ["lc"])
    g = to_geosparql(cg.edges, cg.nodes, cg.membership)
    fm = rdflib.Namespace("urn:feral-mapper:vocab#")
    regions = set(g.subjects(RDF.type, fm.Region))
    assert len(regions) == (cg.nodes["kind"] == "region").sum()
    assert all(g.value(r, GEO.hasMetricArea) is not None for r in regions)
    assert (
        len(list(g.triples((None, GEO.sfWithin, None))))
        == (cg.membership.index != cg.membership.to_numpy()).sum()
    )
    path = write_geosparql(g, tmp_path / "net.ttl")
    assert len(rdflib.Graph().parse(path)) == len(g)


def test_write_rejects_unknown_suffix(edges, tmp_path):
    with pytest.raises(ValueError):
        write_geosparql(to_geosparql(edges), tmp_path / "x.csv")

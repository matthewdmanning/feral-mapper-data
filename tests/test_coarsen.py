import geopandas as gpd
import numpy as np
import pytest

from feral_mapper_data import coarsen_homogeneous, edge_classes, edges_from_lines
from tests.conftest import CRS, line

SPACING = 100.0


@pytest.fixture
def lattice() -> gpd.GeoDataFrame:
    """8x4 street grid; west half forest (ndvi ~0.8), east half urban (ndvi ~0.2)."""
    lines, cover, ndvi = [], [], []
    for i in range(8):
        for j in range(4):
            x, y = i * SPACING, j * SPACING
            for dx, dy in ((1, 0), (0, 1)):
                if i + dx < 8 and j + dy < 4:
                    lines.append(line((x, y), (x + dx * SPACING, y + dy * SPACING)))
                    west = x + dx * SPACING / 2 < 350
                    cover.append("forest" if west else "urban")
                    ndvi.append((0.81 if west else 0.22) + 0.01 * ((i + j) % 2))
    gdf = gpd.GeoDataFrame({"lc": cover, "ndvi": ndvi}, geometry=lines, crs=CRS)
    return edges_from_lines(gdf)


def test_edge_classes_quantise(lattice):
    exact = edge_classes(lattice, ["ndvi"])
    binned = edge_classes(lattice, ["ndvi"], {"ndvi": 0.1})
    assert len(np.unique(exact)) == 4
    assert len(np.unique(binned)) == 2


def test_edge_classes_validation(lattice):
    with pytest.raises(KeyError):
        edge_classes(lattice, ["lc"], {"ndvi": 0.1})
    with pytest.raises(TypeError):
        edge_classes(lattice, ["lc"], {"lc": 1.0})


def test_homogeneous_halves_collapse(lattice):
    cg = coarsen_homogeneous(lattice, ["lc", "ndvi"], tolerances={"ndvi": 0.1})
    regions = cg.nodes[cg.nodes["kind"] == "region"]
    assert len(regions) == 2
    assert len(cg.nodes) < 32 / 2
    assert set(regions["lc"]) == {"forest", "urban"}
    assert regions["ndvi"].between(0.1, 0.9).all()
    assert (regions["area_m2"] > 0).all()
    # border column x=300..400 keeps its nodes: boundary nodes are never merged
    assert (cg.nodes["kind"] == "node").sum() >= 4
    # every original edge is either absorbed or kept, exactly once
    absorbed = regions["n_internal_edges"].sum()
    assert absorbed + len(cg.edges) == len(lattice)
    assert regions["internal_length_m"].sum() == pytest.approx(absorbed * SPACING)
    # coarse edges reference existing coarse nodes
    ends = set(cg.edges.index.get_level_values("u")) | set(cg.edges.index.get_level_values("v"))
    assert ends <= set(cg.nodes.index)
    assert set(cg.membership.unique()) == set(cg.nodes.index)


def test_strict_classes_keep_alternating_values(lattice):
    cg = coarsen_homogeneous(lattice, ["ndvi"])  # no tolerance: checkerboard never homogeneous
    assert (cg.nodes["kind"] == "region").sum() == 0
    assert len(cg.edges) == len(lattice)


def test_tile_caps_region_extent(lattice):
    whole = coarsen_homogeneous(lattice, ["lc"])
    tiled = coarsen_homogeneous(lattice, ["lc"], tile_m=200)
    n_regions = (tiled.nodes["kind"] == "region").sum()
    assert n_regions > (whole.nodes["kind"] == "region").sum()
    bounds = tiled.nodes[tiled.nodes["kind"] == "region"].geometry.bounds
    assert ((bounds.maxx - bounds.minx) <= 200 + 1e-6).all()


def test_min_region_nodes(lattice):
    cg = coarsen_homogeneous(lattice, ["lc"], min_region_nodes=1000)
    assert (cg.nodes["kind"] == "region").sum() == 0
    with pytest.raises(ValueError):
        coarsen_homogeneous(lattice, ["lc"], min_region_nodes=1)


def test_rejects_geographic_crs(lattice):
    with pytest.raises(ValueError, match="projected"):
        coarsen_homogeneous(lattice.to_crs("EPSG:4326"), ["lc"])


def test_to_networkx(lattice):
    cg = coarsen_homogeneous(lattice, ["lc"])
    g = cg.to_networkx()
    assert g.number_of_nodes() == len(cg.nodes)
    assert g.number_of_edges() == len(cg.edges)
    region = next(n for n, d in g.nodes(data=True) if d["kind"] == "region")
    assert g.nodes[region]["n_members"] >= 3


def test_coarse_ids_are_integers(lattice):
    cg = coarsen_homogeneous(lattice, ["lc"])
    assert cg.nodes.index.dtype == np.int64
    assert cg.membership.dtype == np.int64
    for level in ("u", "v"):
        assert cg.edges.index.get_level_values(level).dtype == np.int64
    regions = cg.nodes.index[cg.nodes["kind"] == "region"]
    # region ids continue after the largest original node id, in first-member order
    assert list(regions) == list(range(cg.membership.index.max() + 1, regions.max() + 1))
    assert not set(regions) & set(cg.membership.index)
    # mixed node/region ids sort and group without TypeError
    assert cg.membership.groupby(cg.membership).size().sum() == len(cg.membership)


def test_rejects_non_integer_node_ids(lattice):
    relabelled = lattice.rename(index=lambda n: f"n{n}", level="u").rename(
        index=lambda n: f"n{n}", level="v"
    )
    with pytest.raises(TypeError, match="integer node ids"):
        coarsen_homogeneous(relabelled, ["lc"])

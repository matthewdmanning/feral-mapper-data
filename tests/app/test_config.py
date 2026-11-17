"""PipelineConfig.from_toml rejects malformed configs and resolves paths relative to the file."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from feral_mapper_data.app.config import ConfigError, PipelineConfig

EXAMPLE = Path(__file__).parents[2] / "examples" / "pipeline.toml"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("toml", "match"),
    [
        ('[network]\npath="a"\n', "output"),
        ('[network]\npath="a"\nbogus=1\n[output]\nedges="e"\n', "Extra inputs are not permitted"),
        ('[network]\npath="a"\n[output]\nedges="e"\n[traffic]\nbuffer_m=5\n', "segments"),
        (
            '[network]\npath="a"\n[output]\nedges="e"\n[polygons]\npath="p"\nprefix="x"\n',
            "valid list",
        ),
    ],
)
def test_invalid_config_raises_config_error(tmp_path, toml, match):
    path = tmp_path / "c.toml"
    path.write_text(textwrap.dedent(toml))
    with pytest.raises(ConfigError, match=match):
        PipelineConfig.from_toml(path)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ('[coarsen]\ncolumns=["a"]\n', "coarse_nodes"),
        ('[output]\nedges="e"\ncoarse_nodes="n"\n', "coarsen"),
        ('[output]\nedges="e"\ngeosparql_crs="utm"\n', "geosparql_crs"),
    ],
)
def test_coarsen_and_geosparql_options_are_cross_validated(tmp_path, extra, match):
    base = '[network]\npath="a"\n'
    toml = base + (extra if extra.startswith("[output]") else '[output]\nedges="e"\n' + extra)
    path = tmp_path / "c.toml"
    path.write_text(toml)
    with pytest.raises(ConfigError, match=match):
        PipelineConfig.from_toml(path)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_paths_resolve_relative_to_config_file(tmp_path):
    path = tmp_path / "sub" / "c.toml"
    path.parent.mkdir()
    path.write_text('[network]\npath="net.gpkg"\n[output]\nedges="o.parquet"\n')
    cfg = PipelineConfig.from_toml(path)
    assert cfg.network.path == tmp_path / "sub" / "net.gpkg"


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------

_BASE = '[network]\npath="a"\n[output]\nedges="e"\n'


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ('[roadway]\npath="r"\ncolumns=["c"]\nbuffer_m="wide"\n', "buffer_m"),
        ('[roadway]\npath="r"\ncolumns=["c"]\naggregations={c="median"}\n', "aggregations"),
        ('[[vegetation]]\npath="v"\nstats=["mode"]\n', "stats"),
        ('[[vegetation]]\npath="v"\nvalid_range=[1.0]\n', "valid_range"),
    ],
)
def test_wrongly_typed_values_are_rejected(tmp_path, extra, match):
    path = tmp_path / "c.toml"
    path.write_text(_BASE + extra)
    with pytest.raises(ConfigError, match=match):
        PipelineConfig.from_toml(path)


def test_example_config_loads():
    cfg = PipelineConfig.from_toml(EXAMPLE)
    assert cfg.vegetation[0].valid_range == (-1.0, 1.0)
    assert cfg.output.edges == EXAMPLE.parent / "../data/processed/edges.parquet"

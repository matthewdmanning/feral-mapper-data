"""ensure_projected returns a projected CRS or raises CRSError."""

from __future__ import annotations

import pytest

from feral_mapper_data.geo.crs import CRSError, ensure_projected


def test_ensure_projected(network_lines):
    geographic = network_lines.to_crs("EPSG:4326")
    assert ensure_projected(geographic).crs.is_projected
    assert ensure_projected(network_lines).crs == network_lines.crs
    assert ensure_projected(network_lines, "EPSG:3857").crs.to_epsg() == 3857
    with pytest.raises(CRSError):
        ensure_projected(network_lines, "EPSG:4326")
    with pytest.raises(CRSError):
        ensure_projected(network_lines.set_crs(None, allow_override=True))

import numpy as np

from nugundam.angular.models import AngularAutoConfig
from nugundam.angular.prepare import _resolve_autogrid_mode, best_skgrid_2d


def test_autogrid_mode_resolution_aliases():
    assert _resolve_autogrid_mode(False) == "manual"
    assert _resolve_autogrid_mode(True) == "legacy"
    assert _resolve_autogrid_mode("legacy") == "legacy"
    assert _resolve_autogrid_mode("adaptive") == "adaptive"


def test_legacy_autogrid_mode_is_available():
    ra = np.linspace(5.0, 35.0, 1000)
    h1, h2, dens, info = best_skgrid_2d(1000, ra, mode="legacy")
    assert h1 > 0
    assert h2 > 0
    assert dens > 0
    assert info["mode"] == "legacy"


def test_adaptive_autogrid_uses_geometry_metadata():
    data_ra = np.linspace(10.0, 50.0, 1000)
    data_dec = np.linspace(-10.0, 10.0, 1000)
    rand_ra = np.concatenate([np.linspace(10.0, 20.0, 4000), np.linspace(40.0, 50.0, 4000)])
    rand_dec = np.concatenate([np.linspace(-10.0, -5.0, 4000), np.linspace(5.0, 10.0, 4000)])

    h1, h2, dens, info = best_skgrid_2d(
        len(data_ra),
        data_ra,
        data_dec,
        mode="adaptive",
        sample_ras=rand_ra,
        sample_decs=rand_dec,
        coarse_bins=32,
    )
    assert h1 > 0
    assert h2 > 0
    assert dens > 0
    assert info["mode"] == "adaptive"
    assert info["occupied_cells"] > 0
    assert info["total_cells"] >= info["occupied_cells"]
    assert 0.0 < info["fill_frac"] <= 1.0


def test_config_describe_mentions_new_autogrid_modes():
    text = str(AngularAutoConfig.describe(recursive=True))
    assert "'legacy'" in text
    assert "'adaptive'" in text
    assert "coarse_bins" in text

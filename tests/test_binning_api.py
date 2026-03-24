import numpy as np
import pytest

from nugundam.angular.models import AngularAutoConfig, AngularBinning, ConfigDescription
from nugundam.projected.models import ProjectedAutoConfig, ProjectedBinning


def test_angular_binning_requires_named_constructors():
    with pytest.raises(TypeError):
        AngularBinning()


def test_angular_binning_from_binsize_and_limits_match():
    by_size = AngularBinning.from_binsize(nsep=24, sepmin=30.0 / 3600.0, dsep=0.1, logsep=True)
    by_limits = AngularBinning.from_limits(
        nsep=24,
        sepmin=30.0 / 3600.0,
        sepmax=by_size.sepmax,
        logsep=True,
    )
    assert np.allclose(by_size.edges, by_limits.edges)
    assert np.allclose(by_size.centers, by_limits.centers)
    assert np.allclose(by_size.widths, by_limits.widths)
    assert by_size.sepmax == pytest.approx(by_limits.sepmax)


def test_angular_binning_print_and_table_are_informative():
    binning = AngularBinning.from_limits(nsep=4, sepmin=0.01, sepmax=1.0, logsep=True)
    text = str(binning)
    assert "AngularBinning(log, 4 bins" in text
    assert "sepmin=0.01" in text
    assert "sepmax=1" in text

    table = binning.table()
    assert isinstance(table, ConfigDescription)
    table_text = str(table)
    assert "left" in table_text
    assert "right" in table_text
    assert "center" in table_text
    assert "width" in table_text


def test_angular_binning_describe_mentions_new_constructors_and_helpers():
    text = str(AngularBinning.describe())
    assert "from_binsize(nsep=36, sepmin=0.01, dsep=0.1, logsep=True)" in text
    assert "from_limits(nsep=36, sepmin=0.01, sepmax=10.0, logsep=True)" in text
    assert "edges, centers, widths, sepmax" in text
    assert "table() -> plain-text table of the resolved bins" in text


def test_angular_auto_config_default_binning_uses_named_constructor_defaults():
    cfg = AngularAutoConfig()
    assert isinstance(cfg.binning, AngularBinning)
    assert cfg.binning.nsep == 36
    assert cfg.binning.sepmin == pytest.approx(0.01)
    assert cfg.binning.dsep == pytest.approx(0.1)
    assert cfg.binning.logsep is True


def test_projected_binning_requires_named_constructors():
    with pytest.raises(TypeError):
        ProjectedBinning()


def test_projected_binning_from_limits_exposes_rp_and_pi_bins():
    binning = ProjectedBinning.from_limits(
        nsepp=5,
        seppmin=0.1,
        seppmax=10.0,
        logsepp=True,
        nsepv=4,
        dsepv=5.0,
    )
    assert binning.seppmax == pytest.approx(10.0)
    assert binning.sepvmax == pytest.approx(20.0)
    assert len(binning.rp_edges) == 6
    assert len(binning.pi_edges) == 5
    assert "ProjectedBinning(rp: log, 5 bins" in str(binning)
    assert "sepvmax=20" in str(binning)

    rp_table = str(binning.table("rp"))
    pi_table = str(binning.table("pi"))
    assert "left" in rp_table and "width" in rp_table
    assert "left" in pi_table and "width" in pi_table


def test_projected_binning_describe_mentions_new_constructors_and_helpers():
    text = str(ProjectedBinning.describe())
    assert "from_binsize(nsepp=20, seppmin=0.1, dsepp=0.1, logsepp=True, nsepv=20, dsepv=2.0)" in text
    assert "from_limits(nsepp=20, seppmin=0.1, seppmax=10.0, logsepp=True, nsepv=20, dsepv=2.0)" in text
    assert "rp_edges, rp_centers, rp_widths, seppmax, pi_edges, pi_centers, pi_widths, sepvmax" in text
    assert "table(axis='rp' or 'pi') -> plain-text table of the resolved bins" in text


def test_projected_auto_config_default_binning_uses_named_constructor_defaults():
    cfg = ProjectedAutoConfig()
    assert isinstance(cfg.binning, ProjectedBinning)
    assert cfg.binning.nsepp == 20
    assert cfg.binning.seppmin == pytest.approx(0.1)
    assert cfg.binning.dsepp == pytest.approx(0.1)
    assert cfg.binning.logsepp is True
    assert cfg.binning.nsepv == 20
    assert cfg.binning.dsepv == pytest.approx(2.0)

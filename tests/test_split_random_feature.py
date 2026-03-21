import numpy as np
import pytest
from types import SimpleNamespace

from nugundam.angular import api as angular_api
from nugundam.angular import fortran_bridge as angular_fb
from nugundam.angular.models import AngularAutoConfig, SplitRandomSpec
from nugundam.projected import api as projected_api
from nugundam.projected.models import ProjectedAutoConfig


def _fake_ang_sample(n):
    return SimpleNamespace(table=None, nrows=int(n), wunit=True)


def _fake_proj_sample(n):
    return SimpleNamespace(table=None, nrows=int(n), wunit=True)


def test_split_rr_builder_angular_match_data(theta_grid, monkeypatch):
    edges, centers = theta_grid
    subset_calls = []
    cntids = []

    def fake_subset(sample, idx, *, regrid=False, theta_edges=None):
        subset_calls.append((regrid, None if theta_edges is None else len(theta_edges)))
        return _fake_ang_sample(len(np.asarray(idx)))

    def fake_auto(sample, *, theta_edges, nthreads, weight_mode, doboot, dojk=False, nreg=0, nbts=0, bseed=0, cntid, progress_file=None):
        cntids.append(cntid)
        size = float(sample.nrows)
        return np.array([size, 2.0 * size]), np.zeros((len(theta_edges) - 1, 0)), None, None, None

    def fake_cross(left, right, *, theta_edges, nthreads, weight_mode, doboot, dojk=False, nreg=0, nbts=0, bseed=0, cntid, progress_file=None):
        return np.array([99.0, 199.0]), np.zeros((len(theta_edges) - 1, 0)), None

    monkeypatch.setattr(angular_fb, "subset_prepared_angular_sample", fake_subset)
    monkeypatch.setattr(angular_fb, "run_theta_auto_counts", fake_auto)
    monkeypatch.setattr(angular_fb, "run_theta_cross_counts", fake_cross)

    counts = angular_fb.build_auto_counts(
        _fake_ang_sample(4),
        _fake_ang_sample(10),
        theta_edges=edges,
        theta_centers=centers,
        nthreads=1,
        estimator="LS",
        weight_mode="unweighted",
        doboot=False,
        dojk=False,
        nreg=0,
        nbts=0,
        bseed=7,
        progress_file=None,
        split_random=SplitRandomSpec(enabled=True, mode="match_data", seed=11),
    )

    np.testing.assert_allclose(counts.dd, [4.0, 8.0])
    np.testing.assert_allclose(counts.rr, [10.0, 20.0])
    np.testing.assert_allclose(counts.dr, [99.0, 199.0])
    assert counts.metadata["rr_norm_pairs"] == 12.0
    assert counts.metadata["split_random_enabled"] is True
    assert counts.metadata["split_random_nchunks"] == 3
    assert sorted(counts.metadata["split_random_chunk_sizes"]) == [3, 3, 4]
    assert subset_calls == [(True, len(edges)), (True, len(edges)), (True, len(edges))]
    assert cntids == ["DD", "RR split 1/3", "RR split 2/3", "RR split 3/3"]


def test_acf_split_random_requires_ls():
    cfg = AngularAutoConfig(estimator="NAT", split_random=SplitRandomSpec(enabled=True))
    with pytest.raises(NotImplementedError):
        angular_api.acf(object(), object(), cfg)


def test_pcf_split_random_disallows_jackknife():
    cfg = ProjectedAutoConfig(
        estimator="LS",
        split_random=SplitRandomSpec(enabled=True),
    )
    cfg.jackknife.enabled = True
    with pytest.raises(NotImplementedError):
        projected_api.pcf(object(), object(), cfg)


def test_subset_prepared_angular_sample_regrids_smaller_chunks(monkeypatch):
    from nugundam.angular.models import PreparedAngularSample
    from nugundam.angular.prepare import subset_prepared_angular_sample
    import nugundam.cflibfor as cff

    cff.mod.skll2d = lambda mxh1, mxh2, npt, ra, dec, sbound: (np.zeros((mxh2, mxh1), dtype=np.int32), np.zeros(int(npt), dtype=np.int32))

    n = 4000
    ra = np.linspace(0.0, 359.9, n, dtype=np.float64)
    dec = np.linspace(-30.0, 30.0, n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float32)
    sample = PreparedAngularSample(
        table=np.arange(n),
        ra=ra, dec=dec, weights=weights,
        x=np.zeros(n), y=np.zeros(n), z=np.ones(n),
        sk=np.zeros((60, 60), dtype=np.int32), ll=np.zeros(n, dtype=np.int32), wunit=True,
        sbound=(0.0, 360.0, -31.0, 31.0), mxh1=60, mxh2=60,
        grid_meta={"autogrid_mode": "legacy", "dens": None, "pxorder": "natural"},
    )
    chunk = subset_prepared_angular_sample(sample, np.arange(500), regrid=True, theta_edges=np.array([0.01, 0.02]))
    assert chunk.mxh1 < sample.mxh1


def test_subset_prepared_projected_sample_regrids_smaller_chunks():
    from nugundam.projected.models import PreparedProjectedSample
    from nugundam.projected.prepare import subset_prepared_projected_sample

    n = 4000
    ra = np.linspace(0.0, 359.9, n, dtype=np.float64)
    dec = np.linspace(-30.0, 30.0, n, dtype=np.float64)
    dist = np.linspace(100.0, 250.0, n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float32)
    pi_edges = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    sample = PreparedProjectedSample(
        table=np.arange(n),
        ra=ra, dec=dec, dist=dist, weights=weights,
        x=np.zeros(n), y=np.zeros(n), z=np.ones(n),
        sk=np.zeros((7, 30, 30), dtype=np.int32), ll=np.zeros(n, dtype=np.int32), wunit=True,
        sbound=(0.0, 360.0, -31.0, 31.0, 99.0, 251.0), mxh1=30, mxh2=30, mxh3=7,
        grid_meta={"autogrid": True, "dens": None, "pxorder": "natural", "nsepv": 2, "dsepv": 10.0},
    )
    chunk = subset_prepared_projected_sample(sample, np.arange(500), pi_edges=pi_edges, regrid=True)
    assert chunk.mxh1 < sample.mxh1


def test_split_random_docstrings_present():
    from nugundam.angular.api import acf
    from nugundam.projected.api import pcf
    from nugundam.angular.models import SplitRandomSpec

    assert "split-random" in (acf.__doc__ or "").lower()
    assert "split-random" in (pcf.__doc__ or "").lower()
    assert "landy-szalay" in (SplitRandomSpec.__doc__ or "").lower()

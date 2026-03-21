import numpy as np

from nugundam import pcf, proj_auto_counts
from nugundam.projected import api as projected_api
from nugundam.projected.models import (
    PreparedProjectedSample,
    ProjectedAutoConfig,
    ProjectedAutoCountsConfig,
    ProjectedAutoCountsResult,
    ProjectedAutoCounts,
    ProjectedCorrelationResult,
)


def _prepared(weighted=False):
    w = np.array([1.0, 2.0], dtype=np.float32) if weighted else np.ones(2, dtype=np.float32)
    return PreparedProjectedSample(
        table=[0, 1],
        ra=np.array([0.0, 1.0]),
        dec=np.array([0.0, 1.0]),
        dist=np.array([100.0, 101.0]),
        weights=w,
        x=np.array([1.0, 1.0]),
        y=np.array([0.0, 0.0]),
        z=np.array([0.0, 0.0]),
        sk=np.zeros((1, 1, 1), dtype=int),
        ll=np.zeros(2, dtype=int),
        wunit=not weighted,
        sbound=(0.0, 360.0, -1.0, 1.0, 99.0, 102.0),
        mxh1=1,
        mxh2=1,
        mxh3=1,
    )


def test_pcf_wrapper(monkeypatch, rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    counts = ProjectedAutoCounts(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        pi_edges=pi_edges,
        pi_centers=pi_centers,
        dd=np.array([[3.0, 1.0], [2.0, 1.0]]),
        rr=np.array([[3.0, 1.0], [2.0, 1.0]]),
        dr=np.array([[3.0, 1.0], [2.0, 1.0]]),
        metadata={"n_data": 4, "n_random": 4},
    )
    monkeypatch.setattr(projected_api, 'prepare_projected_auto', lambda d, r, c: (_prepared(), _prepared(), {"rp_edges": rp_edges, "rp_centers": rp_centers, "pi_edges": pi_edges, "pi_centers": pi_centers, "pi_delta": np.array([1.0, 1.0])}))
    monkeypatch.setattr(projected_api, 'build_auto_counts', lambda *a, **k: counts)
    result = pcf(object(), object(), ProjectedAutoConfig(estimator='NAT'))
    assert isinstance(result, ProjectedCorrelationResult)
    assert result.metadata['config']['estimator'] == 'NAT'
    assert result.counts.metadata['provenance']['run_kind'] == 'pcf'


def test_proj_auto_counts_wrapper(monkeypatch, rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    counts = ProjectedAutoCountsResult(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        pi_edges=pi_edges,
        pi_centers=pi_centers,
        dd=np.array([[1.0, 2.0], [3.0, 4.0]]),
        intpi_dd=np.array([6.0, 14.0]),
        metadata={"n_data": 2},
    )
    monkeypatch.setattr(projected_api, 'prepare_projected_auto', lambda d, r, c: (_prepared(), _prepared(), {"rp_edges": rp_edges, "rp_centers": rp_centers, "pi_edges": pi_edges, "pi_centers": pi_centers, "pi_delta": np.array([1.0, 1.0])}))
    monkeypatch.setattr(projected_api, 'build_auto_count_result', lambda *a, **k: counts)
    out = proj_auto_counts(object(), ProjectedAutoCountsConfig())
    assert isinstance(out, ProjectedAutoCountsResult)
    assert out.metadata['config']['columns']['ra'] == 'ra'
    assert out.metadata['provenance']['run_kind'] == 'proj_auto_counts'

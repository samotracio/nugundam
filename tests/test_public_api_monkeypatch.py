import numpy as np
from nugundam import acf, accf, ang_auto_counts, pccf
from nugundam.angular import api as angular_api
from nugundam.angular.models import (
    AngularAutoConfig,
    AngularAutoCountsConfig,
    AngularAutoCounts,
    AngularAutoCountsResult,
    AngularCorrelationResult,
    AngularCrossConfig,
    AngularCrossCounts,
    PreparedAngularSample,
)

def _prepared(weighted=False):
    w = np.array([1.0, 2.0]) if weighted else np.ones(2)
    return PreparedAngularSample(table=[0,1], ra=np.array([0.0,1.0]), dec=np.array([0.0,1.0]), weights=w, x=np.array([0.0,0.0]), y=np.array([0.0,0.0]), z=np.array([1.0,1.0]), sk=np.array([0,0]), ll=np.array([0,1]), wunit=not weighted, sbound=(0.0,1.0,0.0,1.0), mxh1=2, mxh2=2)

def test_acf_wrapper(monkeypatch, theta_grid):
    edges, centers = theta_grid
    counts = AngularAutoCounts(theta_edges=edges, theta_centers=centers, dd=np.array([3.0,1.0]), rr=np.array([3.0,1.0]), dr=np.array([4.0,2.0]), metadata={"n_data":4, "n_random":4})
    monkeypatch.setattr(angular_api, 'prepare_angular_auto', lambda d, r, c: (_prepared(), _prepared(), {"theta_edges": edges, "theta_centers": centers}))
    monkeypatch.setattr(angular_api, 'build_auto_counts', lambda *a, **k: counts)
    result = acf(object(), object(), AngularAutoConfig(estimator='NAT'))
    assert isinstance(result, AngularCorrelationResult)
    assert result.metadata['config']['estimator'] == 'NAT'
    assert result.counts.metadata['provenance']['run_kind'] == 'acf'

def test_ang_auto_counts_wrapper(monkeypatch, theta_grid):
    edges, centers = theta_grid
    counts = AngularAutoCountsResult(theta_edges=edges, theta_centers=centers, dd=np.array([1.0,2.0]), metadata={"n_data":2})
    monkeypatch.setattr(angular_api, 'prepare_angular_auto', lambda d, r, c: (_prepared(), _prepared(), {"theta_edges": edges, "theta_centers": centers}))
    monkeypatch.setattr(angular_api, 'build_auto_count_result', lambda *a, **k: counts)
    out = ang_auto_counts(object(), AngularAutoCountsConfig())
    assert isinstance(out, AngularAutoCountsResult)
    assert out.metadata['config']['columns']['ra'] == 'ra'
    assert out.metadata['provenance']['run_kind'] == 'ang_auto_counts'


from nugundam.projected import api as projected_api
from nugundam.projected.models import ProjectedCrossConfig, ProjectedCrossCounts, ProjectedCorrelationResult, PreparedProjectedSample


def _prepared_projected(weighted=False):
    w = np.array([1.0, 2.0], dtype=np.float32) if weighted else np.ones(2, dtype=np.float32)
    return PreparedProjectedSample(table=[0,1], ra=np.array([0.0,1.0]), dec=np.array([0.0,1.0]), dist=np.array([100.0,101.0]), weights=w, x=np.array([0.0,0.0]), y=np.array([0.0,0.0]), z=np.array([1.0,1.0]), sk=np.zeros((1,1,1), dtype=int), ll=np.array([0,1]), wunit=not weighted, sbound=(0.0,360.0,-1.0,1.0,99.0,102.0), mxh1=1, mxh2=1, mxh3=1)


def test_accf_wrapper_new_signature(monkeypatch, theta_grid):
    edges, centers = theta_grid
    counts = AngularCrossCounts(theta_edges=edges, theta_centers=centers, d1d2=np.array([3.0,1.0]), d1r2=np.array([4.0,2.0]), metadata={"n_data1":4, "n_random1":0, "n_data2":4, "n_random2":4, "primary":"data1"})
    monkeypatch.setattr(angular_api, 'prepare_angular_cross', lambda d1, r1, d2, r2, c: (_prepared(), None, _prepared(), _prepared(), {"theta_edges": edges, "theta_centers": centers}))
    monkeypatch.setattr(angular_api, 'build_cross_counts', lambda *a, **k: counts)
    result = accf(object(), object(), AngularCrossConfig(estimator='DP'), random2=object())
    assert isinstance(result, AngularCorrelationResult)
    assert result.metadata['config']['estimator'] == 'DP'
    assert result.counts.metadata['provenance']['run_kind'] == 'accf'


def test_pccf_wrapper_new_signature(monkeypatch, rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    counts = ProjectedCrossCounts(rp_edges=rp_edges, rp_centers=rp_centers, pi_edges=pi_edges, pi_centers=pi_centers, d1d2=np.array([[3.0, 1.0],[2.0, 1.0]]), d1r2=np.array([[4.0,2.0],[3.0,2.0]]), metadata={"n_data1":4, "n_random1":0, "n_data2":4, "n_random2":4, "primary":"data1"})
    monkeypatch.setattr(projected_api, 'prepare_projected_cross', lambda d1, r1, d2, r2, c: (_prepared_projected(), None, _prepared_projected(), _prepared_projected(), {"rp_edges": rp_edges, "rp_centers": rp_centers, "pi_edges": pi_edges, "pi_centers": pi_centers, "pi_delta": np.array([1.0,1.0])}))
    monkeypatch.setattr(projected_api, 'build_cross_counts', lambda *a, **k: counts)
    result = pccf(object(), object(), ProjectedCrossConfig(estimator='DP'), random2=object())
    assert isinstance(result, ProjectedCorrelationResult)
    assert result.metadata['config']['estimator'] == 'DP'
    assert result.counts.metadata['provenance']['run_kind'] == 'pccf'

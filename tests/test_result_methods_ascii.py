import numpy as np

from nugundam import save_result
from nugundam.angular import api as angular_api
from nugundam.angular.models import (
    AngularAutoConfig,
    AngularCrossCountsConfig,
    AngularAutoCounts,
    AngularCorrelationResult,
    AngularCrossConfig,
    AngularCrossCounts,
    AngularCrossCountsResult,
    PreparedAngularSample,
)


def _prepared(weighted=False):
    w = np.array([1.0, 2.0]) if weighted else np.ones(2)
    return PreparedAngularSample(
        table=[0, 1],
        ra=np.array([0.0, 1.0]),
        dec=np.array([0.0, 1.0]),
        weights=w,
        x=np.array([0.0, 0.0]),
        y=np.array([0.0, 0.0]),
        z=np.array([1.0, 1.0]),
        sk=np.array([0, 0]),
        ll=np.array([0, 1]),
        wunit=not weighted,
        sbound=(0.0, 1.0, 0.0, 1.0),
        mxh1=2,
        mxh2=2,
    )


def test_result_instance_save_and_class_read(theta_grid, tmp_path):
    edges, centers = theta_grid
    counts = AngularAutoCounts(
        theta_edges=edges,
        theta_centers=centers,
        dd=np.array([1.0, 2.0]),
        rr=np.array([3.0, 4.0]),
        dr=np.array([5.0, 6.0]),
        metadata={"config": {"estimator": "LS"}, "n_data": 2, "n_random": 3},
    )
    result = AngularCorrelationResult(
        theta_edges=edges,
        theta_centers=centers,
        wtheta=np.array([0.1, 0.2]),
        wtheta_err=np.array([0.01, 0.02]),
        estimator="LS",
        counts=counts,
        metadata={"config": {"estimator": "LS"}, "provenance": {"run_kind": "acf"}},
    )
    path = tmp_path / "acf_method.gres"
    result.save(path)
    loaded = AngularCorrelationResult.read_result(path)
    np.testing.assert_allclose(loaded.wtheta, result.wtheta)
    np.testing.assert_allclose(loaded.counts.dr, result.counts.dr)
    assert loaded.metadata["provenance"]["run_kind"] == "acf"


def test_result_to_ascii_default_and_custom(theta_grid, tmp_path):
    edges, centers = theta_grid
    counts = AngularAutoCounts(
        theta_edges=edges,
        theta_centers=centers,
        dd=np.array([1.0, 2.0]),
        rr=np.array([3.0, 4.0]),
        dr=np.array([5.0, 6.0]),
        metadata={"config": {"estimator": "LS"}, "n_data": 2, "n_random": 3},
    )
    result = AngularCorrelationResult(
        theta_edges=edges,
        theta_centers=centers,
        wtheta=np.array([0.1, 0.2]),
        wtheta_err=np.array([0.01, 0.02]),
        estimator="LS",
        counts=counts,
        metadata={"config": {"estimator": "LS"}},
    )

    default_path = tmp_path / "acf_default.txt"
    result.to_ascii(default_path)
    header = default_path.read_text().splitlines()[0]
    assert "theta_centers dd rr dr wtheta wtheta_err" in header
    arr = np.loadtxt(default_path)
    assert arr.shape == (2, 6)

    custom_path = tmp_path / "acf_custom.txt"
    result.to_ascii(custom_path, cols=["theta_centers", "dr", "wtheta"])
    custom_header = custom_path.read_text().splitlines()[0]
    assert "theta_centers dr wtheta" in custom_header
    custom_arr = np.loadtxt(custom_path)
    assert custom_arr.shape == (2, 3)


def test_save_result_and_method_support_cross_outputs(monkeypatch, theta_grid, tmp_path):
    edges, centers = theta_grid
    cross_counts = AngularCrossCountsResult(
        theta_edges=edges,
        theta_centers=centers,
        d1d2=np.array([8.0, 4.0]),
        d1d2_boot=np.array([[8.0, 7.0], [4.0, 3.5]]),
        metadata={"n_data1": 4, "n_data2": 2},
    )
    monkeypatch.setattr(
        angular_api,
        "prepare_angular_cross",
        lambda d1, r1, d2, r2, c: (_prepared(), _prepared(), _prepared(), _prepared(), {"theta_edges": edges, "theta_centers": centers}),
    )
    estimator_counts = AngularCrossCounts(
        theta_edges=edges,
        theta_centers=centers,
        d1d2=np.array([8.0, 4.0]),
        d1r2=np.array([4.0, 2.0]),
        r1d2=np.array([4.0, 1.0]),
        r1r2=np.array([2.0, 1.0]),
        metadata={"n_data1": 4, "n_random1": 2, "n_data2": 2, "n_random2": 2},
    )
    monkeypatch.setattr(angular_api, "build_cross_count_result", lambda *a, **k: cross_counts)
    monkeypatch.setattr(angular_api, "build_cross_counts", lambda *a, **k: estimator_counts)

    counts_out = angular_api.ang_cross_counts(object(), object(), AngularCrossCountsConfig())
    counts_path = tmp_path / "cross_counts.gres"
    save_result(counts_out, counts_path)
    loaded_counts = AngularCrossCountsResult.read_result(counts_path)
    np.testing.assert_allclose(loaded_counts.d1d2, counts_out.d1d2)
    assert loaded_counts.metadata["provenance"]["run_kind"] == "ang_cross_counts"

    result_out = angular_api.accf(object(), object(), AngularCrossConfig(estimator="LS"), random1=object(), random2=object())
    result_path = tmp_path / "cross_result.gres"
    result_out.save_result(result_path)
    loaded_result = AngularCorrelationResult.read_result(result_path)
    np.testing.assert_allclose(loaded_result.counts.r1d2, result_out.counts.r1d2)
    assert loaded_result.estimator == "LS"

    ascii_path = tmp_path / "cross_result.txt"
    result_out.to_ascii(ascii_path)
    header = ascii_path.read_text().splitlines()[0]
    assert "theta_centers d1d2 d1r2 r1d2 r1r2 wtheta wtheta_err" in header

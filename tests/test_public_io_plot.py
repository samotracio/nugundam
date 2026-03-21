import numpy as np
from nugundam import read_result, save_result, write_result, plot_result
from nugundam.angular.models import AngularAutoCounts, AngularCorrelationResult


def test_save_read_counts_roundtrip(tmp_path, theta_grid):
    edges, centers = theta_grid
    counts = AngularAutoCounts(
        theta_edges=edges,
        theta_centers=centers,
        dd=np.array([1.0, 2.0]),
        rr=np.array([1.0, 2.0]),
        dr=np.array([1.0, 2.0]),
        metadata={"n_data": 3, "n_random": 3, "theta_delta": np.array([0.1, 0.2]), "config": {"estimator": "LS"}},
    )
    path = tmp_path / 'counts.gres'
    save_result(counts, path)
    loaded = read_result(path)
    np.testing.assert_allclose(loaded.dd, counts.dd)
    np.testing.assert_allclose(loaded.metadata["theta_delta"], counts.metadata["theta_delta"])
    assert loaded.metadata["config"]["estimator"] == "LS"


def test_write_read_result_roundtrip(tmp_path, theta_grid):
    edges, centers = theta_grid
    counts = AngularAutoCounts(
        theta_edges=edges,
        theta_centers=centers,
        dd=np.array([1.0, 2.0]),
        rr=np.array([1.0, 2.0]),
        dr=np.array([1.0, 2.0]),
        dd_boot=np.array([[1.0, 2.0], [3.0, 4.0]]),
        metadata={
            "n_data": 3,
            "n_random": 3,
            "theta_delta": np.array([0.1, 0.2]),
            "config": {"binning": {"nsep": 2}},
            "provenance": {"run_kind": "acf"},
        },
    )
    result = AngularCorrelationResult(
        theta_edges=edges,
        theta_centers=centers,
        wtheta=np.array([0.1, 0.2]),
        wtheta_err=np.array([0.01, 0.02]),
        estimator='NAT',
        counts=counts,
        metadata={
            "weighted": False,
            "theta_delta": np.array([0.1, 0.2]),
            "config": {"estimator": "NAT"},
            "provenance": {"run_kind": "acf"},
        },
    )
    path = tmp_path / 'acf_result.gres'
    write_result(result, path)
    loaded = read_result(path)
    assert isinstance(loaded, AngularCorrelationResult)
    np.testing.assert_allclose(loaded.wtheta, result.wtheta)
    np.testing.assert_allclose(loaded.wtheta_err, result.wtheta_err)
    np.testing.assert_allclose(loaded.counts.dd_boot, result.counts.dd_boot)
    np.testing.assert_allclose(loaded.metadata["theta_delta"], result.metadata["theta_delta"])
    assert loaded.estimator == result.estimator
    assert loaded.metadata["config"]["estimator"] == "NAT"
    assert loaded.counts.metadata["provenance"]["run_kind"] == "acf"


def test_ascii_writers_and_plot(tmp_path, theta_grid):
    edges, centers = theta_grid
    counts = AngularAutoCounts(theta_edges=edges, theta_centers=centers, dd=np.array([1.0,2.0]), rr=np.array([1.0,2.0]), dr=np.array([1.0,2.0]), metadata={"n_data":3, "n_random":3})
    result = AngularCorrelationResult(theta_edges=edges, theta_centers=centers, wtheta=np.array([0.1,0.2]), wtheta_err=np.array([0.01,0.02]), estimator='NAT', counts=counts, metadata={})
    cf_path = tmp_path / 'cf.txt'
    ct_path = tmp_path / 'counts.txt'
    result.to_ascii(cf_path)
    counts.to_ascii(ct_path)
    assert cf_path.exists() and ct_path.exists()
    assert plot_result(result) is not None

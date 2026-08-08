import numpy as np

from nugundam.io import read_result, save_result
from nugundam.projected.estimators import (
    apply_bootstrap_storage_policy,
    estimate_auto,
)
from nugundam.projected.models import BootstrapSpec, ProjectedAutoCounts


def _counts():
    return ProjectedAutoCounts(
        rp_edges=np.array([0.1, 1.0], dtype=np.float64),
        rp_centers=np.array([0.31622777], dtype=np.float64),
        pi_edges=np.array([0.0, 2.0, 5.0], dtype=np.float64),
        pi_centers=np.array([1.0, 3.5], dtype=np.float64),
        dd=np.array([[3.0, 4.0]], dtype=np.float64),
        rr=np.array([[5.0, 5.0]], dtype=np.float64),
        dr=None,
        dd_boot=np.array(
            [
                [
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                ]
            ],
            dtype=np.float64,
        ),
        metadata={
            "n_data": 3,
            "n_random": 4,
            "rr_norm_pairs": 6.0,
        },
    )


def test_estimate_auto_stores_cumulative_bootstrap_realizations():
    result = estimate_auto(_counts(), estimator="NAT")

    assert result.bootstrap_realizations.shape == (3, 1)
    assert result.bootstrap_cumulative_realizations.shape == (3, 1, 2)
    np.testing.assert_allclose(
        result.bootstrap_cumulative_realizations[:, :, -1],
        result.bootstrap_realizations,
    )
    np.testing.assert_allclose(
        np.std(result.bootstrap_realizations, axis=0),
        result.wp_err,
    )



def test_estimator_can_skip_cumulative_bootstrap_allocation():
    result = estimate_auto(
        _counts(),
        estimator="NAT",
        store_bootstrap_cumulative=False,
    )

    assert result.bootstrap_realizations.shape == (3, 1)
    assert result.bootstrap_cumulative_realizations is None
    np.testing.assert_allclose(
        np.std(result.bootstrap_realizations, axis=0),
        result.wp_err,
    )

def test_bootstrap_storage_policy_can_drop_counts_or_cumulative():
    result = estimate_auto(_counts(), estimator="NAT")
    result.bootstrap_counts = _counts()

    spec = BootstrapSpec(
        enabled=True,
        store_counts=False,
        store_cumulative=False,
    )
    result = apply_bootstrap_storage_policy(result, spec)

    assert result.counts.dd_boot is None
    assert result.bootstrap_counts is None
    assert result.bootstrap_cumulative_realizations is None
    assert result.bootstrap_realizations is not None
    assert result.metadata["bootstrap_store_counts"] is False
    assert result.metadata["bootstrap_store_cumulative"] is False


def test_bootstrap_counts_and_cumulative_roundtrip(tmp_path):
    result = estimate_auto(_counts(), estimator="NAT")
    result.bootstrap_counts = _counts()

    path = tmp_path / "bootstrap_result.npz"
    save_result(result, path)
    restored = read_result(path)

    assert restored.bootstrap_counts is not None
    np.testing.assert_allclose(
        restored.bootstrap_counts.dd_boot,
        result.bootstrap_counts.dd_boot,
    )
    np.testing.assert_allclose(
        restored.bootstrap_cumulative_realizations,
        result.bootstrap_cumulative_realizations,
    )

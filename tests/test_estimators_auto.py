import numpy as np
from nugundam.angular.estimators import estimate_auto
from nugundam.angular.models import AngularAutoCounts

def make_counts(theta_grid):
    edges, centers = theta_grid
    return AngularAutoCounts(theta_edges=edges, theta_centers=centers, dd=np.array([3.0, 1.0]), rr=np.array([3.0, 1.0]), dr=np.array([4.0, 2.0]), metadata={"n_data": 4, "n_random": 4})

def test_estimate_auto_nat(theta_grid):
    result = estimate_auto(make_counts(theta_grid), estimator="NAT")
    np.testing.assert_allclose(result.wtheta, [0.0, 0.0])

def test_estimate_auto_dp(theta_grid):
    result = estimate_auto(make_counts(theta_grid), estimator="DP")
    np.testing.assert_allclose(result.wtheta, [1.0, 1/3], rtol=1e-12)

def test_estimate_auto_ls(theta_grid):
    result = estimate_auto(make_counts(theta_grid), estimator="LS")
    np.testing.assert_allclose(result.wtheta, [1.0, 0.5], rtol=1e-12)


def test_estimate_auto_ls_split_rr_norm(theta_grid):
    counts = make_counts(theta_grid)
    counts.metadata["rr_norm_pairs"] = 2.0
    result = estimate_auto(counts, estimator="LS")
    np.testing.assert_allclose(result.wtheta, [1.0, 5.0/6.0], rtol=1e-12)

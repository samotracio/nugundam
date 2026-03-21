import numpy as np
from nugundam.angular.estimators import estimate_cross
from nugundam.angular.models import AngularCrossCounts

def make_counts(theta_grid):
    edges, centers = theta_grid
    return AngularCrossCounts(theta_edges=edges, theta_centers=centers, d1d2=np.array([8.0, 4.0]), d1r2=np.array([4.0, 2.0]), r1d2=np.array([4.0, 1.0]), r1r2=np.array([2.0, 1.0]), metadata={"n_data1": 4, "n_random1": 2, "n_data2": 2, "n_random2": 2})

def test_estimate_cross_nat(theta_grid):
    result = estimate_cross(make_counts(theta_grid), estimator="NAT")
    np.testing.assert_allclose(result.wtheta, [1.0, 1.0])

def test_estimate_cross_dp(theta_grid):
    result = estimate_cross(make_counts(theta_grid), estimator="DP")
    np.testing.assert_allclose(result.wtheta, [1.0, 1.0])

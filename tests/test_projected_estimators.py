import numpy as np

from nugundam.projected.estimators import estimate_auto, estimate_cross
from nugundam.projected.models import ProjectedAutoCounts, ProjectedCrossCounts


def make_auto_counts(rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    dd = np.array([[12.0, 12.0], [12.0, 12.0]])
    rr = np.array([[6.0, 6.0], [6.0, 6.0]])
    dr = np.array([[16.0, 16.0], [16.0, 16.0]])
    return ProjectedAutoCounts(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        pi_edges=pi_edges,
        pi_centers=pi_centers,
        dd=dd,
        rr=rr,
        dr=dr,
        intpi_dd=np.array([24.0, 24.0]),
        intpi_rr=np.array([12.0, 12.0]),
        intpi_dr=np.array([32.0, 32.0]),
        metadata={"n_data": 4, "n_random": 4},
    )


def make_cross_counts(rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    d1d2 = np.array([[16.0, 16.0], [16.0, 16.0]])
    d1r2 = np.array([[8.0, 8.0], [8.0, 8.0]])
    r1d2 = np.array([[4.0, 4.0], [4.0, 4.0]])
    r1r2 = np.array([[4.0, 4.0], [4.0, 4.0]])
    return ProjectedCrossCounts(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        pi_edges=pi_edges,
        pi_centers=pi_centers,
        d1d2=d1d2,
        d1r2=d1r2,
        r1d2=r1d2,
        r1r2=r1r2,
        intpi_d1d2=np.array([32.0, 32.0]),
        intpi_d1r2=np.array([16.0, 16.0]),
        intpi_r1d2=np.array([8.0, 8.0]),
        intpi_r1r2=np.array([8.0, 8.0]),
        metadata={"n_data1": 4, "n_random1": 2, "n_data2": 2, "n_random2": 2},
    )


def test_estimate_projected_auto_nat(rp_pi_grid):
    result = estimate_auto(make_auto_counts(rp_pi_grid), estimator="NAT")
    np.testing.assert_allclose(result.wp, [4.0, 4.0])


def test_estimate_projected_auto_dp(rp_pi_grid):
    result = estimate_auto(make_auto_counts(rp_pi_grid), estimator="DP")
    np.testing.assert_allclose(result.wp, [4.0, 4.0])


def test_estimate_projected_cross_nat(rp_pi_grid):
    result = estimate_cross(make_cross_counts(rp_pi_grid), estimator="NAT")
    np.testing.assert_allclose(result.wp, [4.0, 4.0])


def test_estimate_projected_cross_ls(rp_pi_grid):
    result = estimate_cross(make_cross_counts(rp_pi_grid), estimator="LS")
    np.testing.assert_allclose(result.wp, [4.0, 4.0])


def test_estimate_projected_auto_ls_bootstrap_shape(rp_pi_grid):
    counts = make_auto_counts(rp_pi_grid)
    counts.dd_boot = np.array([
        [[12.0, 14.0], [12.0, 10.0]],
        [[12.0, 11.0], [12.0, 13.0]],
    ])
    result = estimate_auto(counts, estimator="LS")
    assert result.wp_err.shape == (2,)
    assert np.all(result.wp_err >= 0.0)


def test_estimate_projected_cross_ls_bootstrap_shape(rp_pi_grid):
    counts = make_cross_counts(rp_pi_grid)
    counts.d1d2_boot = np.array([
        [[16.0, 18.0], [16.0, 15.0]],
        [[16.0, 14.0], [16.0, 17.0]],
    ])
    counts.d1r2_boot = np.array([
        [[8.0, 9.0], [8.0, 7.0]],
        [[8.0, 8.5], [8.0, 7.5]],
    ])
    result = estimate_cross(counts, estimator="LS")
    assert result.wp_err.shape == (2,)
    assert np.all(result.wp_err >= 0.0)


def test_estimate_projected_auto_ls_split_rr_norm(rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    counts = ProjectedAutoCounts(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        pi_edges=pi_edges,
        pi_centers=pi_centers,
        dd=np.array([[6.0, 6.0], [6.0, 6.0]]),
        rr=np.array([[2.0, 2.0], [2.0, 2.0]]),
        dr=np.array([[4.0, 4.0], [4.0, 4.0]]),
        intpi_dd=np.array([12.0, 12.0]),
        intpi_rr=np.array([4.0, 4.0]),
        intpi_dr=np.array([8.0, 8.0]),
        metadata={"n_data": 4, "n_random": 4, "rr_norm_pairs": 1.0},
    )
    result = estimate_auto(counts, estimator="LS")
    np.testing.assert_allclose(result.wp, [5.0, 5.0], rtol=1e-12)

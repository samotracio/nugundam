import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from nugundam import plot_cov_matrix, plot_corr_matrix, plot_jk_regions
from nugundam.projected.models import ProjectedAutoCounts, ProjectedCorrelationResult


def _make_projected_result(rp_edges, rp_centers, pi_edges, pi_centers):
    counts = ProjectedAutoCounts(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        pi_edges=pi_edges,
        pi_centers=pi_centers,
        dd=np.array([[10.0, 8.0], [6.0, 5.0]]),
        rr=np.array([[9.0, 7.0], [5.0, 4.0]]),
        dr=np.array([[9.5, 7.5], [5.5, 4.5]]),
        metadata={},
    )
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    return ProjectedCorrelationResult(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        wp=np.array([0.2, 0.1]),
        wp_err=np.array([0.02, 0.03]),
        estimator="LS",
        counts=counts,
        cov=cov,
        metadata={},
    )


def test_plot_cov_and_corr_matrix_methods(rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    result = _make_projected_result(rp_edges, rp_centers, pi_edges, pi_centers)

    fig1, ax1 = plt.subplots()
    ax1 = plot_cov_matrix(result, ax=ax1, title="cov")
    assert ax1.get_xlabel() == r'$r_p\,[h^{-1}\,\mathrm{Mpc}]$'
    assert ax1.get_ylabel() == r'$r_p\,[h^{-1}\,\mathrm{Mpc}]$'
    assert ax1.get_title() == "cov"

    fig2, ax2 = plt.subplots()
    ax2 = result.plot_corr_matrix(ax=ax2, title="corr")
    assert ax2.get_xlabel() == r'$r_p\,[h^{-1}\,\mathrm{Mpc}]$'
    assert ax2.get_ylabel() == r'$r_p\,[h^{-1}\,\mathrm{Mpc}]$'
    assert ax2.get_title() == "corr"
    plt.close(fig1)
    plt.close(fig2)


def test_plot_corr_matrix_from_raw_covariance():
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    x = np.array([0.1, 1.0])
    fig, ax = plt.subplots()
    ax = plot_corr_matrix(cov, x=x, ax=ax)
    assert ax.get_xlabel() == "Separation"
    assert ax.get_ylabel() == "Separation"
    plt.close(fig)


def test_plot_jk_regions_from_prepared_sample():
    sample = SimpleNamespace(
        ra=np.array([1.0, 1.5, 2.0, 2.5]),
        dec=np.array([-0.5, -0.2, 0.1, 0.4]),
        region_id=np.array([0, 1, 1, 2], dtype=int),
    )
    fig, ax = plt.subplots()
    ax = plot_jk_regions(sample=sample, ax=ax, title="regions")
    assert ax.get_xlabel() == r'$\alpha\,[^\circ]$'
    assert ax.get_ylabel() == r'$\delta\,[^\circ]$'
    assert ax.get_title() == 'regions'
    assert ax.collections
    plt.close(fig)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from nugundam import plot_result2d, plotcf2d
from nugundam.projected.models import ProjectedAutoCounts, ProjectedCorrelationResult


def _make_projected_counts(rp_edges, rp_centers, pi_edges, pi_centers):
    return ProjectedAutoCounts(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        pi_edges=pi_edges,
        pi_centers=pi_centers,
        dd=np.array([[4.0, 6.0], [8.0, 10.0]]),
        rr=np.array([[2.0, 3.0], [4.0, 5.0]]),
        dr=np.array([[2.0, 3.0], [4.0, 5.0]]),
        intpi_dd=np.array([10.0, 18.0]),
        intpi_rr=np.array([5.0, 9.0]),
        intpi_dr=np.array([5.0, 9.0]),
        metadata={"n_data": 5, "n_random": 5, "config": {"estimator": "LS"}},
    )


def test_plotcf2d_symlog_and_contours(rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    z2d = np.array([[1.0, -0.5], [0.2, 2.0]])
    fig, ax = plt.subplots()
    ax = plotcf2d(
        rp_edges,
        pi_edges,
        z2d,
        ax=ax,
        color_scale='symlog',
        contours=True,
        mirror='four',
        colorbar=True,
    )
    assert ax.get_xlabel() == r'$r_p\,[h^{-1}\,\mathrm{Mpc}]$'
    assert ax.get_ylabel() == r'$\pi\,[h^{-1}\,\mathrm{Mpc}]$'
    assert len(ax.collections) >= 2  # heatmap + contours
    assert len(fig.axes) == 2  # main axes + colorbar axes
    plt.close(fig)


def test_plot_result2d_rebuilds_xi_with_log_positive(rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    counts = _make_projected_counts(rp_edges, rp_centers, pi_edges, pi_centers)
    result = ProjectedCorrelationResult(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        wp=np.array([2.0, 2.0]),
        wp_err=np.array([0.1, 0.1]),
        estimator='LS',
        counts=counts,
        metadata={'config': {'estimator': 'LS'}},
    )
    fig, ax = plt.subplots()
    ax = plot_result2d(result, ax=ax, which='xi', color_scale='log-positive', contours=True)
    assert ax.get_xlabel() == r'$r_p\,[h^{-1}\,\mathrm{Mpc}]$'
    assert ax.get_ylabel() == r'$\pi\,[h^{-1}\,\mathrm{Mpc}]$'
    assert len(fig.axes) == 2
    plt.close(fig)



def test_projected_counts_plot2d_method_with_raw_field(rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    counts = _make_projected_counts(rp_edges, rp_centers, pi_edges, pi_centers)
    fig, ax = plt.subplots()
    ax = counts.plot2d(ax=ax, which='dd', color_scale='linear', contours=False, mirror='none')
    assert ax.get_xlabel() == r'$r_p\,[h^{-1}\,\mathrm{Mpc}]$'
    assert ax.get_ylabel() == r'$\pi\,[h^{-1}\,\mathrm{Mpc}]$'
    assert len(fig.axes) == 2
    plt.close(fig)

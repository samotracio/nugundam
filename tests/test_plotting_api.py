import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from nugundam import plot_compare_ratio, plot_result, plotcf
from nugundam.angular.models import AngularAutoCounts, AngularCorrelationResult


def _make_result(theta_edges, theta_centers):
    counts = AngularAutoCounts(
        theta_edges=theta_edges,
        theta_centers=theta_centers,
        dd=np.array([10.0, 6.0]),
        rr=np.array([8.0, 5.0]),
        dr=np.array([9.0, 5.5]),
        metadata={'config': {'estimator': 'LS'}},
    )
    return AngularCorrelationResult(
        theta_edges=theta_edges,
        theta_centers=theta_centers,
        wtheta=np.array([0.2, 0.1]),
        wtheta_err=np.array([0.02, 0.01]),
        estimator='LS',
        counts=counts,
        metadata={'config': {'estimator': 'LS'}},
    )


def test_plotcf_routes_line_and_band_kwargs(theta_grid):
    _, centers = theta_grid
    fig, ax = plt.subplots()
    ax = plotcf(
        centers,
        np.array([0.2, 0.1]),
        yerr=np.array([0.02, 0.01]),
        ax=ax,
        errors='band',
        color='tab:red',
        marker='s',
        band_kwargs={'alpha': 0.35},
        label='sample',
    )
    assert ax.get_xscale() == 'log'
    assert ax.get_yscale() == 'log'
    line = ax.lines[0]
    assert line.get_label() == 'sample'
    assert line.get_marker() == 's'
    assert line.get_color() == 'tab:red'
    assert ax.collections  # fill_between polygon
    plt.close(fig)


def test_plot_result_and_method_defaults(theta_grid):
    edges, centers = theta_grid
    result = _make_result(edges, centers)
    fig, ax = plt.subplots()
    ax = plot_result(result, ax=ax)
    assert ax.get_xlabel() == r'$\theta\,[^\circ]$'
    assert ax.get_ylabel() == r'$w(\theta)$'

    fig2, ax2 = plt.subplots()
    ax2 = result.plot(ax=ax2, label='corr')
    assert ax2.lines[0].get_label() == 'corr'
    plt.close(fig)
    plt.close(fig2)


def test_counts_plot_overlay_and_custom_field(theta_grid):
    edges, centers = theta_grid
    result = _make_result(edges, centers)
    counts = result.counts
    fig, ax = plt.subplots()
    counts.plot(ax=ax, which='rr', color='tab:blue')
    result.plot(ax=ax, errors='none', color='black')
    assert len(ax.lines) == 2
    assert ax.get_ylabel() == r'$w(\theta)$'  # last call wins on shared axes
    plt.close(fig)



def test_plot_compare_ratio_basic(theta_grid):
    edges, centers = theta_grid
    res_data = _make_result(edges, centers)
    res_ctrl = AngularCorrelationResult(
        theta_edges=edges,
        theta_centers=centers,
        wtheta=np.array([0.1, 0.05]),
        wtheta_err=np.array([0.01, 0.005]),
        estimator='LS',
        counts=res_data.counts,
        metadata={},
    )
    res_auto = AngularCorrelationResult(
        theta_edges=edges,
        theta_centers=centers,
        wtheta=np.array([0.08, 0.04]),
        wtheta_err=np.array([0.008, 0.004]),
        estimator='LS',
        counts=res_data.counts,
        metadata={},
    )

    fig, (ax_top, ax_ratio) = plot_compare_ratio(
        curves=[
            {'key': 'data', 'result': res_data, 'label': 'data', 'marker': 'o'},
            {'key': 'control', 'result': res_ctrl, 'label': 'control', 'linestyle': ':'},
            {'key': 'auto', 'result': res_auto, 'label': 'auto', 'marker': 'd', 'linestyle': 'none'},
        ],
        ratios=[
            {'numerator': 'data', 'denominator': 'control', 'label': 'data/control', 'marker': 'o'},
        ],
        ratio_ylabel='data/control',
        top_errors='none',
        ratio_errors='none',
    )

    assert ax_top.get_ylabel() == r'$w(\theta)$'
    assert ax_ratio.get_ylabel() == 'data/control'
    assert ax_ratio.get_xlabel() == r'$\theta\,[^\circ]$'
    assert ax_top.get_xscale() == 'log'
    assert ax_ratio.get_xscale() == 'log'
    assert ax_ratio.get_yscale() == 'linear'
    assert len(ax_top.lines) == 3
    assert len(ax_ratio.lines) == 2  # ratio curve + reference line
    np.testing.assert_allclose(ax_ratio.lines[0].get_ydata(), np.array([2.0, 2.0]))
    assert ax_top.get_legend() is not None
    plt.close(fig)


def test_plot_compare_ratio_rejects_x_with_no_common_plotted_bins(theta_grid):
    edges, centers = theta_grid
    res_a = _make_result(edges, centers)
    res_b = AngularCorrelationResult(
        theta_edges=np.array([0.5, 0.7, 1.0]),
        theta_centers=np.array([0.6, 0.85]),
        wtheta=np.array([0.1, 0.2]),
        wtheta_err=np.array([0.01, 0.02]),
        estimator='LS',
        counts=res_a.counts,
        metadata={},
    )

    fig, (ax_top, ax_ratio) = plt.subplots(2, 1, sharex=True)
    try:
        try:
            plot_compare_ratio(
                {'a': res_a, 'b': res_b},
                ratios=[('a', 'b')],
                axes=(ax_top, ax_ratio),
            )
        except ValueError as exc:
            assert 'do not share any plotted x values' in str(exc)
        else:
            raise AssertionError('Expected ValueError when no common plotted bins exist')
    finally:
        plt.close(fig)


def test_plot_compare_ratio_supports_ratio_band(theta_grid):
    edges, centers = theta_grid
    res_data = _make_result(edges, centers)
    res_ctrl = AngularCorrelationResult(
        theta_edges=edges,
        theta_centers=centers,
        wtheta=np.array([0.1, 0.05]),
        wtheta_err=np.array([0.01, 0.005]),
        estimator='LS',
        counts=res_data.counts,
        metadata={},
    )

    fig, (ax_top, ax_ratio) = plot_compare_ratio(
        {'data': res_data, 'control': res_ctrl},
        ratios=[{'num': 'data', 'den': 'control', 'errors': 'band', 'label': 'ratio'}],
        ratio_legend=True,
    )
    assert ax_ratio.collections  # fill_between polygon from ratio band
    assert ax_ratio.get_legend() is not None
    plt.close(fig)


def test_plot_compare_ratio_uses_only_bins_visible_upstairs():
    edges = np.array([1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3])
    centers = np.array([1.5e-3, 3.0e-3, 6.0e-3])
    counts = AngularAutoCounts(
        theta_edges=edges,
        theta_centers=centers,
        dd=np.array([10.0, 9.0, 8.0]),
        rr=np.array([7.0, 6.0, 5.0]),
        dr=np.array([8.0, 7.0, 6.0]),
        metadata={'config': {'estimator': 'LS'}},
    )
    res_num = AngularCorrelationResult(
        theta_edges=edges,
        theta_centers=centers,
        wtheta=np.array([0.10, 0.05, 0.02]),
        wtheta_err=np.array([0.01, 0.01, 0.03]),
        estimator='LS',
        counts=counts,
        metadata={},
    )
    res_den = AngularCorrelationResult(
        theta_edges=edges,
        theta_centers=centers,
        wtheta=np.array([0.05, 0.025, 0.01]),
        wtheta_err=np.array([0.005, 0.005, 0.001]),
        estimator='LS',
        counts=counts,
        metadata={},
    )

    fig, (ax_top, ax_ratio) = plot_compare_ratio(
        {'num': res_num, 'den': res_den},
        ratios=[('num', 'den')],
        top_errors='bar',
        ratio_errors='none',
    )

    # Third numerator point is masked upstairs because y - yerr <= 0 on log-y.
    assert len(ax_top.lines[0].get_xdata()) == 2
    np.testing.assert_allclose(ax_ratio.lines[0].get_xdata(), ax_top.lines[0].get_xdata())
    np.testing.assert_allclose(ax_ratio.lines[0].get_ydata(), np.array([2.0, 2.0]))
    plt.close(fig)

import numpy as np

from nugundam import plot_result, write_result, read_result
from nugundam.projected.models import ProjectedAutoCounts, ProjectedCorrelationResult


def test_projected_roundtrip_and_ascii(tmp_path, rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    counts = ProjectedAutoCounts(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        pi_edges=pi_edges,
        pi_centers=pi_centers,
        dd=np.array([[1.0, 2.0], [3.0, 4.0]]),
        rr=np.array([[1.0, 2.0], [3.0, 4.0]]),
        dr=np.array([[1.0, 2.0], [3.0, 4.0]]),
        intpi_dd=np.array([6.0, 14.0]),
        intpi_rr=np.array([6.0, 14.0]),
        intpi_dr=np.array([6.0, 14.0]),
        metadata={"n_data": 3, "n_random": 3, "config": {"estimator": "LS"}},
    )
    result = ProjectedCorrelationResult(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        wp=np.array([0.1, 0.2]),
        wp_err=np.array([0.01, 0.02]),
        estimator='LS',
        counts=counts,
        metadata={'config': {'estimator': 'LS'}},
    )
    path = tmp_path / 'pcf_result.gres'
    write_result(result, path)
    loaded = read_result(path)
    np.testing.assert_allclose(loaded.wp, result.wp)
    cf_path = tmp_path / 'pcf.txt'
    ct_path = tmp_path / 'proj_counts.txt'
    result.to_ascii(cf_path)
    counts.to_ascii(ct_path)
    assert cf_path.exists() and ct_path.exists()
    ax = plot_result(result)
    assert ax.get_xlabel() == r'$r_p\,[h^{-1}\,\mathrm{Mpc}]$'

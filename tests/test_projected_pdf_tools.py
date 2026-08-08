import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from nugundam import CompressedPdfGMM, compress_pdfs_to_gmm, plot_gmm_for_object


def _column_names(tab):
    if hasattr(tab, "colnames"):
        return list(tab.colnames)
    if hasattr(tab, "columns"):
        return list(tab.columns)
    return list(tab.dtype.names)


def test_compress_pdfs_to_gmm_returns_result_and_arrays():
    pdfs = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
        ],
        dtype=np.float64,
    )
    chi_grid = np.array([100.0, 110.0, 120.0], dtype=np.float64)

    out = compress_pdfs_to_gmm(
        pdfs,
        chi_grid=chi_grid,
        k=2,
    )
    assert isinstance(out, CompressedPdfGMM)
    assert out.alpha.shape == (2, 2)
    np.testing.assert_allclose(out.alpha.sum(axis=0), 1.0)
    assert out.chi_grid.shape == (3,)

    alpha, mu, sigma = compress_pdfs_to_gmm(
        pdfs,
        chi_grid=chi_grid,
        k=2,
        output="arrays",
    )
    np.testing.assert_allclose(alpha, out.alpha)
    np.testing.assert_allclose(mu, out.mu)
    np.testing.assert_allclose(sigma, out.sigma)


def test_compress_pdfs_to_gmm_table_exports():
    pdfs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
        ],
        dtype=np.float64,
    )
    chi_edges = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float64)

    tab = compress_pdfs_to_gmm(
        pdfs,
        chi_grid=chi_edges,
        grid_kind="edges",
        k=3,
        ids=np.array([101, 102]),
        output="table",
        table_layout="wide",
    )
    names = _column_names(tab)
    assert "object_index" in names
    assert "object_id" in names
    assert "alpha0" in names
    assert len(tab) == 2

    out = compress_pdfs_to_gmm(pdfs, chi_grid=chi_edges, grid_kind="edges", k=3)
    long_tab = out.to_table(layout="long")
    names_long = _column_names(long_tab)
    assert len(long_tab) == 6
    assert set(["object_index", "component", "alpha", "mu", "sigma"]).issubset(names_long)


def test_plot_gmm_for_object_top_level_helper_returns_payload():
    pdfs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
        ],
        dtype=np.float64,
    )
    chi_edges = np.array([100.0, 110.0, 125.0, 145.0], dtype=np.float64)

    fig, ax = plt.subplots()
    ax, payload = plot_gmm_for_object(
        0,
        pdfs,
        chi_grid=chi_edges,
        grid_kind="edges",
        k=3,
        ax=ax,
        return_data=True,
    )
    assert ax.get_xlabel() == r"$\chi$"
    assert ax.get_ylabel() == r"$p(\chi)$"
    assert payload["alpha"].shape == (3,)
    assert payload["component_curves"].shape[0] == 3
    assert payload["grid_kind"] == "edges"
    assert len(ax.lines) == 5
    plt.close(fig)


@pytest.mark.skipif(__import__("importlib").util.find_spec("astropy") is None, reason="astropy not installed")
def test_plot_gmm_for_object_adds_redshift_top_axis_when_zgrid_is_available():
    pdfs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
        ],
        dtype=np.float64,
    )
    z_edges = np.array([0.05, 0.10, 0.20, 0.35], dtype=np.float64)

    fig, ax = plt.subplots()
    ax, payload = plot_gmm_for_object(
        0,
        pdfs,
        z_grid=z_edges,
        grid_kind="edges",
        k=3,
        ax=ax,
        return_data=True,
    )
    assert payload["secax"] is not None
    assert payload["secax"].get_xlabel() == r"$z$"
    assert payload["z_dense"] is not None
    assert payload["chi_dense"] is not None
    plt.close(fig)


def test_plot_gmm_for_object_skips_redshift_top_axis_for_chi_only_input():
    pdfs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
        ],
        dtype=np.float64,
    )
    chi_edges = np.array([100.0, 110.0, 125.0, 145.0], dtype=np.float64)

    fig, ax = plt.subplots()
    ax, payload = plot_gmm_for_object(
        0,
        pdfs,
        chi_grid=chi_edges,
        grid_kind="edges",
        k=3,
        ax=ax,
        return_data=True,
    )
    assert payload["secax"] is None
    assert payload["z_dense"] is None
    assert payload["chi_dense"] is None
    plt.close(fig)


def test_edge_grid_gmm_compression_includes_within_bin_variance():
    pdfs = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    chi_edges = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    out = compress_pdfs_to_gmm(pdfs, chi_grid=chi_edges, grid_kind="edges", k=1)
    np.testing.assert_allclose(out.alpha[:, 0], [1.0])
    np.testing.assert_allclose(out.mu[:, 0], [15.0])
    np.testing.assert_allclose(out.sigma[:, 0], [10.0 / np.sqrt(12.0)], rtol=1e-12)


def test_edge_grid_gmm_compression_can_preserve_legacy_center_moments():
    pdfs = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    chi_edges = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    out = compress_pdfs_to_gmm(pdfs, chi_grid=chi_edges, grid_kind="edges", k=1, edge_moments=False)
    np.testing.assert_allclose(out.mu[:, 0], [15.0])
    assert out.sigma[0, 0] < 1e-3

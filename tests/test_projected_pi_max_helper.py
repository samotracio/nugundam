import numpy as np
import pytest

from nugundam import PiMaxEstimate, estimate_pi_max_from_pdfs


def test_estimate_pi_max_from_pdfs_returns_expected_auto_summary_for_probabilities():
    pdfs = np.array(
        [
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float64,
    )
    chi_grid = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    out = estimate_pi_max_from_pdfs(
        pdfs,
        chi_grid=chi_grid,
        pdf_normalization="probability",
        multiplier=2.5,
    )

    assert isinstance(out, PiMaxEstimate)
    assert out.case == "auto"
    assert out.zc is None
    assert out.zc2 is None
    np.testing.assert_allclose(out.chi, chi_grid)

    mu0 = 0.2 * 10.0 + 0.5 * 20.0 + 0.3 * 30.0
    mu1 = 0.1 * 10.0 + 0.2 * 20.0 + 0.7 * 30.0
    var0 = 0.2 * (10.0 - mu0) ** 2 + 0.5 * (20.0 - mu0) ** 2 + 0.3 * (30.0 - mu0) ** 2
    var1 = 0.1 * (10.0 - mu1) ** 2 + 0.2 * (20.0 - mu1) ** 2 + 0.7 * (30.0 - mu1) ** 2
    sig = np.sqrt([var0, var1])
    sigma_chi_eff_1 = np.sqrt(np.median(sig ** 2))
    sigma_pw_eff = np.sqrt(2.0 * np.median(sig ** 2))

    np.testing.assert_allclose(out.mu_chi, [mu0, mu1])
    np.testing.assert_allclose(out.sig_chi, sig)
    np.testing.assert_allclose(out.sigma_chi_eff_1, sigma_chi_eff_1)
    assert out.sigma_chi_eff_2 is None
    np.testing.assert_allclose(out.sigma_pw_eff, sigma_pw_eff)
    np.testing.assert_allclose(out.sigma_pw_eff_auto, sigma_pw_eff)
    np.testing.assert_allclose(out.pi_max_guess, 2.5 * sigma_pw_eff)
    assert out.pdf_normalization == "probability"
    assert out.pdf_normalization2 is None


def test_estimate_pi_max_from_pdfs_auto_detects_density_on_uneven_bins():
    pdfs = np.array(
        [
            [0.50, 0.25],  # integrates to 1 over widths [1, 2]
            [0.20, 0.40],
        ],
        dtype=np.float64,
    )
    chi_edges = np.array([0.0, 1.0, 3.0], dtype=np.float64)

    out = estimate_pi_max_from_pdfs(
        pdfs,
        chi_grid=chi_edges,
        grid_kind="edges",
        pdf_normalization="auto",
        multiplier=3.0,
        output="dict",
    )

    assert out["pdf_normalization"] == "density"
    assert out["case"] == "auto"
    np.testing.assert_allclose(out["chi"], [0.5, 2.0])
    # First row becomes discrete probabilities [0.5, 0.5].
    np.testing.assert_allclose(out["mu_chi"][0], 1.25)
    np.testing.assert_allclose(out["sig_chi"][0], 0.75)
    np.testing.assert_allclose(out["pi_max_guess"], 3.0 * out["sigma_pw_eff"])


@pytest.mark.skipif(__import__("importlib").util.find_spec("astropy") is None, reason="astropy not installed")
def test_estimate_pi_max_from_pdfs_returns_z_centers_when_using_z_grid():
    pdfs = np.array([[0.2, 0.8]], dtype=np.float64)
    z_edges = np.array([0.1, 0.2, 0.5], dtype=np.float64)

    out = estimate_pi_max_from_pdfs(
        pdfs,
        z_grid=z_edges,
        grid_kind="edges",
        pdf_normalization="probability",
    )

    np.testing.assert_allclose(out.zc, [0.15, 0.35])
    assert out.chi.shape == (2,)
    assert np.all(np.diff(out.chi) > 0.0)


def test_estimate_pi_max_from_pdfs_supports_photo_photo_cross_case():
    pdfs1 = np.array(
        [
            [0.5, 0.5],
            [0.8, 0.2],
        ],
        dtype=np.float64,
    )
    pdfs2 = np.array(
        [
            [0.25, 0.75],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    chi_grid = np.array([0.0, 10.0], dtype=np.float64)

    out = estimate_pi_max_from_pdfs(
        pdfs1,
        pdfs2=pdfs2,
        chi_grid=chi_grid,
        pdf_normalization="probability",
        multiplier=4.0,
    )

    sig1 = np.array([5.0, 4.0])
    sig2 = np.array([np.sqrt(18.75), 3.0])
    sigma1 = np.sqrt(np.median(sig1 ** 2))
    sigma2 = np.sqrt(np.median(sig2 ** 2))
    sigma_pw = np.sqrt(sigma1 ** 2 + sigma2 ** 2)

    assert out.case == "cross_photo_photo"
    np.testing.assert_allclose(out.sig_chi, sig1)
    np.testing.assert_allclose(out.sig_chi2, sig2)
    np.testing.assert_allclose(out.sigma_chi_eff_1, sigma1)
    np.testing.assert_allclose(out.sigma_chi_eff_2, sigma2)
    np.testing.assert_allclose(out.sigma_pw_eff, sigma_pw)
    np.testing.assert_allclose(out.sigma_pw_eff_cross, sigma_pw)
    np.testing.assert_allclose(out.pi_max_guess, 4.0 * sigma_pw)



def test_estimate_pi_max_from_pdfs_supports_spec_photo_cross_case():
    pdfs = np.array(
        [
            [0.2, 0.8],
            [0.6, 0.4],
        ],
        dtype=np.float64,
    )
    chi_grid = np.array([100.0, 160.0], dtype=np.float64)

    out = estimate_pi_max_from_pdfs(
        pdfs,
        chi_grid=chi_grid,
        pdf_normalization="probability",
        sample2_kind="spec",
        multiplier=3.0,
    )

    sig = np.array([24.0, 29.393876913398138])
    sigma1 = np.sqrt(np.median(sig ** 2))

    assert out.case == "cross_spec_photo"
    assert out.sig_chi2 is None
    assert out.sigma_chi_eff_2 is None
    np.testing.assert_allclose(out.sigma_chi_eff_1, sigma1)
    np.testing.assert_allclose(out.sigma_pw_eff, sigma1)
    np.testing.assert_allclose(out.pi_max_guess, 3.0 * sigma1)

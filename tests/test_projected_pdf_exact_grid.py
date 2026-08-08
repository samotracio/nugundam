import numpy as np

from nugundam.projected.fortran_bridge import run_rppi_auto_counts, run_rppi_cross_counts
from nugundam.projected.models import PDFSourceSpec, ProjectedAutoConfig, ProjectedBinning, ProjectedCrossConfig
from nugundam.projected.prepare import prepare_projected_auto, prepare_projected_cross


def test_prepare_projected_auto_accepts_exact_grid_pdf_source():
    data = {
        "ra": np.array([0.0, 0.5, 1.0], dtype=np.float64),
        "dec": np.array([0.0, 0.2, -0.1], dtype=np.float64),
        "wei": np.ones(3, dtype=np.float64),
    }
    random = {
        "ra": np.array([0.1, 0.6, 1.1, 1.4], dtype=np.float64),
        "dec": np.array([0.0, 0.1, -0.2, 0.3], dtype=np.float64),
    }
    matrix = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
        ],
        dtype=np.float64,
    )
    chi_centers = np.array([100.0, 110.0, 120.0], dtype=np.float64)

    cfg = ProjectedAutoConfig(
        estimator="NAT",
        binning=ProjectedBinning.from_binsize(nsepp=2, seppmin=0.1, dsepp=0.5, logsepp=True, nsepv=1, dsepv=20.0),
    )
    cfg.pdf.enabled = True
    cfg.pdf.kind = "grid_chi_exact"
    cfg.pdf.prob_floor = 0.0
    cfg.pdf_source.enabled = True
    cfg.pdf_source.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.pdf_source.chi_grid = chi_centers

    data_p, rand_p, meta = prepare_projected_auto(data, random, cfg)
    assert data_p.pdf_repr == "grid_chi_exact"
    assert data_p.pdf_prob_lib.shape == (3, 3)
    assert data_p.pdf_cdf_lib.shape == (3, 3)
    assert np.all(data_p.pdf_hi_idx >= data_p.pdf_lo_idx)
    assert rand_p.pdf_repr == "grid_chi_exact"
    assert rand_p.pdf_prob_lib is data_p.pdf_prob_lib
    assert meta["pdf_kind"] == "grid_chi_exact"
    assert meta["pdf_input_mode"] == "empirical_grid"


def test_run_rppi_auto_counts_exact_grid_delta_pdfs_counts_one_pair():
    data = {
        "ra": np.array([0.0, 0.1], dtype=np.float64),
        "dec": np.array([0.0, 0.0], dtype=np.float64),
        "wei": np.ones(2, dtype=np.float64),
    }
    random = {
        "ra": np.array([0.2, 0.3], dtype=np.float64),
        "dec": np.array([0.0, 0.0], dtype=np.float64),
    }
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    chi_centers = np.array([10.0, 20.0], dtype=np.float64)

    cfg = ProjectedAutoConfig(
        estimator="NAT",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.01, dsepp=3.0, logsepp=True, nsepv=1, dsepv=15.0),
    )
    cfg.progress.enabled = False
    cfg.pdf.enabled = True
    cfg.pdf.kind = "grid_chi_exact"
    cfg.pdf.prob_floor = 0.0
    cfg.pdf.seed = 1
    cfg.pdf_source.enabled = True
    cfg.pdf_source.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.pdf_source.chi_grid = chi_centers

    data_p, _rand_p, _meta = prepare_projected_auto(data, random, cfg)
    dd, _boot, _normb, _sumwb, _touch = run_rppi_auto_counts(
        data_p,
        rp_edges=np.array([0.01, 100.0], dtype=np.float64),
        pi_edges=np.array([0.0, 15.0], dtype=np.float64),
        nthreads=1,
        weight_mode="unweighted",
        doboot=False,
        dojk=False,
        nreg=0,
        nbts=0,
        bseed=123,
        cntid="DD",
        progress_file=None,
    )
    np.testing.assert_allclose(dd, np.array([[1.0]], dtype=np.float64), rtol=0.0, atol=1e-12)


def test_run_rppi_cross_counts_exact_grid_delta_pdfs_counts_one_pair():
    data1 = {
        "ra": np.array([0.0], dtype=np.float64),
        "dec": np.array([0.0], dtype=np.float64),
        "wei": np.ones(1, dtype=np.float64),
    }
    data2 = {
        "ra": np.array([0.1], dtype=np.float64),
        "dec": np.array([0.0], dtype=np.float64),
        "wei": np.ones(1, dtype=np.float64),
    }
    random1 = {
        "ra": np.array([0.2], dtype=np.float64),
        "dec": np.array([0.0], dtype=np.float64),
    }
    random2 = {
        "ra": np.array([0.3], dtype=np.float64),
        "dec": np.array([0.0], dtype=np.float64),
    }
    matrix1 = np.array([[1.0, 0.0]], dtype=np.float64)
    matrix2 = np.array([[0.0, 1.0]], dtype=np.float64)
    chi_centers = np.array([10.0, 20.0], dtype=np.float64)

    cfg = ProjectedCrossConfig(
        estimator="NAT",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.01, dsepp=3.0, logsepp=True, nsepv=1, dsepv=15.0),
    )
    cfg.progress.enabled = False
    cfg.pdf.enabled = True
    cfg.pdf.kind = "grid_chi_exact"
    cfg.pdf.prob_floor = 0.0
    cfg.pdf.seed = 1
    cfg.pdf_source.enabled = True
    cfg.pdf_source.chi_grid = chi_centers
    cfg.pdf_source.pdf_data1 = PDFSourceSpec(matrix=matrix1)
    cfg.pdf_source.pdf_data2 = PDFSourceSpec(matrix=matrix2)

    prep1, _prep_r1, prep2, _prep_r2, _meta = prepare_projected_cross(data1, random1, data2, random2, cfg)
    d1d2, _boot, _touch = run_rppi_cross_counts(
        prep1,
        prep2,
        rp_edges=np.array([0.01, 100.0], dtype=np.float64),
        pi_edges=np.array([0.0, 15.0], dtype=np.float64),
        nthreads=1,
        weight_mode="unweighted",
        doboot=False,
        dojk=False,
        nreg=0,
        nbts=0,
        bseed=123,
        cntid="D1D2",
        progress_file=None,
    )
    np.testing.assert_allclose(d1d2, np.array([[1.0]], dtype=np.float64), rtol=0.0, atol=1e-12)


def test_run_rppi_auto_counts_exact_grid_multi_pi_delta_pair_shell():
    data = {
        "ra": np.array([0.0, 0.1], dtype=np.float64),
        "dec": np.array([0.0, 0.0], dtype=np.float64),
    }
    random = {
        "ra": np.array([0.2, 0.3], dtype=np.float64),
        "dec": np.array([0.0, 0.0], dtype=np.float64),
    }
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    chi_centers = np.array([10.0, 20.0], dtype=np.float64)

    cfg = ProjectedAutoConfig(
        estimator="NAT",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.01, dsepp=3.0, logsepp=True, nsepv=2, dsepv=5.0),
    )
    cfg.progress.enabled = False
    cfg.pdf.enabled = True
    cfg.pdf.kind = "grid_chi_exact"
    cfg.pdf.prob_floor = 0.0
    cfg.pdf.seed = 1
    cfg.pdf_source.enabled = True
    cfg.pdf_source.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.pdf_source.chi_grid = chi_centers

    data_p, _rand_p, _meta = prepare_projected_auto(data, random, cfg)
    dd, _boot, _normb, _sumwb, _touch = run_rppi_auto_counts(
        data_p,
        rp_edges=np.array([0.01, 100.0], dtype=np.float64),
        pi_edges=np.array([0.0, 5.0, 15.0], dtype=np.float64),
        nthreads=1,
        weight_mode="unweighted",
        doboot=False,
        dojk=False,
        nreg=0,
        nbts=0,
        bseed=123,
        cntid="DD",
        progress_file=None,
    )
    np.testing.assert_allclose(dd, np.array([[0.0, 1.0]], dtype=np.float64), rtol=0.0, atol=1e-12)


def test_run_rppi_auto_counts_exact_grid_multi_pi_adds_to_wide_bin():
    data = {
        "ra": np.array([0.0, 0.1, 0.2], dtype=np.float64),
        "dec": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    }
    random = {
        "ra": np.array([0.2, 0.3, 0.4], dtype=np.float64),
        "dec": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    }
    matrix = np.array(
        [[0.70, 0.30, 0.00], [0.20, 0.50, 0.30], [0.10, 0.40, 0.50]],
        dtype=np.float64,
    )
    chi_centers = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    cfg = ProjectedAutoConfig(
        estimator="NAT",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.01, dsepp=3.0, logsepp=True, nsepv=1, dsepv=30.0),
    )
    cfg.progress.enabled = False
    cfg.pdf.enabled = True
    cfg.pdf.kind = "grid_chi_exact"
    cfg.pdf.prob_floor = 0.0
    cfg.pdf.seed = 1
    cfg.pdf_source.enabled = True
    cfg.pdf_source.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.pdf_source.chi_grid = chi_centers

    data_p, _rand_p, _meta = prepare_projected_auto(data, random, cfg)
    rp_edges = np.array([0.01, 100.0], dtype=np.float64)
    wide, *_ = run_rppi_auto_counts(
        data_p,
        rp_edges=rp_edges,
        pi_edges=np.array([0.0, 30.0], dtype=np.float64),
        nthreads=1,
        weight_mode="unweighted",
        doboot=False,
        dojk=False,
        nreg=0,
        nbts=0,
        bseed=123,
        cntid="DD",
        progress_file=None,
    )
    multi, *_ = run_rppi_auto_counts(
        data_p,
        rp_edges=rp_edges,
        pi_edges=np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64),
        nthreads=1,
        weight_mode="unweighted",
        doboot=False,
        dojk=False,
        nreg=0,
        nbts=0,
        bseed=123,
        cntid="DD",
        progress_file=None,
    )
    np.testing.assert_allclose(multi.sum(axis=1), wide[:, 0], rtol=0.0, atol=1e-12)

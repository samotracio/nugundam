import numpy as np

from nugundam.projected.models import PDFSourceSpec, ProjectedAutoConfig, ProjectedBinning, ProjectedCrossConfig
from nugundam.projected.prepare import prepare_projected_auto, prepare_projected_cross


def _auto_catalogs():
    data = {
        "ra": np.array([0.0, 0.5, 1.0], dtype=np.float64),
        "dec": np.array([0.0, 0.2, -0.1], dtype=np.float64),
        "wei": np.ones(3, dtype=np.float64),
    }
    random = {
        "ra": np.array([0.1, 0.6, 1.1, 1.4], dtype=np.float64),
        "dec": np.array([0.0, 0.1, -0.2, 0.3], dtype=np.float64),
    }
    return data, random


def _cross_catalogs():
    data1 = {
        "ra": np.array([0.0, 0.5, 1.0], dtype=np.float64),
        "dec": np.array([0.0, 0.2, -0.1], dtype=np.float64),
        "wei": np.ones(3, dtype=np.float64),
    }
    random1 = {
        "ra": np.array([0.1, 0.6, 1.1], dtype=np.float64),
        "dec": np.array([0.0, 0.1, -0.2], dtype=np.float64),
    }
    data2 = {
        "ra": np.array([2.0, 2.5, 3.0], dtype=np.float64),
        "dec": np.array([0.1, -0.2, 0.0], dtype=np.float64),
        "wei": np.ones(3, dtype=np.float64),
    }
    random2 = {
        "ra": np.array([2.1, 2.6, 3.1], dtype=np.float64),
        "dec": np.array([0.2, -0.1, 0.1], dtype=np.float64),
    }
    return data1, random1, data2, random2


def test_prepare_projected_auto_accepts_empirical_pdf_source_edges():
    data, random = _auto_catalogs()
    matrix = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
        ],
        dtype=np.float64,
    )
    chi_edges = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float64)

    cfg = ProjectedAutoConfig(
        estimator="NAT",
        binning=ProjectedBinning.from_binsize(nsepp=2, seppmin=0.1, dsepp=0.5, logsepp=True, nsepv=1, dsepv=20.0),
    )
    cfg.pdf.enabled = True
    cfg.pdf.k = 5
    cfg.pdf_source.enabled = True
    cfg.pdf_source.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.pdf_source.chi_grid = chi_edges
    cfg.pdf_source.grid_kind = "edges"

    data_p, rand_p, meta = prepare_projected_auto(data, random, cfg)
    assert data_p.pdf_k == 5
    assert data_p.pdf_alpha_lib.shape == (5, 3)
    assert rand_p.pdf_k == 5
    assert meta["pdf_input_mode"] == "empirical_grid"
    assert meta["pdf_compressor"] == "segments_equal_mass"


def test_prepare_projected_cross_accepts_empirical_pdf_source():
    data1, random1, data2, random2 = _cross_catalogs()
    matrix1 = np.array(
        [
            [0.60, 0.30, 0.10],
            [0.10, 0.80, 0.10],
            [0.15, 0.25, 0.60],
        ],
        dtype=np.float64,
    )
    matrix2 = np.array(
        [
            [0.20, 0.50, 0.30],
            [0.30, 0.40, 0.30],
            [0.50, 0.30, 0.20],
        ],
        dtype=np.float64,
    )
    chi_centers = np.array([90.0, 110.0, 130.0], dtype=np.float64)

    cfg = ProjectedCrossConfig(
        estimator="LS",
        binning=ProjectedBinning.from_binsize(nsepp=2, seppmin=0.1, dsepp=0.5, logsepp=True, nsepv=1, dsepv=20.0),
    )
    cfg.pdf.enabled = True
    cfg.pdf.k = 4
    cfg.pdf_source.enabled = True
    cfg.pdf_source.chi_grid = chi_centers
    cfg.pdf_source.pdf_data1 = PDFSourceSpec(matrix=matrix1)
    cfg.pdf_source.pdf_data2 = PDFSourceSpec(matrix=matrix2)

    prep1, prep_r1, prep2, prep_r2, meta = prepare_projected_cross(data1, random1, data2, random2, cfg)
    assert prep1.pdf_k == 4
    assert prep2.pdf_k == 4
    assert prep1.pdf_alpha_lib.shape == (4, 3)
    assert prep2.pdf_alpha_lib.shape == (4, 3)
    assert prep_r1 is not None and prep_r2 is not None
    assert meta["pdf_input_mode"] == "empirical_grid"

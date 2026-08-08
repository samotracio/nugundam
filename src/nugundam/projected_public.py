"""Flat re-exports for the projected public API."""
from __future__ import annotations

from .projected.api import pcf, pccf, proj_auto_counts, proj_cross_counts
from .projected.diagnostics import summarize_pair_diagnostics, print_pair_diagnostics
from .projected.fixed_rp_diagnostic import count_auto_fixed_rp, count_cross_fixed_rp, run_auto_mc_pdf_fixed_rp
from .projected.pdf_tools import CompressedPdfGMM, PiMaxEstimate, compress_pdfs_to_gmm, estimate_pi_max_from_pdfs, plot_gmm_for_object
from .projected.models import (
    BootstrapSpec,
    JackknifeSpec,
    DistanceSpec,
    PreparedProjectedSample,
    ProjectedAutoConfig,
    ProjectedAutoCounts,
    ProjectedAutoCountsConfig,
    ProjectedAutoCountsResult,
    ProjectedBinning,
    ProjectedCatalogColumns,
    ProjectedCorrelationResult,
    PDFSourceSpec,
    ProjectedPdfSourceSpec,
    ProjectedMCPdfSpec,
    ProjectedCrossConfig,
    ProjectedCrossCounts,
    ProjectedCrossCountsConfig,
    ProjectedCrossCountsResult,
    ProjectedGridSpec,
    SplitRandomSpec,
)

__all__ = [
    "ProjectedCatalogColumns",
    "AutoMarkSpec",
    "CrossMarkSpec",
    "ProjectedBinning",
    "ProjectedGridSpec",
    "SplitRandomSpec",
    "DistanceSpec",
    "PDFSourceSpec",
    "ProjectedPdfSourceSpec",
    "ProjectedMCPdfSpec",
    "BootstrapSpec",
    "JackknifeSpec",
    "ProjectedAutoConfig",
    "ProjectedCrossConfig",
    "ProjectedAutoCountsConfig",
    "ProjectedCrossCountsConfig",
    "PreparedProjectedSample",
    "ProjectedAutoCounts",
    "ProjectedCrossCounts",
    "ProjectedAutoCountsResult",
    "ProjectedCrossCountsResult",
    "ProjectedCorrelationResult",
    "MarkedProjectedCorrelationResult",
    "proj_auto_counts",
    "proj_cross_counts",
    "pcf",
    "pccf",
    "mpcf",
    "mpccf",
    "plot_cov_matrix",
    "plot_corr_matrix",
    "plot_jk_regions",
    "CompressedPdfGMM",
    "PiMaxEstimate",
    "compress_pdfs_to_gmm",
    "estimate_pi_max_from_pdfs",
    "plot_gmm_for_object",
    "summarize_pair_diagnostics",
    "print_pair_diagnostics",
    "count_auto_fixed_rp",
    "count_cross_fixed_rp",
    "run_auto_mc_pdf_fixed_rp",
]

from .plotting import plot_cov_matrix, plot_corr_matrix, plot_jk_regions
from .marked import AutoMarkSpec, CrossMarkSpec, MarkedProjectedCorrelationResult, mpcf, mpccf

"""Public projected-correlation API and data models."""
from .api import pcf, pccf, proj_auto_counts, proj_cross_counts
from .pdf_tools import CompressedPdfGMM, compress_pdfs_to_gmm, plot_gmm_for_object
from .diagnostics import summarize_pair_diagnostics, print_pair_diagnostics
from .fixed_rp_diagnostic import count_auto_fixed_rp, count_cross_fixed_rp, run_auto_mc_pdf_fixed_rp
from .models import (
    BootstrapSpec,
    JackknifeSpec,
    ProjectedAutoConfig,
    ProjectedCrossConfig,
    ProjectedAutoCountsConfig,
    ProjectedCrossCountsConfig,
    ProjectedBinning,
    ProjectedGridSpec,
    ProjectedCatalogColumns,
    SplitRandomSpec,
    DistanceSpec,
    PDFSourceSpec,
    ProjectedPdfSourceSpec,
    ProjectedMCPdfSpec,
    ProjectedAutoCounts,
    ProjectedCrossCounts,
    ProjectedAutoCountsResult,
    ProjectedCrossCountsResult,
    ProjectedCorrelationResult,
    PreparedProjectedSample,
)

__all__ = [
    "pcf", "pccf", "proj_auto_counts", "proj_cross_counts",
    "BootstrapSpec", "JackknifeSpec",
    "ProjectedAutoConfig", "ProjectedCrossConfig", "ProjectedAutoCountsConfig", "ProjectedCrossCountsConfig",
    "ProjectedBinning", "ProjectedGridSpec", "ProjectedCatalogColumns", "SplitRandomSpec", "DistanceSpec", "PDFSourceSpec", "ProjectedPdfSourceSpec", "ProjectedMCPdfSpec",
    "ProjectedAutoCounts", "ProjectedCrossCounts", "ProjectedAutoCountsResult", "ProjectedCrossCountsResult",
    "ProjectedCorrelationResult", "PreparedProjectedSample",
    "CompressedPdfGMM", "compress_pdfs_to_gmm", "plot_gmm_for_object",
    "summarize_pair_diagnostics", "print_pair_diagnostics",
    "count_auto_fixed_rp", "count_cross_fixed_rp", "run_auto_mc_pdf_fixed_rp",
]


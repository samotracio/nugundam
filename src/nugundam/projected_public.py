"""Flat re-exports for the projected public API."""
from __future__ import annotations

from .projected.api import pcf, pccf, proj_auto_counts, proj_cross_counts
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
]

from .plotting import plot_cov_matrix, plot_corr_matrix, plot_jk_regions
from .marked import AutoMarkSpec, CrossMarkSpec, MarkedProjectedCorrelationResult, mpcf, mpccf

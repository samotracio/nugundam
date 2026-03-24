"""Flat re-exports for the angular public API."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


from .angular.api import acf, accf, ang_auto_counts, ang_cross_counts
from .angular.models import (
    AngularAutoConfig,
    AngularAutoCountsConfig,
    AngularAutoCounts,
    AngularAutoCountsResult,
    AngularBinning,
    AngularCorrelationResult,
    AngularCrossConfig,
    AngularCrossCountsConfig,
    AngularCrossCounts,
    AngularCrossCountsResult,
    AngularGridSpec,
    BootstrapSpec,
    JackknifeSpec,
    CatalogColumns,
    WeightSpec,
    ProgressSpec,
    SplitRandomSpec,
)
from .io import read_result, save_result, write_result
from .marked import AutoMarkSpec, CrossMarkSpec, MarkedAngularCorrelationResult, macf, maccf
from .plotting import plotcf, plotcf2d, plot_compare_ratio, plot_result, plot_result2d, plot_cov_matrix, plot_corr_matrix, plot_jk_regions

__all__ = [
    "AngularAutoConfig", "AngularCrossConfig", "AngularAutoCountsConfig", "AngularCrossCountsConfig", "AngularBinning", "AngularGridSpec", "WeightSpec", "BootstrapSpec", "JackknifeSpec", "ProgressSpec", "SplitRandomSpec", "CatalogColumns", "AutoMarkSpec", "CrossMarkSpec", "AngularAutoCounts", "AngularCrossCounts", "AngularAutoCountsResult", "AngularCrossCountsResult", "AngularCorrelationResult", "MarkedAngularCorrelationResult", "ang_auto_counts", "ang_cross_counts", "acf", "accf", "macf", "maccf", "save_result", "write_result", "read_result", "plotcf", "plotcf2d", "plot_result", "plot_result2d", "plot_cov_matrix", "plot_corr_matrix", "plot_jk_regions", "plot_compare_ratio", "result_to_dict",
]


def result_to_dict(result: Any) -> dict[str, Any]:
    """
    Convert a result object into a plain nested dictionary.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    raise TypeError(f"Unsupported result type: {type(result)!r}")



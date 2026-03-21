"""ASCII export helpers for nuGUNDAM result and count objects."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .angular.models import (
    AngularAutoCounts,
    AngularAutoCountsResult,
    AngularCrossCounts,
    AngularCrossCountsResult,
    AngularCorrelationResult,
)
from .projected.models import (
    ProjectedAutoCounts,
    ProjectedAutoCountsResult,
    ProjectedCrossCounts,
    ProjectedCrossCountsResult,
    ProjectedCorrelationResult,
)


def _is_projected_cf_result(obj: Any) -> bool:
    """
    Return whether projected cf result holds for the supplied object.
    
    Parameters
    ----------
    obj : object
        Value for ``obj``.
    
    Returns
    -------
    bool
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    return isinstance(obj, ProjectedCorrelationResult) or (hasattr(obj, "wp") and hasattr(obj, "rp_centers"))


def _is_cf_result(obj: Any) -> bool:
    """
    Return whether cf result holds for the supplied object.
    
    Parameters
    ----------
    obj : object
        Value for ``obj``.
    
    Returns
    -------
    bool
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    return isinstance(obj, AngularCorrelationResult) or (hasattr(obj, "wtheta") and hasattr(obj, "theta_centers")) or _is_projected_cf_result(obj)


def _is_projected_auto_counts(obj: Any) -> bool:
    """
    Return whether projected auto counts holds for the supplied object.
    
    Parameters
    ----------
    obj : object
        Value for ``obj``.
    
    Returns
    -------
    bool
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    return isinstance(obj, (ProjectedAutoCounts, ProjectedAutoCountsResult)) or (hasattr(obj, "dd") and hasattr(obj, "rp_centers") and not hasattr(obj, "wp"))


def _is_projected_cross_counts(obj: Any) -> bool:
    """
    Return whether projected cross counts holds for the supplied object.
    
    Parameters
    ----------
    obj : object
        Value for ``obj``.
    
    Returns
    -------
    bool
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    return isinstance(obj, (ProjectedCrossCounts, ProjectedCrossCountsResult)) or (hasattr(obj, "d1d2") and hasattr(obj, "rp_centers") and not hasattr(obj, "wp"))


def _is_auto_counts(obj: Any) -> bool:
    """
    Return whether auto counts holds for the supplied object.
    
    Parameters
    ----------
    obj : object
        Value for ``obj``.
    
    Returns
    -------
    bool
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    return isinstance(obj, (AngularAutoCounts, AngularAutoCountsResult)) or (hasattr(obj, "dd") and hasattr(obj, "theta_centers") and not hasattr(obj, "wtheta")) or _is_projected_auto_counts(obj)


def _is_cross_counts(obj: Any) -> bool:
    """
    Return whether cross counts holds for the supplied object.
    
    Parameters
    ----------
    obj : object
        Value for ``obj``.
    
    Returns
    -------
    bool
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    return isinstance(obj, (AngularCrossCounts, AngularCrossCountsResult)) or (hasattr(obj, "d1d2") and hasattr(obj, "theta_centers") and not hasattr(obj, "wtheta")) or _is_projected_cross_counts(obj)


def _get_estimator(obj: Any) -> str | None:
    """
    Return estimator.
    
    Parameters
    ----------
    obj : object
        Value for ``obj``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    estimator = getattr(obj, 'estimator', None)
    if estimator is not None:
        return str(estimator)
    metadata = getattr(obj, 'metadata', {}) or {}
    cfg = metadata.get('config', {}) if isinstance(metadata, dict) else {}
    if isinstance(cfg, dict):
        est = cfg.get('estimator')
        return None if est is None else str(est)
    return None


def _available_count_columns(counts: Any) -> list[str]:
    """
    Available count columns.
    
    Parameters
    ----------
    counts : object
        Value for ``counts``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if _is_projected_auto_counts(counts):
        names = ['intpi_dd', 'intpi_rr', 'intpi_dr']
    elif _is_projected_cross_counts(counts):
        names = ['intpi_d1d2', 'intpi_d1r2', 'intpi_r1d2', 'intpi_r1r2']
    elif _is_auto_counts(counts):
        names = ['dd', 'rr', 'dr']
    elif _is_cross_counts(counts):
        names = ['d1d2', 'd1r2', 'r1d2', 'r1r2']
    else:
        return []
    return [name for name in names if getattr(counts, name, None) is not None]


def default_ascii_columns(result: Any) -> list[str]:
    """
    Choose a sensible default set of ASCII-export columns for a result object.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    estimator = (_get_estimator(result) or '').upper()
    if _is_cf_result(result):
        counts = result.counts
        if _is_projected_cf_result(result):
            if _is_projected_auto_counts(counts):
                by_est = {
                    'NAT': ['intpi_dd', 'intpi_rr'],
                    'DP': ['intpi_dd', 'intpi_dr'],
                    'LS': ['intpi_dd', 'intpi_rr', 'intpi_dr'],
                }
            elif _is_projected_cross_counts(counts):
                by_est = {
                    'NAT': ['intpi_d1d2', 'intpi_r1r2'],
                    'DP': ['intpi_d1d2', 'intpi_d1r2'],
                    'LS': ['intpi_d1d2', 'intpi_d1r2', 'intpi_r1d2', 'intpi_r1r2'],
                }
            else:
                by_est = {}
            cols = ['rp_centers']
            cols.extend([name for name in by_est.get(estimator, _available_count_columns(counts)) if getattr(counts, name, None) is not None])
            cols.extend(['wp', 'wp_err'])
            return cols
        if _is_auto_counts(counts):
            by_est = {
                'NAT': ['dd', 'rr'],
                'DP': ['dd', 'dr'],
                'LS': ['dd', 'rr', 'dr'],
            }
        elif _is_cross_counts(counts):
            by_est = {
                'NAT': ['d1d2', 'r1r2'],
                'DP': ['d1d2', 'd1r2'],
                'LS': ['d1d2', 'd1r2', 'r1d2', 'r1r2'],
            }
        else:
            by_est = {}
        cols = ['theta_centers']
        cols.extend([name for name in by_est.get(estimator, _available_count_columns(counts)) if getattr(counts, name, None) is not None])
        cols.extend(['wtheta', 'wtheta_err'])
        return cols
    if _is_auto_counts(result):
        if _is_projected_auto_counts(result):
            by_est = {
                'NAT': ['intpi_dd'],
                'DP': ['intpi_dd', 'intpi_dr'],
                'LS': ['intpi_dd', 'intpi_rr', 'intpi_dr'],
            }
            cols = ['rp_centers']
        else:
            by_est = {
                'NAT': ['dd'],
                'DP': ['dd', 'dr'],
                'LS': ['dd', 'rr', 'dr'],
            }
            cols = ['theta_centers']
        cols.extend([name for name in by_est.get(estimator, _available_count_columns(result)) if getattr(result, name, None) is not None])
        return cols
    if _is_cross_counts(result):
        if _is_projected_cross_counts(result):
            by_est = {
                'NAT': ['intpi_d1d2'],
                'DP': ['intpi_d1d2', 'intpi_d1r2'],
                'LS': ['intpi_d1d2', 'intpi_d1r2', 'intpi_r1d2', 'intpi_r1r2'],
            }
            cols = ['rp_centers']
        else:
            by_est = {
                'NAT': ['d1d2'],
                'DP': ['d1d2', 'd1r2'],
                'LS': ['d1d2', 'd1r2', 'r1d2', 'r1r2'],
            }
            cols = ['theta_centers']
        cols.extend([name for name in by_est.get(estimator, _available_count_columns(result)) if getattr(result, name, None) is not None])
        return cols
    raise TypeError(f"Unsupported result type for ASCII export: {type(result)!r}")


def _resolve_column(result: Any, col: str) -> np.ndarray:
    """
    Resolve column.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    col : object
        Value for ``col``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if hasattr(result, col):
        value = getattr(result, col)
    elif _is_cf_result(result) and hasattr(result.counts, col):
        value = getattr(result.counts, col)
    else:
        raise AttributeError(f"Column {col!r} not found in result or nested counts")
    if value is None:
        raise ValueError(f"Column {col!r} is not available in this result")
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"Column {col!r} must be 1-D for ASCII export; got shape {arr.shape}")
    return arr


def write_ascii(result: Any, path: str | Path, cols: Iterable[str] | None = None) -> Path:
    """
    Write a 1D ASCII table representation of a result or count object.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    path : object
        Value for ``path``.
    cols : object, optional
        Value for ``cols``.
    
    Returns
    -------
    None
        Object returned by this helper.
    """
    path = Path(path)
    columns = list(default_ascii_columns(result) if cols is None else cols)
    if not columns:
        raise ValueError('No columns selected for ASCII export')
    arrays = [_resolve_column(result, name) for name in columns]
    nrows = len(arrays[0])
    for name, arr in zip(columns, arrays):
        if len(arr) != nrows:
            raise ValueError(f"Column {name!r} has length {len(arr)} but expected {nrows}")
    data = np.column_stack(arrays)
    np.savetxt(path, data, header=' '.join(columns))
    return path

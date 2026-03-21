"""Projected estimator formulas and uncertainty propagation helpers."""
from __future__ import annotations

import numpy as np

from .models import ProjectedAutoCounts, ProjectedCrossCounts, ProjectedCorrelationResult


def _boot_std(values):
    """
    Boot std.
    
    Parameters
    ----------
    values : object
        Value for ``values``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if values is None or values.size == 0:
        return 0.0
    return float(np.std(values))


def _integrate_bootstrap(xi_boot: np.ndarray, pi_delta: np.ndarray) -> np.ndarray:
    """
    Integrate bootstrap.
    
    Parameters
    ----------
    xi_boot : object
        Value for ``xi_boot``.
    pi_delta : object
        Value for ``pi_delta``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    return 2.0 * np.sum(xi_boot * np.asarray(pi_delta, dtype=np.float64)[None, :, None], axis=1)


def _normalize_auto_unweighted(counts: ProjectedAutoCounts):
    """
    Normalize auto unweighted.
    
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
    n_data = float(counts.metadata["n_data"])
    n_random = float(counts.metadata["n_random"])
    dd_n = counts.dd / (0.5 * n_data * (n_data - 1.0))
    rr_norm = float(counts.metadata.get("rr_norm_pairs", 0.5 * n_random * (n_random - 1.0)))
    rr_n = counts.rr / rr_norm if counts.rr is not None and rr_norm > 0 else None
    dr_n = counts.dr / (n_data * n_random) if counts.dr is not None else None
    bdd_n = counts.dd_boot / (0.5 * n_data * (n_data - 1.0)) if counts.dd_boot is not None and counts.dd_boot.size > 0 else None
    return dd_n, rr_n, dr_n, bdd_n


def _normalize_auto_weighted(counts: ProjectedAutoCounts, data_weights: np.ndarray):
    """
    Normalize auto weighted.
    
    Parameters
    ----------
    counts : object
        Value for ``counts``.
    data_weights : object
        Value for ``data_weights``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    wD = np.asarray(data_weights, dtype=np.float64)
    sum_wD = wD.sum()
    sum_wD2 = np.square(wD).sum()
    norm_DD = 0.5 * (sum_wD * sum_wD - sum_wD2)
    norm_DR = sum_wD * float(counts.metadata["n_random"])
    norm_RR = float(counts.metadata.get("rr_norm_pairs", 0.5 * counts.metadata["n_random"] * (counts.metadata["n_random"] - 1.0)))
    dd_n = counts.dd / norm_DD if norm_DD > 0 else np.zeros_like(counts.dd)
    rr_n = counts.rr / norm_RR if counts.rr is not None else None
    dr_n = counts.dr / norm_DR if counts.dr is not None and norm_DR > 0 else None
    bdd_n = None
    if counts.dd_boot is not None and counts.dd_boot.size > 0:
        bdd_n = np.zeros_like(counts.dd_boot, dtype=np.float64)
        if counts.norm_dd_boot is not None:
            for j in range(counts.dd_boot.shape[2]):
                nrm = counts.norm_dd_boot[j]
                if nrm > 0:
                    bdd_n[:, :, j] = counts.dd_boot[:, :, j] / nrm
        elif norm_DD > 0:
            bdd_n = counts.dd_boot / norm_DD
    return dd_n, rr_n, dr_n, bdd_n


def compute_auto_xi2d(counts: ProjectedAutoCounts, *, estimator: str, data_weights: np.ndarray | None = None, sum_w_data: float | None = None, sum_w2_data: float | None = None) -> np.ndarray:
    """
    Reconstruct the projected auto-correlation field ``xi(r_p, pi)`` from stored counts.
    
    Parameters
    ----------
    counts : object
        Value for ``counts``.
    estimator : object
        Value for ``estimator``. This argument is keyword-only.
    data_weights : object, optional
        Value for ``data_weights``. This argument is keyword-only.
    sum_w_data : object, optional
        Value for ``sum_w_data``. This argument is keyword-only.
    sum_w2_data : object, optional
        Value for ``sum_w2_data``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    if data_weights is None and sum_w_data is None:
        dd_n, rr_n, dr_n, _ = _normalize_auto_unweighted(counts)
    elif data_weights is not None:
        dd_n, rr_n, dr_n, _ = _normalize_auto_weighted(counts, data_weights)
    else:
        n_random = float(counts.metadata["n_random"])
        sw = float(sum_w_data)
        sw2 = float(sum_w2_data) if sum_w2_data is not None else sw
        norm_DD = 0.5 * (sw * sw - sw2)
        norm_DR = sw * n_random
        norm_RR = float(counts.metadata.get("rr_norm_pairs", 0.5 * n_random * (n_random - 1.0)))
        dd_n = counts.dd / norm_DD if norm_DD > 0 else np.zeros_like(counts.dd)
        rr_n = counts.rr / norm_RR if counts.rr is not None else None
        dr_n = counts.dr / norm_DR if counts.dr is not None and norm_DR > 0 else None
    xi2d = np.zeros_like(counts.dd, dtype=np.float64)
    for i in range(counts.dd.shape[0]):
        if estimator == "NAT":
            if rr_n is not None:
                mask = rr_n[i] > 0
                xi2d[i, mask] = dd_n[i, mask] / rr_n[i, mask] - 1.0
        elif estimator == "DP":
            if dr_n is not None:
                mask = dr_n[i] > 0
                xi2d[i, mask] = dd_n[i, mask] / dr_n[i, mask] - 1.0
        elif estimator == "LS":
            if rr_n is not None and dr_n is not None:
                mask = rr_n[i] > 0
                xi2d[i, mask] = (dd_n[i, mask] - 2.0 * dr_n[i, mask] + rr_n[i, mask]) / rr_n[i, mask]
        else:
            raise ValueError(f"Unsupported projected auto estimator: {estimator}")
    return xi2d


def estimate_auto(counts: ProjectedAutoCounts, *, estimator: str, data_weights: np.ndarray | None = None):
    """
    Evaluate the configured auto-correlation estimator from stored count terms.
    
    Parameters
    ----------
    counts : object
        Value for ``counts``.
    estimator : object
        Value for ``estimator``. This argument is keyword-only.
    data_weights : object, optional
        Value for ``data_weights``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    if data_weights is None:
        dd_n, rr_n, dr_n, bdd_n = _normalize_auto_unweighted(counts)
        sum_w_data = None
        sum_w2_data = None
    else:
        dd_n, rr_n, dr_n, bdd_n = _normalize_auto_weighted(counts, data_weights)
        w = np.asarray(data_weights, dtype=np.float64)
        sum_w_data = float(np.sum(w))
        sum_w2_data = float(np.sum(np.square(w)))
    pi_delta = np.asarray(counts.pi_edges[1:] - counts.pi_edges[:-1], dtype=np.float64)
    xi2d = np.zeros_like(counts.dd, dtype=np.float64)
    xierr = np.zeros(counts.dd.shape[0], dtype=np.float64)
    bwp = None
    if bdd_n is not None:
        bxi = np.zeros_like(bdd_n, dtype=np.float64)
    else:
        bxi = None
    for i in range(counts.dd.shape[0]):
        if estimator == "NAT":
            if rr_n is not None:
                mask = rr_n[i] > 0
                xi2d[i, mask] = dd_n[i, mask] / rr_n[i, mask] - 1.0
                if bxi is not None and np.any(mask):
                    bxi[i, mask, :] = bdd_n[i, mask, :] / rr_n[i, mask][:, None] - 1.0
        elif estimator == "DP":
            if dr_n is not None:
                mask = dr_n[i] > 0
                xi2d[i, mask] = dd_n[i, mask] / dr_n[i, mask] - 1.0
                if bxi is not None and np.any(mask):
                    bxi[i, mask, :] = bdd_n[i, mask, :] / dr_n[i, mask][:, None] - 1.0
        elif estimator == "LS":
            if rr_n is not None and dr_n is not None:
                mask = rr_n[i] > 0
                xi2d[i, mask] = (dd_n[i, mask] - 2.0 * dr_n[i, mask] + rr_n[i, mask]) / rr_n[i, mask]
                if bxi is not None and np.any(mask):
                    bxi[i, mask, :] = (bdd_n[i, mask, :] - 2.0 * dr_n[i, mask][:, None] + rr_n[i, mask][:, None]) / rr_n[i, mask][:, None]
        else:
            raise ValueError(f"Unsupported projected auto estimator: {estimator}")
    wp = 2.0 * np.sum(xi2d * pi_delta[None, :], axis=1)
    if bxi is not None:
        bwp = _integrate_bootstrap(bxi, pi_delta)
        xierr = np.std(bwp, axis=1)
    metadata = {"weighted": data_weights is not None}
    if sum_w_data is not None:
        metadata["sum_w_data"] = sum_w_data
        metadata["sum_w2_data"] = sum_w2_data
    return ProjectedCorrelationResult(rp_edges=counts.rp_edges, rp_centers=counts.rp_centers, wp=wp, wp_err=xierr, estimator=estimator, counts=counts, metadata=metadata)


def compute_cross_xi2d(counts: ProjectedCrossCounts, *, estimator: str, sum_w1: float | None = None, sum_w2: float | None = None) -> np.ndarray:
    """
    Reconstruct the projected cross-correlation field ``xi(r_p, pi)`` from stored counts.
    
    Parameters
    ----------
    counts : object
        Value for ``counts``.
    estimator : object
        Value for ``estimator``. This argument is keyword-only.
    sum_w1 : object, optional
        Value for ``sum_w1``. This argument is keyword-only.
    sum_w2 : object, optional
        Value for ``sum_w2``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    n1 = float(counts.metadata["n_data1"])
    nr1 = float(counts.metadata["n_random1"])
    n2 = float(counts.metadata["n_data2"])
    nr2 = float(counts.metadata["n_random2"])
    sw1 = n1 if sum_w1 is None else float(sum_w1)
    sw2 = n2 if sum_w2 is None else float(sum_w2)
    d1d2_n = counts.d1d2 / (sw1 * sw2)
    d1r2_n = counts.d1r2 / (sw1 * nr2) if counts.d1r2 is not None else None
    r1d2_n = counts.r1d2 / (nr1 * sw2) if counts.r1d2 is not None else None
    r1r2_n = counts.r1r2 / (nr1 * nr2) if counts.r1r2 is not None else None
    xi2d = np.zeros_like(counts.d1d2, dtype=np.float64)
    for i in range(counts.d1d2.shape[0]):
        if estimator == "NAT":
            if r1r2_n is not None:
                mask = r1r2_n[i] > 0
                xi2d[i, mask] = d1d2_n[i, mask] / r1r2_n[i, mask] - 1.0
        elif estimator == "DP":
            if d1r2_n is not None:
                mask = d1r2_n[i] > 0
                xi2d[i, mask] = d1d2_n[i, mask] / d1r2_n[i, mask] - 1.0
        elif estimator == "LS":
            if r1r2_n is not None and d1r2_n is not None and r1d2_n is not None:
                mask = r1r2_n[i] > 0
                xi2d[i, mask] = (d1d2_n[i, mask] - d1r2_n[i, mask] - r1d2_n[i, mask] + r1r2_n[i, mask]) / r1r2_n[i, mask]
        else:
            raise ValueError(f"Unsupported projected cross estimator: {estimator}")
    return xi2d


def estimate_cross(counts: ProjectedCrossCounts, *, estimator: str, sum_w1: float | None = None, sum_w2: float | None = None):
    """
    Evaluate the configured cross-correlation estimator from stored count terms.
    
    Parameters
    ----------
    counts : object
        Value for ``counts``.
    estimator : object
        Value for ``estimator``. This argument is keyword-only.
    sum_w1 : object, optional
        Value for ``sum_w1``. This argument is keyword-only.
    sum_w2 : object, optional
        Value for ``sum_w2``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    n1 = float(counts.metadata["n_data1"])
    nr1 = float(counts.metadata["n_random1"])
    n2 = float(counts.metadata["n_data2"])
    nr2 = float(counts.metadata["n_random2"])
    sw1 = n1 if sum_w1 is None else float(sum_w1)
    sw2 = n2 if sum_w2 is None else float(sum_w2)
    d1d2_n = counts.d1d2 / (sw1 * sw2)
    d1r2_n = counts.d1r2 / (sw1 * nr2) if counts.d1r2 is not None else None
    r1d2_n = counts.r1d2 / (nr1 * sw2) if counts.r1d2 is not None else None
    r1r2_n = counts.r1r2 / (nr1 * nr2) if counts.r1r2 is not None else None
    b_d1d2_n = counts.d1d2_boot / (sw1 * sw2) if counts.d1d2_boot is not None and counts.d1d2_boot.size > 0 else None
    b_d1r2_n = counts.d1r2_boot / (sw1 * nr2) if counts.d1r2_boot is not None and counts.d1r2_boot.size > 0 and d1r2_n is not None else None
    pi_delta = np.asarray(counts.pi_edges[1:] - counts.pi_edges[:-1], dtype=np.float64)
    xi2d = np.zeros_like(counts.d1d2, dtype=np.float64)
    wp_err = np.zeros(counts.d1d2.shape[0], dtype=np.float64)
    if b_d1d2_n is not None:
        bxi = np.zeros_like(b_d1d2_n, dtype=np.float64)
    else:
        bxi = None
    for i in range(counts.d1d2.shape[0]):
        if estimator == "NAT":
            if r1r2_n is not None:
                mask = r1r2_n[i] > 0
                xi2d[i, mask] = d1d2_n[i, mask] / r1r2_n[i, mask] - 1.0
                if bxi is not None and np.any(mask):
                    bxi[i, mask, :] = b_d1d2_n[i, mask, :] / r1r2_n[i, mask][:, None] - 1.0
        elif estimator == "DP":
            if d1r2_n is not None:
                mask = d1r2_n[i] > 0
                xi2d[i, mask] = d1d2_n[i, mask] / d1r2_n[i, mask] - 1.0
                if bxi is not None and b_d1r2_n is not None and np.any(mask):
                    bxi[i, mask, :] = b_d1d2_n[i, mask, :] / b_d1r2_n[i, mask, :] - 1.0
        elif estimator == "LS":
            if r1r2_n is not None and d1r2_n is not None and r1d2_n is not None:
                mask = r1r2_n[i] > 0
                xi2d[i, mask] = (d1d2_n[i, mask] - d1r2_n[i, mask] - r1d2_n[i, mask] + r1r2_n[i, mask]) / r1r2_n[i, mask]
                if bxi is not None and b_d1r2_n is not None and np.any(mask):
                    bxi[i, mask, :] = (b_d1d2_n[i, mask, :] - b_d1r2_n[i, mask, :] - r1d2_n[i, mask][:, None] + r1r2_n[i, mask][:, None]) / r1r2_n[i, mask][:, None]
        else:
            raise ValueError(f"Unsupported projected cross estimator: {estimator}")
    wp = 2.0 * np.sum(xi2d * pi_delta[None, :], axis=1)
    if bxi is not None:
        bwp = _integrate_bootstrap(bxi, pi_delta)
        wp_err = np.std(bwp, axis=1)
    metadata = {"weighted": sum_w1 is not None or sum_w2 is not None}
    if sum_w1 is not None:
        metadata["sum_w1"] = float(sum_w1)
    if sum_w2 is not None:
        metadata["sum_w2"] = float(sum_w2)
    return ProjectedCorrelationResult(rp_edges=counts.rp_edges, rp_centers=counts.rp_centers, wp=wp, wp_err=wp_err, estimator=estimator, counts=counts, metadata=metadata)

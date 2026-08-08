"""Quantile compression for deterministic projected PDF integration."""
from __future__ import annotations

import numpy as np


def _normalise_pdf_matrix(matrix: np.ndarray, *, eps: float = 0.0, label: str = "pdf") -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{label} matrix must be two-dimensional.")
    if eps > 0.0:
        arr = arr + float(eps)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    row_sum = np.sum(arr, axis=1)
    if np.any(row_sum <= 0.0):
        raise ValueError(f"{label} rows must sum to a positive value.")
    return arr / row_sum[:, None]


def compress_pdf_quantiles(
    matrix,
    chi_grid,
    *,
    nquant: int = 16,
    chi_edges=None,
    eps: float = 0.0,
    positions: str = "midpoint",
    dtype: str | np.dtype = "float32",
    label: str = "pdf",
):
    """Compress empirical PDFs into equal-probability chi quantiles.

    Parameters
    ----------
    matrix : array-like, shape (nobj, ngrid)
        Per-object empirical PDF rows on a shared grid. Rows are normalized
        internally after optional ``eps`` addition.
    chi_grid : array-like, shape (ngrid,)
        Common chi grid centers. Used directly for center-grid PDFs and for
        validation/metadata in edge-grid mode.
    nquant : int, default=16
        Number of midpoint quantiles per object.
    chi_edges : array-like, optional
        If supplied, the PDF row is treated as piecewise-uniform over these
        chi-bin edges. The matrix must then have ``len(chi_edges)-1`` columns.
    eps : float, default=0
        Additive floor before row normalization.
    positions : {'midpoint'}, default='midpoint'
        Quantile-node convention. Only midpoint nodes ``(q+0.5)/Nq`` are
        currently implemented because they are the equal-weight quadrature
        nodes used by the compiled kernels.
    dtype : str or dtype, default='float32'
        Storage dtype for the returned quantile library.

    Returns
    -------
    qchi : ndarray, shape (nquant, nobj), Fortran-contiguous
        Equal-probability chi quantiles per object.
    qmean, qlo, qhi : ndarray, shape (nobj,)
        Mean proxy and support bounds used by gridding/pruning.
    """
    nquant = int(nquant)
    if nquant <= 0:
        raise ValueError("nquant must be positive.")
    if str(positions).strip().lower() != "midpoint":
        raise ValueError("Only quantile_positions='midpoint' is currently supported.")

    prob = _normalise_pdf_matrix(matrix, eps=float(eps), label=label)
    chi_grid = np.asarray(chi_grid, dtype=np.float64)
    if chi_grid.ndim != 1:
        raise ValueError("chi_grid must be one-dimensional.")

    targets = (np.arange(nquant, dtype=np.float64) + 0.5) / float(nquant)
    qout = np.empty((prob.shape[0], nquant), dtype=np.float64)

    if chi_edges is not None:
        edges = np.asarray(chi_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("chi_edges must be one-dimensional with at least two entries.")
        if prob.shape[1] != edges.size - 1:
            raise ValueError(
                f"{label} matrix column count {prob.shape[1]} does not match chi_edges length {edges.size} "
                f"(expected {edges.size - 1})."
            )
        widths = np.diff(edges)
        if np.any(widths <= 0.0):
            raise ValueError("chi_edges must be strictly increasing.")
        for i in range(prob.shape[0]):
            cdf = np.cumsum(prob[i])
            bins = np.searchsorted(cdf, targets, side="left")
            bins = np.clip(bins, 0, prob.shape[1] - 1)
            prev = np.where(bins > 0, cdf[bins - 1], 0.0)
            mass = np.maximum(prob[i, bins], 1.0e-300)
            frac = np.clip((targets - prev) / mass, 0.0, 1.0)
            qout[i] = edges[bins] + frac * widths[bins]
    else:
        if prob.shape[1] != chi_grid.size:
            raise ValueError(
                f"{label} matrix column count {prob.shape[1]} does not match chi_grid length {chi_grid.size}."
            )
        if chi_grid.size == 1:
            qout[:] = chi_grid[0]
        else:
            # Discrete/center-grid fallback: interpolate inverse CDF on the
            # supplied centers. This is intentionally simple; edge grids are the
            # preferred route for quantization-safe empirical PDFs.
            for i in range(prob.shape[0]):
                cdf = np.cumsum(prob[i])
                xcdf = np.concatenate(([0.0], cdf))
                xchi = np.concatenate(([chi_grid[0]], chi_grid))
                qout[i] = np.interp(targets, xcdf, xchi)

    qmean = np.asarray(np.mean(qout, axis=1), dtype=np.float64)
    qlo = np.asarray(np.min(qout, axis=1), dtype=np.float64)
    qhi = np.asarray(np.max(qout, axis=1), dtype=np.float64)
    qlo = np.maximum(qlo, 1.0e-9)
    qhi = np.maximum(qhi, qlo)
    qchi = np.asarray(qout.T, dtype=np.dtype(dtype), order="F")
    return qchi, qmean, qlo, qhi

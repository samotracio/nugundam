"""Fast compression of common-grid empirical PDFs into small chi-space GMMs."""
from __future__ import annotations

import numpy as np


def _normalise_pdf_matrix(pdf_matrix: np.ndarray, *, eps: float) -> np.ndarray:
    p = np.asarray(pdf_matrix, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("pdf_matrix must be a 2-D array with shape (nobj, ngrid).")
    if p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError("pdf_matrix cannot be empty.")
    if np.any(p < 0.0):
        raise ValueError("pdf_matrix must be non-negative.")
    if eps > 0.0:
        p = p + float(eps)
    row_sum = np.asarray(p.sum(axis=1), dtype=np.float64)
    if np.any(row_sum <= 0.0):
        raise ValueError("pdf_matrix rows must have strictly positive total probability.")
    return np.asarray(p / row_sum[:, None], dtype=np.float64)


def _uniform_interval_moments(a: float, b: float, mass: float) -> tuple[float, float, float]:
    """Return mass, first raw moment, second raw moment for uniform [a,b]."""
    if mass <= 0.0 or b <= a:
        return 0.0, 0.0, 0.0
    m1 = mass * 0.5 * (a + b)
    m2 = mass * (a * a + a * b + b * b) / 3.0
    return mass, m1, m2


def _compress_pdf_segments_edge_moments(
    p: np.ndarray,
    chi_edges: np.ndarray,
    *,
    k: int,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-probability segment compression for histogram PDFs.

    The probability in each input bin is treated as uniformly distributed
    between the supplied chi edges. Segment boundaries may cut through input
    bins; truncated-bin moments are included analytically. This prevents the
    GMM from inheriting a radial comb/quantization from coarse PDF grids.
    """
    edges = np.asarray(chi_edges, dtype=np.float64)
    if edges.ndim != 1:
        raise ValueError("chi_edges must be one-dimensional when edge_moments=True.")
    if edges.size != p.shape[1] + 1:
        raise ValueError("chi_edges must have length pdf_matrix.shape[1] + 1 when edge_moments=True.")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("chi_edges must be strictly increasing when edge_moments=True.")

    nobj, ngrid = p.shape
    alpha = np.zeros((k, nobj), dtype=np.float64)
    mu = np.zeros((k, nobj), dtype=np.float64)
    sig = np.full((k, nobj), float(sigma_floor), dtype=np.float64)
    floor2 = float(sigma_floor) ** 2

    # Row-wise loop is intentionally simple and robust. k is normally small
    # (1--3) and ngrid is modest (typically a few hundred).
    for i in range(nobj):
        row = np.asarray(p[i], dtype=np.float64)
        cdf = np.cumsum(row)
        cdf[-1] = 1.0
        for ik in range(k):
            t0 = float(ik) / float(k)
            t1 = float(ik + 1) / float(k)
            if t1 <= t0:
                continue

            b0 = int(np.searchsorted(cdf, t0, side="right"))
            b1 = int(np.searchsorted(cdf, t1, side="left"))
            if b0 >= ngrid:
                b0 = ngrid - 1
            if b1 >= ngrid:
                b1 = ngrid - 1

            mass_tot = 0.0
            m1_tot = 0.0
            m2_tot = 0.0

            for ib in range(b0, b1 + 1):
                pbin = float(row[ib])
                if pbin <= 0.0:
                    continue
                cprev = 0.0 if ib == 0 else float(cdf[ib - 1])
                cnext = float(cdf[ib])
                plo = max(t0, cprev)
                phi = min(t1, cnext)
                if phi <= plo:
                    continue
                frac_lo = (plo - cprev) / pbin
                frac_hi = (phi - cprev) / pbin
                frac_lo = min(1.0, max(0.0, frac_lo))
                frac_hi = min(1.0, max(0.0, frac_hi))
                a = float(edges[ib] + frac_lo * (edges[ib + 1] - edges[ib]))
                b = float(edges[ib] + frac_hi * (edges[ib + 1] - edges[ib]))
                mass = float(phi - plo)
                ma, m1, m2 = _uniform_interval_moments(a, b, mass)
                mass_tot += ma
                m1_tot += m1
                m2_tot += m2

            if mass_tot > 0.0:
                alpha[ik, i] = mass_tot
                mui = m1_tot / mass_tot
                vari = max(m2_tot / mass_tot - mui * mui, floor2)
                mu[ik, i] = mui
                sig[ik, i] = np.sqrt(vari)
            else:
                # This should only happen for pathological rows after numeric
                # clipping. Put empty components at the nearest grid edge with
                # zero weight; final alpha renormalization will ignore them.
                alpha[ik, i] = 0.0
                mu[ik, i] = edges[0] if ik == 0 else edges[-1]
                sig[ik, i] = float(sigma_floor)

    asum = np.sum(alpha, axis=0)
    bad = asum <= 0.0
    if np.any(bad):
        raise ValueError("Compressed GMM alpha arrays must sum to > 0 for all objects.")
    alpha = np.asarray(alpha / asum[None, :], dtype=np.float64, order="F")
    mu = np.asarray(mu, dtype=np.float64, order="F")
    sig = np.asarray(sig, dtype=np.float64, order="F")
    return alpha, mu, sig


def _compress_pdf_segments_center_moments(
    p: np.ndarray,
    chi_grid: np.ndarray,
    *,
    k: int,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original equal-mass segment compressor using point masses at centers."""
    chi = np.asarray(chi_grid, dtype=np.float64)
    if chi.ndim != 1:
        raise ValueError("chi_grid must be one-dimensional.")
    if p.shape[1] != chi.size:
        raise ValueError("pdf_matrix column count must match the length of chi_grid.")

    nobj, ngrid = p.shape
    rows = np.arange(nobj, dtype=np.int64)
    cdf = np.cumsum(p, axis=1)
    if ngrid:
        cdf[:, -1] = 1.0

    cuts = []
    for j in range(1, k):
        frac = float(j) / float(k)
        cuts.append(np.argmax(cdf >= frac, axis=1).astype(np.int64))

    starts = [np.zeros(nobj, dtype=np.int64)]
    ends = []
    for cut in cuts:
        ends.append(np.asarray(cut, dtype=np.int64))
        starts.append(np.asarray(cut + 1, dtype=np.int64))
    ends.append(np.full(nobj, ngrid - 1, dtype=np.int64))

    P = np.cumsum(p, axis=1)
    Pc = np.cumsum(p * chi[None, :], axis=1)
    Pc2 = np.cumsum(p * (chi[None, :] ** 2), axis=1)

    def seg(prefix: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        out = np.zeros(nobj, dtype=np.float64)
        valid = hi >= lo
        if not np.any(valid):
            return out
        hi_v = hi[valid]
        lo_v = lo[valid]
        out_v = prefix[rows[valid], hi_v].astype(np.float64, copy=False)
        sub = lo_v > 0
        if np.any(sub):
            out_v[sub] -= prefix[rows[valid][sub], lo_v[sub] - 1]
        out[valid] = out_v
        return out

    alpha_list = []
    mu_list = []
    sig_list = []
    chi_lo = float(chi[0]) if chi.size else 0.0
    chi_hi = float(chi[-1]) if chi.size else 0.0
    for ik, (lo, hi) in enumerate(zip(starts, ends)):
        lo = np.asarray(lo, dtype=np.int64)
        hi = np.asarray(hi, dtype=np.int64)
        hi = np.maximum(hi, np.minimum(lo, ngrid - 1))
        a = seg(P, lo, hi)
        m = seg(Pc, lo, hi)
        s2 = seg(Pc2, lo, hi)
        mui = np.empty(nobj, dtype=np.float64)
        sigi = np.empty(nobj, dtype=np.float64)
        ok = a > 0.0
        mui[ok] = m[ok] / a[ok]
        var = np.maximum(s2[ok] / a[ok] - mui[ok] ** 2, float(sigma_floor) ** 2)
        sigi[ok] = np.sqrt(var)
        mui[~ok] = chi_lo if ik == 0 else chi_hi
        sigi[~ok] = float(sigma_floor)
        alpha_list.append(a)
        mu_list.append(mui)
        sig_list.append(sigi)

    alpha = np.asarray(alpha_list, dtype=np.float64)
    asum = np.sum(alpha, axis=0)
    bad = asum <= 0.0
    if np.any(bad):
        raise ValueError("Compressed GMM alpha arrays must sum to > 0 for all objects.")
    alpha = np.asarray(alpha / asum[None, :], dtype=np.float64, order="F")
    mu = np.asarray(mu_list, dtype=np.float64, order="F")
    sig = np.asarray(sig_list, dtype=np.float64, order="F")
    return alpha, mu, sig


def compress_pdf_segments(
    pdf_matrix: np.ndarray,
    chi_grid: np.ndarray,
    *,
    k: int,
    compressor: str = "segments_equal_mass",
    eps: float = 0.0,
    sigma_floor: float = 1.0e-6,
    chi_edges: np.ndarray | None = None,
    edge_moments: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress empirical PDFs into an equal-mass segment GMM.

    Parameters
    ----------
    pdf_matrix : ndarray
        Normalized empirical PDF matrix with shape ``(nobj, ngrid)``.
    chi_grid : ndarray
        Shared one-dimensional chi grid centers aligned with ``pdf_matrix``.
        This is used for the traditional point-at-center compression and is
        also stored/validated by callers when ``edge_moments=True``.
    k : int
        Number of Gaussian components to produce per object.
    compressor : str, default='segments_equal_mass'
        Compression scheme. Currently only ``'segments_equal_mass'`` is
        implemented.
    eps : float, default=0.0
        Optional additive floor applied before row renormalization.
    sigma_floor : float, default=1e-6
        Minimum component width enforced for degenerate or empty segments.
    chi_edges : ndarray, optional
        Chi bin edges with length ``ngrid + 1``. Required when
        ``edge_moments=True``.
    edge_moments : bool, default=False
        If True, treat each input PDF value as probability mass uniformly
        distributed inside its chi bin instead of as a delta-function at the
        bin center. Segment boundaries are allowed to cut through bins.

    Returns
    -------
    alpha, mu, sig : tuple of ndarray
        Mixture weights, means, and sigmas with shape ``(k, nobj)``.
    """
    if str(compressor).strip().lower() != "segments_equal_mass":
        raise ValueError("Only compressor='segments_equal_mass' is currently supported.")
    k = int(k)
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if sigma_floor <= 0.0:
        raise ValueError("sigma_floor must be positive.")

    p = _normalise_pdf_matrix(pdf_matrix, eps=float(eps))
    chi = np.asarray(chi_grid, dtype=np.float64)
    if chi.ndim != 1:
        raise ValueError("chi_grid must be one-dimensional.")
    if p.shape[1] != chi.size:
        raise ValueError("pdf_matrix column count must match the length of chi_grid.")

    if bool(edge_moments):
        if chi_edges is None:
            raise ValueError("chi_edges must be provided when edge_moments=True.")
        return _compress_pdf_segments_edge_moments(
            p,
            np.asarray(chi_edges, dtype=np.float64),
            k=k,
            sigma_floor=float(sigma_floor),
        )

    return _compress_pdf_segments_center_moments(
        p,
        chi,
        k=k,
        sigma_floor=float(sigma_floor),
    )

"""Diagnostic MC projected correlation with transverse distances held fixed.

This module is intentionally separate from the production MC implementation.
It is designed to test whether Monte-Carlo redshift sampling moves pairs between
``r_p`` bins.  For every pair it uses

``pi = abs(chi_mc_i - chi_mc_j)``

but

``r_p^2 = chi_ref_i * chi_ref_j * |u_i - u_j|^2``,

where ``chi_ref`` is normally obtained from the spectroscopic/true-redshift
column and ``u`` is the sky-direction unit vector.

The full DD, DR, and RR terms use the same hybrid geometry, so Landy--Szalay,
Davis--Peebles, and natural estimators remain internally consistent.  This is a
diagnostic coordinate construction, not a physical estimator intended for
science measurements.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .estimators import estimate_auto
from .models import ProjectedAutoConfig, ProjectedAutoCounts
from .pdf_common import resolve_common_chi_grid
from .prepare import _distance_array, _prepared_projected_from_arrays
from .mc_pdf import (
    _auto_meta,
    _build_cdf,
    _catalog_template,
    _grid_meta,
    _grid_tuple,
    _load_pdf_matrix,
    _mean_pdf,
    _resolve_sample_edges_chi,
    _sample_dist_from_cdf,
    _sample_random_dist,
    _validate_mc_pdf_config,
)

try:  # pragma: no cover - import path depends on optional dependency
    from numba import config as numba_config
    from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads
except Exception:  # pragma: no cover
    numba_config = None
    get_num_threads = get_thread_id = njit = prange = set_num_threads = None


_DEG2RAD = np.pi / 180.0
_RAD2DEG = 180.0 / np.pi


if njit is not None:

    @njit(cache=True, inline="always")
    def _cell_index(value: float, lo: float, width: float, ncell: int) -> int:
        q = int((value - lo) / width)
        if q < 0 or q >= ncell:
            return -1
        return q


    @njit(cache=True, inline="always")
    def _wrap_ra_cell(q: int, ncell: int) -> int:
        while q < 0:
            q += ncell
        while q >= ncell:
            q -= ncell
        return q


    @njit(cache=True, inline="always")
    def _dalp(stm2: float, dec_deg: float, dec_rad: float, jq1: int,
              hc1: float, dec_lo: float, same_dec_cell: bool) -> float:
        """Numba translation of the Fortran ``dalp`` helper, in degrees."""
        band_lo = dec_lo + jq1 * hc1
        band_hi = band_lo + hc1
        cmin = min(np.cos(band_lo * _DEG2RAD), np.cos(band_hi * _DEG2RAD))
        if cmin <= 0.0:
            return 180.0
        denom = np.cos(dec_rad) * cmin
        if denom <= 0.0:
            return 180.0
        if same_dec_cell:
            s2 = stm2 / np.sqrt(denom)
        else:
            a = np.sin(0.5 * (dec_deg - band_lo) * _DEG2RAD) ** 2
            b = np.sin(0.5 * (dec_deg - band_hi) * _DEG2RAD) ** 2
            term = (stm2 * stm2 - min(a, b)) / denom
            if term < 0.0:
                term = 0.0
            s2 = np.sqrt(term)
        if s2 >= 1.0:
            return 180.0
        return 2.0 * np.arcsin(s2) * _RAD2DEG


    @njit(cache=True, inline="always")
    def _rp_bin(rp2: float, rp2_edges: np.ndarray) -> int:
        # Match the production counters: lower edge is exclusive, and pairs
        # below the first lower edge are not counted.
        n = rp2_edges.size - 1
        if rp2 <= rp2_edges[0] or rp2 > rp2_edges[n]:
            return -1
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if rp2 > rp2_edges[mid + 1]:
                lo = mid + 1
            else:
                hi = mid
        return lo


    @njit(cache=True)
    def _layer_min_reference_distance(
        dist_pi: np.ndarray,
        dist_rp: np.ndarray,
        dmin: float,
        hc3: float,
        nc3: int,
    ) -> np.ndarray:
        out = np.full(nc3, np.inf, dtype=np.float64)
        for i in range(dist_pi.size):
            q = _cell_index(dist_pi[i], dmin, hc3, nc3)
            if q >= 0 and dist_rp[i] > 0.0 and dist_rp[i] < out[q]:
                out[q] = dist_rp[i]
        return out


    @njit(cache=False, parallel=True)
    def _count_cross_fixed_rp_impl(
        ra_l: np.ndarray,
        dec_l: np.ndarray,
        dist_pi_l: np.ndarray,
        dist_rp_l: np.ndarray,
        weights_l: np.ndarray,
        x_l: np.ndarray,
        y_l: np.ndarray,
        z_l: np.ndarray,
        dist_pi_r: np.ndarray,
        dist_rp_r: np.ndarray,
        weights_r: np.ndarray,
        x_r: np.ndarray,
        y_r: np.ndarray,
        z_r: np.ndarray,
        sk_r: np.ndarray,
        ll_r: np.ndarray,
        sbound: np.ndarray,
        rp_edges: np.ndarray,
        pi_edges: np.ndarray,
    ) -> np.ndarray:
        mxh3, mxh2, mxh1 = sk_r.shape
        n_rp = rp_edges.size - 1
        n_pi = pi_edges.size - 1
        rp2_edges = rp_edges * rp_edges
        rpmax = rp_edges[-1]
        pimax = pi_edges[-1]
        dpi = pi_edges[1] - pi_edges[0]

        ra_lo, ra_hi, dec_lo, dec_hi, dmin, dmax = sbound
        hc1 = (dec_hi - dec_lo) / mxh1
        hc2 = (ra_hi - ra_lo) / mxh2
        nc3 = int((dmax - dmin) / max(pimax, 1.0e-12))
        if nc3 < 1:
            nc3 = 1
        if nc3 > mxh3:
            nc3 = mxh3
        hc3 = (dmax - dmin) / nc3
        layer_min = _layer_min_reference_distance(dist_pi_r, dist_rp_r, dmin, hc3, nc3)

        nth = get_num_threads()
        hist = np.zeros((nth, n_rp, n_pi), dtype=np.float64)

        for i in prange(ra_l.size):
            tid = get_thread_id()
            iq1 = _cell_index(dec_l[i], dec_lo, hc1, mxh1)
            iq2 = _cell_index(ra_l[i], ra_lo, hc2, mxh2)
            iq3 = _cell_index(dist_pi_l[i], dmin, hc3, nc3)
            if iq1 < 0 or iq2 < 0 or iq3 < 0 or dist_rp_l[i] <= 0.0:
                continue
            dec_rad = dec_l[i] * _DEG2RAD
            for jq3 in range(max(0, iq3 - 1), min(nc3, iq3 + 2)):
                dref_min = layer_min[jq3]
                if not np.isfinite(dref_min) or dref_min <= 0.0:
                    continue
                # This is the same deliberately conservative angular bound used
                # by the production Fortran code, but evaluated with fixed-rp
                # distances rather than MC distances.
                stm2 = rpmax / np.sqrt(2.0 * dist_rp_l[i] * dref_min)
                if stm2 > 1.0:
                    stm2 = 1.0
                ddec = 2.0 * np.arcsin(stm2) * _RAD2DEG
                jq1m = int(ddec / hc1) + 1
                for jq1 in range(max(0, iq1 - jq1m), min(mxh1, iq1 + jq1m + 1)):
                    dra = _dalp(stm2, dec_l[i], dec_rad, jq1, hc1, dec_lo, jq1 == iq1)
                    jq2m = int(dra / hc2) + 1
                    nscan = min(mxh2, 2 * jq2m + 1)
                    start = iq2 - jq2m
                    for kk in range(nscan):
                        jq2 = _wrap_ra_cell(start + kk, mxh2)
                        j1 = sk_r[jq3, jq2, jq1]
                        while j1 != 0:
                            j = j1 - 1
                            dpi_pair = abs(dist_pi_l[i] - dist_pi_r[j])
                            if dpi_pair < pimax:
                                ang2 = ((x_l[i] - x_r[j]) ** 2 +
                                        (y_l[i] - y_r[j]) ** 2 +
                                        (z_l[i] - z_r[j]) ** 2)
                                rp2 = dist_rp_l[i] * dist_rp_r[j] * ang2
                                irp = _rp_bin(rp2, rp2_edges)
                                if irp >= 0:
                                    ipi = int((dpi_pair - pi_edges[0]) / dpi)
                                    if 0 <= ipi < n_pi:
                                        hist[tid, irp, ipi] += weights_l[i] * weights_r[j]
                            j1 = ll_r[j]
        return np.sum(hist, axis=0)


    @njit(cache=False, parallel=True)
    def _count_auto_fixed_rp_impl(
        ra: np.ndarray,
        dec: np.ndarray,
        dist_pi: np.ndarray,
        dist_rp: np.ndarray,
        weights: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        sk: np.ndarray,
        ll: np.ndarray,
        sbound: np.ndarray,
        rp_edges: np.ndarray,
        pi_edges: np.ndarray,
    ) -> np.ndarray:
        mxh3, mxh2, mxh1 = sk.shape
        n_rp = rp_edges.size - 1
        n_pi = pi_edges.size - 1
        rp2_edges = rp_edges * rp_edges
        rpmax = rp_edges[-1]
        pimax = pi_edges[-1]
        dpi = pi_edges[1] - pi_edges[0]

        ra_lo, ra_hi, dec_lo, dec_hi, dmin, dmax = sbound
        hc1 = (dec_hi - dec_lo) / mxh1
        hc2 = (ra_hi - ra_lo) / mxh2
        nc3 = int((dmax - dmin) / max(pimax, 1.0e-12))
        if nc3 < 1:
            nc3 = 1
        if nc3 > mxh3:
            nc3 = mxh3
        hc3 = (dmax - dmin) / nc3
        layer_min = _layer_min_reference_distance(dist_pi, dist_rp, dmin, hc3, nc3)

        nth = get_num_threads()
        hist = np.zeros((nth, n_rp, n_pi), dtype=np.float64)

        for i in prange(ra.size):
            tid = get_thread_id()
            iq1 = _cell_index(dec[i], dec_lo, hc1, mxh1)
            iq2 = _cell_index(ra[i], ra_lo, hc2, mxh2)
            iq3 = _cell_index(dist_pi[i], dmin, hc3, nc3)
            if iq1 < 0 or iq2 < 0 or iq3 < 0 or dist_rp[i] <= 0.0:
                continue
            dec_rad = dec[i] * _DEG2RAD
            for jq3 in range(max(0, iq3 - 1), min(nc3, iq3 + 2)):
                dref_min = layer_min[jq3]
                if not np.isfinite(dref_min) or dref_min <= 0.0:
                    continue
                stm2 = rpmax / np.sqrt(2.0 * dist_rp[i] * dref_min)
                if stm2 > 1.0:
                    stm2 = 1.0
                ddec = 2.0 * np.arcsin(stm2) * _RAD2DEG
                jq1m = int(ddec / hc1) + 1
                for jq1 in range(max(0, iq1 - jq1m), min(mxh1, iq1 + jq1m + 1)):
                    dra = _dalp(stm2, dec[i], dec_rad, jq1, hc1, dec_lo, jq1 == iq1)
                    jq2m = int(dra / hc2) + 1
                    nscan = min(mxh2, 2 * jq2m + 1)
                    start = iq2 - jq2m
                    for kk in range(nscan):
                        jq2 = _wrap_ra_cell(start + kk, mxh2)
                        j1 = sk[jq3, jq2, jq1]
                        while j1 != 0:
                            j = j1 - 1
                            if j > i:
                                dpi_pair = abs(dist_pi[i] - dist_pi[j])
                                if dpi_pair < pimax:
                                    ang2 = ((x[i] - x[j]) ** 2 +
                                            (y[i] - y[j]) ** 2 +
                                            (z[i] - z[j]) ** 2)
                                    rp2 = dist_rp[i] * dist_rp[j] * ang2
                                    irp = _rp_bin(rp2, rp2_edges)
                                    if irp >= 0:
                                        ipi = int((dpi_pair - pi_edges[0]) / dpi)
                                        if 0 <= ipi < n_pi:
                                            hist[tid, irp, ipi] += weights[i] * weights[j]
                            j1 = ll[j]
        return np.sum(hist, axis=0)

else:  # pragma: no cover
    _count_auto_fixed_rp_impl = None
    _count_cross_fixed_rp_impl = None


def _require_numba() -> None:
    if njit is None:
        raise ImportError(
            "The fixed-rp MC diagnostic requires numba. Install nugundam with "
            "its normal runtime dependencies or install numba explicitly."
        )



def _set_requested_numba_threads(nthreads: int) -> None:
    if int(nthreads) <= 0:
        return
    maximum = int(getattr(numba_config, "NUMBA_NUM_THREADS", int(nthreads)))
    set_num_threads(max(1, min(int(nthreads), maximum)))


def _validate_linear_pi_edges(pi_edges: np.ndarray) -> None:
    delta = np.diff(np.asarray(pi_edges, dtype=np.float64))
    if delta.size == 0 or not np.allclose(delta, delta[0]):
        raise ValueError("The fixed-rp diagnostic currently requires uniform pi bins.")
    if abs(float(pi_edges[0])) > 1.0e-12:
        raise ValueError("The fixed-rp diagnostic currently requires pi_edges[0] == 0.")


def count_auto_fixed_rp(sample, *, rp_edges, pi_edges, nthreads: int = 0) -> np.ndarray:
    """Count auto pairs using ``sample.dist`` for pi and ``sample.dcang`` for rp."""
    _require_numba()
    if sample.dcang is None:
        raise ValueError("sample.dcang must contain the fixed/reference transverse distances.")
    rp_edges = np.asarray(rp_edges, dtype=np.float64)
    pi_edges = np.asarray(pi_edges, dtype=np.float64)
    _validate_linear_pi_edges(pi_edges)
    _set_requested_numba_threads(int(nthreads))
    return _count_auto_fixed_rp_impl(
        np.asarray(sample.ra, dtype=np.float64),
        np.asarray(sample.dec, dtype=np.float64),
        np.asarray(sample.dist, dtype=np.float64),
        np.asarray(sample.dcang, dtype=np.float64),
        np.asarray(sample.weights, dtype=np.float64),
        np.asarray(sample.x, dtype=np.float64),
        np.asarray(sample.y, dtype=np.float64),
        np.asarray(sample.z, dtype=np.float64),
        np.asarray(sample.sk, dtype=np.int32),
        np.asarray(sample.ll, dtype=np.int32),
        np.asarray(sample.sbound, dtype=np.float64),
        rp_edges,
        pi_edges,
    )


def count_cross_fixed_rp(left, right, *, rp_edges, pi_edges, nthreads: int = 0) -> np.ndarray:
    """Count cross pairs using MC distances for pi and reference distances for rp."""
    _require_numba()
    if left.dcang is None or right.dcang is None:
        raise ValueError("Both samples must store fixed/reference transverse distances in dcang.")
    rp_edges = np.asarray(rp_edges, dtype=np.float64)
    pi_edges = np.asarray(pi_edges, dtype=np.float64)
    _validate_linear_pi_edges(pi_edges)
    _set_requested_numba_threads(int(nthreads))
    return _count_cross_fixed_rp_impl(
        np.asarray(left.ra, dtype=np.float64),
        np.asarray(left.dec, dtype=np.float64),
        np.asarray(left.dist, dtype=np.float64),
        np.asarray(left.dcang, dtype=np.float64),
        np.asarray(left.weights, dtype=np.float64),
        np.asarray(left.x, dtype=np.float64),
        np.asarray(left.y, dtype=np.float64),
        np.asarray(left.z, dtype=np.float64),
        np.asarray(right.dist, dtype=np.float64),
        np.asarray(right.dcang, dtype=np.float64),
        np.asarray(right.weights, dtype=np.float64),
        np.asarray(right.x, dtype=np.float64),
        np.asarray(right.y, dtype=np.float64),
        np.asarray(right.z, dtype=np.float64),
        np.asarray(right.sk, dtype=np.int32),
        np.asarray(right.ll, dtype=np.int32),
        np.asarray(right.sbound, dtype=np.float64),
        rp_edges,
        pi_edges,
    )


def _build_hybrid_prepared(
    template: dict[str, Any],
    dist_pi: np.ndarray,
    dist_rp: np.ndarray,
    *,
    sbound,
    grid_tuple,
    pi_edges,
    grid_meta,
):
    return _prepared_projected_from_arrays(
        ra=np.asarray(template["ra"], dtype=np.float64),
        dec=np.asarray(template["dec"], dtype=np.float64),
        dist=np.asarray(dist_pi, dtype=np.float64),
        dcang=np.asarray(dist_rp, dtype=np.float64),
        weights=np.asarray(template["weights"], dtype=np.float64),
        sbound=sbound,
        mxh1=int(grid_tuple[0]),
        mxh2=int(grid_tuple[1]),
        mxh3=int(grid_tuple[2]),
        pi_edges=np.asarray(pi_edges, dtype=np.float64),
        grid_meta=grid_meta,
        sort_rows=True,
    )


def _mean_counts(
    dd_sum: np.ndarray,
    rr_sum: np.ndarray | None,
    dr_sum: np.ndarray | None,
    *,
    nreal: int,
    meta: dict[str, Any],
    n_data: int,
    n_random: int,
    weighted: bool,
) -> ProjectedAutoCounts:
    scale = 1.0 / float(nreal)
    dd = np.asarray(dd_sum, dtype=np.float64) * scale
    rr = None if rr_sum is None else np.asarray(rr_sum, dtype=np.float64) * scale
    dr = None if dr_sum is None else np.asarray(dr_sum, dtype=np.float64) * scale
    pi_delta = np.asarray(meta["pi_delta"], dtype=np.float64)
    return ProjectedAutoCounts(
        rp_edges=np.asarray(meta["rp_edges"], dtype=np.float64),
        rp_centers=np.asarray(meta["rp_centers"], dtype=np.float64),
        pi_edges=np.asarray(meta["pi_edges"], dtype=np.float64),
        pi_centers=np.asarray(meta["pi_centers"], dtype=np.float64),
        dd=dd,
        rr=rr,
        dr=dr,
        intpi_dd=2.0 * np.sum(dd * pi_delta[None, :], axis=1),
        intpi_rr=None if rr is None else 2.0 * np.sum(rr * pi_delta[None, :], axis=1),
        intpi_dr=None if dr is None else 2.0 * np.sum(dr * pi_delta[None, :], axis=1),
        metadata={
            "n_data": int(n_data),
            "n_random": int(n_random),
            "data_weighted": bool(weighted),
            "rr_norm_pairs": 0.5 * float(n_random) * float(max(n_random - 1, 0)),
            "jk_nregions": 0,
            "jk_touch_available": False,
        },
    )



def _sample_paired_random_hybrid(
    *,
    nrand: int,
    data_cdf: np.ndarray,
    data_dist: np.ndarray,
    data_rp_distance: np.ndarray,
    chi_grid: np.ndarray,
    random_mode: str,
    rng: np.random.Generator,
    sample_edges_chi: np.ndarray | None,
    donor_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a random hybrid catalog preserving ``(chi_ref, chi_MC)``.

    A fixed-rp Landy--Szalay diagnostic needs the random catalog to reproduce
    the *joint* distribution of the reference distance used for ``r_p`` and
    the MC distance used for ``pi``.  Sampling only ``chi_MC`` from the data
    while retaining an unrelated random-catalog ``chi_ref`` breaks that joint
    distribution and biases DR/RR.

    Returns ``(chi_mc_random, chi_ref_random, donor_index)``.
    """
    if nrand == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_f, empty_f, empty_i
    ndata = int(len(data_dist))
    if ndata == 0:
        raise ValueError("The paired fixed-rp random construction requires non-empty data.")
    if donor_index is None:
        donor_index = rng.integers(0, ndata, size=int(nrand), dtype=np.int64)
    else:
        donor_index = np.asarray(donor_index, dtype=np.int64)
        if donor_index.shape != (int(nrand),):
            raise ValueError("donor_index must have one entry per random object.")

    mode = str(random_mode).strip().lower()
    chi_ref_random = np.asarray(data_rp_distance, dtype=np.float64)[donor_index]
    if mode == "inherit_realization":
        chi_mc_random = np.asarray(data_dist, dtype=np.float64)[donor_index]
    elif mode in {"fixed_global", "rerun_global"}:
        # Draw from each donor object's own PDF.  This has the same ensemble
        # marginal as the global mean PDF, but unlike an independent global
        # draw it retains the required correlation with chi_ref.
        chi_mc_random = _sample_dist_from_cdf(
            np.asarray(data_cdf, dtype=np.float64)[donor_index],
            chi_grid,
            rng,
            sample_edges_chi=sample_edges_chi,
        )
    else:
        raise ValueError(
            "mc_pdf.random_mode must be 'fixed_global', 'rerun_global', "
            "or 'inherit_realization'."
        )
    return (
        np.asarray(chi_mc_random, dtype=np.float64),
        np.asarray(chi_ref_random, dtype=np.float64),
        donor_index,
    )


def run_auto_mc_pdf_fixed_rp(
    data,
    random,
    config: ProjectedAutoConfig,
    *,
    data_rp_distance: np.ndarray | None = None,
    random_rp_distance: np.ndarray | None = None,
):
    """Run an auto-correlation MC diagnostic with fixed transverse distances.

    Parameters
    ----------
    data, random
        The same catalogs supplied to :func:`nugundam.pcf`.
    config
        A normal projected auto configuration with ``mc_pdf.enabled=True``.
        Bootstrap, jackknife, and split-random execution are intentionally not
        supported by this diagnostic helper.
    data_rp_distance, random_rp_distance, optional
        Reference comoving distances in Mpc/h used only for ``r_p``.  When
        omitted, they are computed from the catalog columns selected by
        ``config.columns_data`` and ``config.columns_random``.  For the data
        catalog this should be the true/spectroscopic distance.

    Returns
    -------
    ProjectedCorrelationResult
        A normal result object.  Its metadata contains
        ``fixed_rp_diagnostic=True``.

    Notes
    -----
    This hybrid coordinate definition is only a controlled diagnostic.  If its
    small-scale ``w_p`` agrees with the true-redshift result while the normal MC
    result is low, the missing amplitude is strong evidence for migration
    between transverse-separation bins.
    """
    _validate_mc_pdf_config(config, cross=False)
    if bool(config.bootstrap.enabled) or bool(config.jackknife.enabled):
        raise NotImplementedError("Disable bootstrap and jackknife for the fixed-rp diagnostic.")
    if bool(getattr(config.split_random, "enabled", False)):
        raise NotImplementedError("Disable split_random for the fixed-rp diagnostic.")

    spec = config.mc_pdf
    chi_grid = resolve_common_chi_grid(
        z_grid=spec.z_grid,
        chi_grid=spec.chi_grid,
        config=config,
        grid_kind=str(getattr(spec, "grid_kind", "centers")),
        label="mc_pdf",
    )
    sample_edges_chi = _resolve_sample_edges_chi(spec, config)
    support_grid = sample_edges_chi if sample_edges_chi is not None else chi_grid
    meta = _auto_meta(config, data, random, spec.pdf_data, spec.pdf_random, support_grid)
    _validate_linear_pi_edges(np.asarray(meta["pi_edges"], dtype=np.float64))
    sbound = meta["sbound"]
    grid_meta = _grid_meta(config)

    data_template = _catalog_template(
        data, config.columns_data, config,
        use_weights=(config.weights.weight_mode != "unweighted"),
    )
    random_template = _catalog_template(random, config.columns_random, config, use_weights=False)
    data_grid = _grid_tuple(data_template, config, sbound=sbound)
    random_grid = _grid_tuple(random_template, config, sbound=sbound)

    data_reference_source = "catalog" if data_rp_distance is None else "explicit"
    if data_rp_distance is None:
        data_rp_distance = _distance_array(data, config.columns_data, config)
    data_rp_distance = np.asarray(data_rp_distance, dtype=np.float64)
    if data_rp_distance.shape != (data_template["nrows"],):
        raise ValueError("data_rp_distance must have one value per data object.")
    if np.any(~np.isfinite(data_rp_distance)) or np.any(data_rp_distance <= 0.0):
        raise ValueError("data_rp_distance must contain finite positive distances.")

    # With explicit per-random PDFs, each random row has a meaningful joint
    # (reference distance, PDF) and may use its catalog reference distance.
    # Without explicit random PDFs, the reference and MC distances must be
    # inherited as a *paired* draw from data donors; an independently retained
    # random-catalog reference distance biases DR and RR.
    if spec.pdf_random is not None:
        random_reference_source = "catalog" if random_rp_distance is None else "explicit"
        if random_rp_distance is None:
            random_rp_distance = _distance_array(random, config.columns_random, config)
        random_rp_distance = np.asarray(random_rp_distance, dtype=np.float64)
        if random_rp_distance.shape != (random_template["nrows"],):
            raise ValueError("random_rp_distance must have one value per random object.")
        if np.any(~np.isfinite(random_rp_distance)) or np.any(random_rp_distance <= 0.0):
            raise ValueError("random_rp_distance must contain finite positive distances.")
    else:
        random_reference_source = "paired_data_donor"
        random_rp_distance = None

    p_data = _load_pdf_matrix(spec.pdf_data, data, nrows=data_template["nrows"])
    cdf_data = _build_cdf(p_data)
    pbar_data = _mean_pdf(p_data)
    p_random = None if spec.pdf_random is None else _load_pdf_matrix(
        spec.pdf_random, random, nrows=random_template["nrows"]
    )
    cdf_random = None if p_random is None else _build_cdf(p_random)

    estimator = str(config.estimator).upper()
    need_rr = estimator in {"NAT", "LS"}
    need_dr = estimator in {"DP", "LS"}
    rng = np.random.default_rng(int(spec.seed))
    fixed_random = str(spec.random_mode).strip().lower() == "fixed_global"
    fixed_random_prepared = None
    fixed_rr = None
    fixed_random_donor_index = None

    dd_sum = np.zeros((len(meta["rp_centers"]), len(meta["pi_centers"])), dtype=np.float64)
    rr_sum = np.zeros_like(dd_sum) if need_rr else None
    dr_sum = np.zeros_like(dd_sum) if need_dr else None
    wp_realizations = []

    for ireal in range(int(spec.nreal)):
        if bool(getattr(config.progress, "enabled", False)):
            print(
                f"[pcf:mc_pdf:fixed_rp] realization {ireal + 1}/{int(spec.nreal)}",
                flush=True,
            )
        data_dist = _sample_dist_from_cdf(
            cdf_data, chi_grid, rng, sample_edges_chi=sample_edges_chi
        )
        data_p = _build_hybrid_prepared(
            data_template,
            data_dist,
            data_rp_distance,
            sbound=sbound,
            grid_tuple=data_grid,
            pi_edges=meta["pi_edges"],
            grid_meta=grid_meta,
        )

        if fixed_random and fixed_random_prepared is not None:
            random_p = fixed_random_prepared
        else:
            if cdf_random is not None:
                random_dist = _sample_random_dist(
                    random_template["nrows"],
                    random_source_cdf=cdf_random,
                    data_source_mean=pbar_data,
                    data_draw=data_dist,
                    chi_grid=chi_grid,
                    random_mode=spec.random_mode,
                    rng=rng,
                    sample_edges_chi=sample_edges_chi,
                )
                random_rp_this = np.asarray(random_rp_distance, dtype=np.float64)
            else:
                random_dist, random_rp_this, donor_index = _sample_paired_random_hybrid(
                    nrand=random_template["nrows"],
                    data_cdf=cdf_data,
                    data_dist=data_dist,
                    data_rp_distance=data_rp_distance,
                    chi_grid=chi_grid,
                    random_mode=spec.random_mode,
                    rng=rng,
                    sample_edges_chi=sample_edges_chi,
                    donor_index=fixed_random_donor_index if fixed_random else None,
                )
                if fixed_random and fixed_random_donor_index is None:
                    fixed_random_donor_index = donor_index.copy()
            random_p = _build_hybrid_prepared(
                random_template,
                random_dist,
                random_rp_this,
                sbound=sbound,
                grid_tuple=random_grid,
                pi_edges=meta["pi_edges"],
                grid_meta=grid_meta,
            )
            if fixed_random:
                fixed_random_prepared = random_p

        dd_i = count_auto_fixed_rp(
            data_p,
            rp_edges=meta["rp_edges"],
            pi_edges=meta["pi_edges"],
            nthreads=config.nthreads,
        )
        dd_sum += dd_i

        rr_i = None
        if need_rr:
            if fixed_random and fixed_rr is not None:
                rr_i = fixed_rr
            else:
                rr_i = count_auto_fixed_rp(
                    random_p,
                    rp_edges=meta["rp_edges"],
                    pi_edges=meta["pi_edges"],
                    nthreads=config.nthreads,
                )
                if fixed_random:
                    fixed_rr = rr_i.copy()
            rr_sum += rr_i

        dr_i = None
        if need_dr:
            dr_i = count_cross_fixed_rp(
                data_p,
                random_p,
                rp_edges=meta["rp_edges"],
                pi_edges=meta["pi_edges"],
                nthreads=config.nthreads,
            )
            dr_sum += dr_i

        if bool(spec.store_realizations):
            one = _mean_counts(
                dd_i,
                rr_i,
                dr_i,
                nreal=1,
                meta=meta,
                n_data=data_template["nrows"],
                n_random=random_template["nrows"],
                weighted=not data_template["wunit"],
            )
            weighted = config.weights.weight_mode == "weighted" or (
                config.weights.weight_mode == "auto" and not data_template["wunit"]
            )
            one_result = estimate_auto(
                one,
                estimator=estimator,
                data_weights=(data_template["weights"] if weighted else None),
            )
            wp_realizations.append(np.asarray(one_result.wp, dtype=np.float64))

    counts = _mean_counts(
        dd_sum,
        rr_sum,
        dr_sum,
        nreal=int(spec.nreal),
        meta=meta,
        n_data=data_template["nrows"],
        n_random=random_template["nrows"],
        weighted=not data_template["wunit"],
    )
    weighted = config.weights.weight_mode == "weighted" or (
        config.weights.weight_mode == "auto" and not data_template["wunit"]
    )
    result = estimate_auto(
        counts,
        estimator=estimator,
        data_weights=(data_template["weights"] if weighted else None),
    )
    if wp_realizations:
        arr = np.asarray(wp_realizations, dtype=np.float64)
        result.mc_realizations = arr
        result.mc_wp_std = np.std(arr, axis=0)
    result.metadata.update({
        "mc_pdf": True,
        "fixed_rp_diagnostic": True,
        "fixed_rp_definition": "rp_from_reference_distance__pi_from_mc_draw",
        "mc_nreal": int(spec.nreal),
        "mc_random_mode": str(spec.random_mode),
        "mc_rr_fixed": bool(fixed_random),
        "reference_data_distance": data_reference_source,
        "reference_random_distance": random_reference_source,
        "fixed_rp_random_joint_policy": (
            "explicit_random_pdf_pair" if cdf_random is not None
            else "paired_data_donor"
        ),
    })
    return result


__all__ = [
    "count_auto_fixed_rp",
    "count_cross_fixed_rp",
    "run_auto_mc_pdf_fixed_rp",
]

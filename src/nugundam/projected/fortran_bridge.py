"""Adapters between prepared projected samples and the compiled pair counters."""
from __future__ import annotations

import os
import time

import numpy as np

try:  # pragma: no cover - optional speedup for exact-grid fallback
    from numba import njit, prange, set_num_threads as _numba_set_num_threads
    from numba.np.ufunc.parallel import get_thread_id as _numba_get_thread_id
except Exception:  # pragma: no cover
    njit = None
    prange = range
    _numba_set_num_threads = None
    _numba_get_thread_id = None

import nugundam.cflibfor as cff

from ..core.common import set_threads
from .prepare import subset_prepared_projected_sample
from .models import (
    PreparedProjectedSample,
    ProjectedAutoCounts,
    ProjectedAutoCountsResult,
    ProjectedCrossCounts,
    ProjectedCrossCountsResult,
)

_LOG_SINK = os.devnull

_QCHI_DIAG_KEYS = (
    "linked_candidates_visited",
    "pairs_after_rv_pruning",
    "pairs_rejected_by_pi_support",
    "pairs_rejected_by_zero_angle",
    "pairs_rejected_by_rp_support",
    "quantile_products_entered",
    "quantile_products_rejected_by_pi",
    "quantile_products_rejected_by_rp",
    "quantile_products_accepted",
)
_LAST_PAIR_DIAGNOSTICS = None


def _set_last_pair_diagnostics(value):
    global _LAST_PAIR_DIAGNOSTICS
    _LAST_PAIR_DIAGNOSTICS = value


def _consume_last_pair_diagnostics():
    global _LAST_PAIR_DIAGNOSTICS
    value = _LAST_PAIR_DIAGNOSTICS
    _LAST_PAIR_DIAGNOSTICS = None
    return value


def _safe_frac(num, den):
    den = float(den)
    return float(num) / den if den > 0.0 else 0.0


def _qchi_prepare_metadata(sample: PreparedProjectedSample) -> dict | None:
    if _pdf_repr(sample) != "quantile_chi":
        return None
    gm = getattr(sample, "grid_meta", {}) or {}
    keys = (
        "qchi_prepare_compress_time_s",
        "qchi_storage_dtype",
        "qchi_library_nbytes",
        "qchi_nquant",
        "qchi_nlib",
        "qchi_span_min",
        "qchi_span_median",
        "qchi_span_max",
        "qchi_random_inherits_library",
    )
    out = {k: gm[k] for k in keys if k in gm}
    if "qchi_nquant" not in out:
        out["qchi_nquant"] = int(getattr(sample, "pdf_k", 0) or 0)
    return out


def _qchi_diag_dict(kind: str, cntid: str, diag, *, n_left: int, n_right: int, nq_left: int, nq_right: int) -> dict:
    arr = np.asarray(diag, dtype=np.int64).ravel()
    vals = {name: int(arr[i]) if i < arr.size else 0 for i, name in enumerate(_QCHI_DIAG_KEYS)}
    rv = vals["pairs_after_rv_pruning"]
    nprod = vals["quantile_products_entered"]
    nacc = vals["quantile_products_accepted"]
    vals.update({
        "kernel_family": "quantile_chi",
        "kernel_kind": str(kind),
        "count_id": str(cntid),
        "n_left": int(n_left),
        "n_right": int(n_right),
        "nq_left": int(nq_left),
        "nq_right": int(nq_right),
        "support_pair_reject_fraction_after_rv": _safe_frac(
            vals["pairs_rejected_by_pi_support"] + vals["pairs_rejected_by_zero_angle"] + vals["pairs_rejected_by_rp_support"], rv
        ),
        "pi_support_reject_fraction_after_rv": _safe_frac(vals["pairs_rejected_by_pi_support"], rv),
        "rp_support_reject_fraction_after_rv": _safe_frac(vals["pairs_rejected_by_rp_support"], rv),
        "quantile_accept_fraction": _safe_frac(nacc, nprod),
        "quantile_reject_fraction_pi": _safe_frac(vals["quantile_products_rejected_by_pi"], nprod),
        "quantile_reject_fraction_rp": _safe_frac(vals["quantile_products_rejected_by_rp"], nprod),
    })
    return vals


def _attach_timing_and_diag(label: str, elapsed_s: float, diag, metadata: dict) -> None:
    times = metadata.setdefault("pair_times_s", {})
    times[str(label)] = float(elapsed_s)
    if diag is not None:
        diag = dict(diag)
        diag["elapsed_s"] = float(elapsed_s)
        metadata.setdefault("pair_diagnostics", {})[str(label)] = diag


def _diag_enabled_flag(pair_diagnostics: bool) -> int:
    return 1 if bool(pair_diagnostics) else 0


def _boot_array(nsepp: int, nsepv: int):
    return np.zeros((nsepp, nsepv, 0), dtype=np.float64)


def _transpose_counts(raw):
    return np.asarray(raw, dtype=np.float64).T


def _transpose_bootstrap(raw):
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 3:
        return arr
    return np.transpose(arr, (1, 2, 0))


def _transpose_jk_touch(raw):
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 3:
        return arr
    return np.transpose(arr, (2, 1, 0))


def _progress_arg(progress_file: str | None) -> str:
    return "" if progress_file is None else str(progress_file)


def _resolve_weight_mode(weight_mode: str, *flags: bool) -> bool:
    if weight_mode == "unweighted":
        return False
    if weight_mode == "weighted":
        return True
    return not all(flags)


def _integrate_pi(counts_2d: np.ndarray | None, pi_delta: np.ndarray) -> np.ndarray | None:
    if counts_2d is None:
        return None
    return 2.0 * np.sum(np.asarray(counts_2d, dtype=np.float64) * np.asarray(pi_delta, dtype=np.float64)[None, :], axis=1)


def _has_kernel(name: str) -> bool:
    try:
        getattr(cff.mod, name)
        return True
    except Exception:
        return False


def pdf_auto_bootstrap_fast_available(*, weighted: bool) -> bool:
    return _has_kernel('rppi_Ab_gmm_wp_wg' if weighted else 'rppi_Ab_gmm_wp')


def pdf_cross_bootstrap_fast_available(*, weighted: bool) -> bool:
    return _has_kernel('rppi_Cb_gmm_wp_wg' if weighted else 'rppi_Cb_gmm_wp')


def exact_pdf_auto_bootstrap_fast_available(*, weighted: bool) -> bool:
    return _has_kernel('rppi_Ab_grid_wp_wg' if weighted else 'rppi_Ab_grid_wp')


def exact_pdf_cross_bootstrap_fast_available(*, weighted: bool) -> bool:
    return _has_kernel('rppi_Cb_grid_wp_wg' if weighted else 'rppi_Cb_grid_wp')


def qchi_auto_bootstrap_fast_available(*, weighted: bool) -> bool:
    # Native fast qchi bootstrap kernels are currently implemented for
    # unweighted runs. Weighted qchi bootstrap should use the rerun backend.
    return (not weighted) and _has_kernel('rppi_Ab_qchi_wp')


def qchi_cross_bootstrap_fast_available(*, weighted: bool) -> bool:
    return (not weighted) and _has_kernel('rppi_Cb_qchi_wp')


def qchi_auto_jackknife_fast_available(*, weighted: bool) -> bool:
    return (not weighted) and _has_kernel('rppi_Ajk_qchi_wp')


def qchi_cross_jackknife_fast_available(*, weighted: bool) -> bool:
    return (not weighted) and _has_kernel('rppi_Cjk_qchi_wp')


def _pdf_repr(sample: PreparedProjectedSample) -> str:
    return str(getattr(sample, "pdf_repr", "none") or "none").strip().lower()


def _has_pdf_payload(sample: PreparedProjectedSample) -> bool:
    return bool(getattr(sample, "pdf_idx", None) is not None and _pdf_repr(sample) != "none")


def _progress_append(progress_file: str | None, line: str) -> None:
    if progress_file is None:
        return
    try:
        with open(progress_file, "a", encoding="utf-8") as fh:
            fh.write(str(line).rstrip() + "\n")
    except Exception:
        pass


def _write_exact_progress_header(progress_file: str | None, cntid: str, total: int) -> None:
    if progress_file is None or total <= 0:
        return
    _progress_append(progress_file, f"======== Counting {cntid} pairs in {int(total)} DEC strips ========")


def _write_exact_progress_step(progress_file: str | None, cntid: str, current: int, total: int, extra: str = "") -> None:
    if progress_file is None or total <= 0:
        return
    suffix = f" {extra}" if str(extra).strip() else ""
    _progress_append(progress_file, f"[{cntid}] stripe {int(current)}/{int(total)}{suffix}")



def _build_exact_active_support(prob, lo_idx, hi_idx):
    """Return flattened non-zero support for exact/ePDF left-side loops.

    The exact-grid accumulator integrates over one PDF explicitly and queries
    the other with a CDF.  This helper stores only non-zero entries of the
    explicit side, within the already-computed support window.  With
    ``active_floor=0`` semantics this is algebraically identical to the dense
    loop, but avoids scanning zero bins introduced by local grids, clipping, or
    edge refinement.
    """
    prob = np.asarray(prob, dtype=np.float64)
    lo_idx = np.asarray(lo_idx, dtype=np.int32)
    hi_idx = np.asarray(hi_idx, dtype=np.int32)
    if prob.ndim != 2:
        raise ValueError("exact/ePDF probability library must be 2D with shape (n_chi, n_pdf).")
    n_chi, n_pdf = prob.shape
    if lo_idx.size != n_pdf or hi_idx.size != n_pdf:
        raise ValueError("exact/ePDF support arrays must match the PDF library width.")
    starts = np.empty(n_pdf, dtype=np.int64)
    counts = np.empty(n_pdf, dtype=np.int64)
    chunks_idx = []
    chunks_prob = []
    offset = 0
    for j in range(n_pdf):
        lo = int(lo_idx[j])
        hi = int(hi_idx[j])
        if lo < 0:
            lo = 0
        if hi >= n_chi:
            hi = n_chi - 1
        if hi < lo:
            hi = lo
        col = prob[lo:hi + 1, j]
        nz = np.flatnonzero(col > 0.0)
        if nz.size == 0:
            # Defensive fallback for pathological rows.  Valid inputs should
            # not reach this branch because PDF rows are checked upstream.
            local = int(np.argmax(col)) if col.size else 0
            nz = np.asarray([local], dtype=np.int64)
        idx = (nz + lo).astype(np.int32, copy=False)
        vals = np.asarray(prob[idx, j], dtype=np.float64)
        starts[j] = offset
        counts[j] = idx.size
        offset += idx.size
        chunks_idx.append(idx)
        chunks_prob.append(vals)
    if chunks_idx:
        active_idx = np.concatenate(chunks_idx).astype(np.int32, copy=False)
        active_prob = np.concatenate(chunks_prob).astype(np.float64, copy=False)
    else:
        active_idx = np.empty(0, dtype=np.int32)
        active_prob = np.empty(0, dtype=np.float64)
    return (
        np.asarray(active_idx, dtype=np.int32),
        np.asarray(active_prob, dtype=np.float64),
        np.asarray(starts, dtype=np.int64),
        np.asarray(counts, dtype=np.int64),
    )


def _exact_pair_hist_py(grid_left, prob_left, lo_left, hi_left, idx_left: int,
                        grid_right, cdf_right, lo_right, hi_right, idx_right: int,
                        ang2: float, rp_edges: np.ndarray, pi_edges: np.ndarray,
                        active_idx_left=None, active_prob_left=None,
                        active_start_left=None, active_count_left=None) -> np.ndarray:
    """Reference Python exact-grid pair histogram.

    The pi-shell contribution is computed from cumulative masses at the
    pi-bin edges and then differenced. This matches the optimized Fortran
    implementation and avoids doing two CDF interval queries for every shell.
    """
    nsepp = len(rp_edges) - 1
    nsepv = len(pi_edges) - 1
    out = np.zeros((nsepp, nsepv), dtype=np.float64)
    if ang2 <= 0.0:
        return out
    lo_i = int(lo_left[idx_left]); hi_i = int(hi_left[idx_left])
    lo_j = int(lo_right[idx_right]); hi_j = int(hi_right[idx_right])
    chi_lo_j = float(grid_right[lo_j]); chi_hi_j = float(grid_right[hi_j])
    rp2_edges = np.asarray(rp_edges, dtype=np.float64) ** 2
    pi_edges = np.asarray(pi_edges, dtype=np.float64)
    if active_idx_left is not None and active_prob_left is not None and active_start_left is not None and active_count_left is not None:
        astart = int(active_start_left[idx_left])
        acount = int(active_count_left[idx_left])
        iterator = ((int(active_idx_left[a]), float(active_prob_left[a])) for a in range(astart, astart + acount))
    else:
        iterator = ((gi, float(prob_left[gi, idx_left])) for gi in range(lo_i, hi_i + 1))
    for gi, p_i in iterator:
        if p_i <= 0.0:
            continue
        chi_i = float(grid_left[gi])
        denom = float(ang2) * chi_i
        if denom <= 0.0:
            continue
        for b in range(nsepp):
            low2 = rp2_edges[b] / denom
            high2 = rp2_edges[b + 1] / denom
            base_lo = max(chi_lo_j, low2)
            base_hi = min(chi_hi_j, high2)
            if base_lo > base_hi:
                continue
            cmass = np.zeros(nsepv + 1, dtype=np.float64)
            for ie in range(nsepv + 1):
                pedge = max(0.0, float(pi_edges[ie]))
                if pedge <= 0.0:
                    continue
                lo = max(base_lo, chi_i - pedge)
                hi = min(base_hi, chi_i + pedge)
                if lo > hi:
                    continue
                jlo = max(lo_j, int(np.searchsorted(grid_right, lo, side='left')))
                jhi = min(hi_j, int(np.searchsorted(grid_right, hi, side='right')) - 1)
                if jhi < jlo:
                    continue
                mass = float(cdf_right[jhi, idx_right])
                if jlo > 0:
                    mass -= float(cdf_right[jlo - 1, idx_right])
                cmass[ie] = mass
            for ip in range(nsepv):
                if pi_edges[ip + 1] <= pi_edges[ip]:
                    continue
                mass = cmass[ip + 1] - cmass[ip]
                if mass > 0.0:
                    out[b, ip] += p_i * mass
    return out

if njit is not None:
    @njit(cache=True)
    def _cdf_interval_mass_numba(cdf, idxj, jlo, jhi):  # pragma: no cover - exercised at runtime
        if jhi < jlo:
            return 0.0
        mass = cdf[jhi, idxj]
        if jlo > 0:
            mass -= cdf[jlo - 1, idxj]
        return mass


    @njit(cache=True)
    def _exact_pair_accumulate_numba(out, grid, active_idx_i, active_prob_i, active_start_i, active_count_i, cdf_j, lo_i_arr, hi_i_arr, lo_j_arr, hi_j_arr, idx_i, idx_j, ang2, rp2_edges, pi_edges, wpair):  # pragma: no cover - exercised at runtime
        if ang2 <= 0.0:
            return
        lo_i = lo_i_arr[idx_i]
        hi_i = hi_i_arr[idx_i]
        lo_j = lo_j_arr[idx_j]
        hi_j = hi_j_arr[idx_j]
        chi_lo_j = grid[lo_j]
        chi_hi_j = grid[hi_j]
        nsepp = rp2_edges.shape[0] - 1
        nsepv = pi_edges.shape[0] - 1
        pimax = pi_edges[nsepv]
        astart = active_start_i[idx_i]
        acount = active_count_i[idx_i]
        for aa in range(astart, astart + acount):
            gi = active_idx_i[aa]
            p_i = active_prob_i[aa]
            if p_i <= 0.0:
                continue
            chi_i = grid[gi]
            low_pm = chi_i - pimax
            if low_pm < chi_lo_j:
                low_pm = chi_lo_j
            high_pm = chi_i + pimax
            if high_pm > chi_hi_j:
                high_pm = chi_hi_j
            if low_pm > high_pm:
                continue
            denom = ang2 * chi_i
            if denom <= 0.0:
                continue
            for b in range(nsepp):
                low2 = rp2_edges[b] / denom
                high2 = rp2_edges[b + 1] / denom
                base_lo = low_pm if low_pm > low2 else low2
                base_hi = high_pm if high_pm < high2 else high2
                if base_lo > base_hi:
                    continue
                cmass = np.zeros(nsepv + 1, dtype=np.float64)
                for ie in range(nsepv + 1):
                    pedge = pi_edges[ie]
                    if pedge < 0.0:
                        pedge = 0.0
                    if pedge <= 0.0:
                        continue
                    lo = chi_i - pedge
                    hi = chi_i + pedge
                    if lo < base_lo:
                        lo = base_lo
                    if hi > base_hi:
                        hi = base_hi
                    if lo > hi:
                        continue
                    jlo = np.searchsorted(grid, lo, side='left')
                    if jlo < lo_j:
                        jlo = lo_j
                    jhi = np.searchsorted(grid, hi, side='right') - 1
                    if jhi > hi_j:
                        jhi = hi_j
                    if jhi < jlo:
                        continue
                    cmass[ie] = _cdf_interval_mass_numba(cdf_j, idx_j, jlo, jhi)
                for ip in range(nsepv):
                    if pi_edges[ip + 1] <= pi_edges[ip]:
                        continue
                    mass = cmass[ip + 1] - cmass[ip]
                    if mass > 0.0:
                        out[b, ip] += p_i * mass * wpair


    @njit(cache=True, parallel=True)
    def _exact_auto_counts_block_numba(start, stop, x, y, z, weights, use_weights, pdf_idx, grid, prob, cdf, lo_idx, hi_idx, active_idx, active_prob, active_start, active_count, rp2_edges, pi_edges):  # pragma: no cover - exercised at runtime
        nsepp = rp2_edges.shape[0] - 1
        nsepv = pi_edges.shape[0] - 1
        acc = np.zeros((256, nsepp, nsepv), dtype=np.float64)
        rp2_min = rp2_edges[0]
        rp2_max = rp2_edges[-1]
        pimax = pi_edges[nsepv]
        n = x.shape[0]
        for i in prange(start, stop):
            tid = _numba_get_thread_id() if _numba_get_thread_id is not None else 0
            if tid >= acc.shape[0]:
                tid = 0
            idx_i = pdf_idx[i]
            chi_lo_i = grid[lo_idx[idx_i]]
            chi_hi_i = grid[hi_idx[idx_i]]
            xi = x[i]; yi = y[i]; zi = z[i]
            wi = weights[i] if use_weights else 1.0
            for j in range(i + 1, n):
                idx_j = pdf_idx[j]
                chi_lo_j = grid[lo_idx[idx_j]]
                chi_hi_j = grid[hi_idx[idx_j]]
                if chi_hi_i < chi_lo_j - pimax or chi_hi_j < chi_lo_i - pimax:
                    continue
                ang2 = (xi - x[j]) ** 2 + (yi - y[j]) ** 2 + (zi - z[j]) ** 2
                if ang2 <= 0.0:
                    continue
                if ang2 * (chi_lo_i * chi_lo_j) > rp2_max:
                    continue
                if ang2 * (chi_hi_i * chi_hi_j) < rp2_min:
                    continue
                wpair = wi * (weights[j] if use_weights else 1.0)
                _exact_pair_accumulate_numba(acc[tid], grid, active_idx, active_prob, active_start, active_count, cdf, lo_idx, hi_idx, lo_idx, hi_idx, idx_i, idx_j, ang2, rp2_edges, pi_edges, wpair)
        out = np.zeros((nsepp, nsepv), dtype=np.float64)
        for t in range(acc.shape[0]):
            out += acc[t]
        return out


    @njit(cache=True, parallel=True)
    def _exact_cross_counts_block_numba(start, stop, xl, yl, zl, wl, use_weights, idx_l, xr, yr, zr, wr, idx_r, grid, prob_l, cdf_r, lo_l, hi_l, lo_r, hi_r, active_idx_l, active_prob_l, active_start_l, active_count_l, rp2_edges, pi_edges):  # pragma: no cover - exercised at runtime
        nsepp = rp2_edges.shape[0] - 1
        nsepv = pi_edges.shape[0] - 1
        acc = np.zeros((256, nsepp, nsepv), dtype=np.float64)
        rp2_min = rp2_edges[0]
        rp2_max = rp2_edges[-1]
        pimax = pi_edges[nsepv]
        nr = xr.shape[0]
        for i in prange(start, stop):
            tid = _numba_get_thread_id() if _numba_get_thread_id is not None else 0
            if tid >= acc.shape[0]:
                tid = 0
            pi_idx = idx_l[i]
            chi_lo_i = grid[lo_l[pi_idx]]
            chi_hi_i = grid[hi_l[pi_idx]]
            xi = xl[i]; yi = yl[i]; zi = zl[i]
            wi = wl[i] if use_weights else 1.0
            for j in range(nr):
                pj_idx = idx_r[j]
                chi_lo_j = grid[lo_r[pj_idx]]
                chi_hi_j = grid[hi_r[pj_idx]]
                if chi_hi_i < chi_lo_j - pimax or chi_hi_j < chi_lo_i - pimax:
                    continue
                ang2 = (xi - xr[j]) ** 2 + (yi - yr[j]) ** 2 + (zi - zr[j]) ** 2
                if ang2 <= 0.0:
                    continue
                if ang2 * (chi_lo_i * chi_lo_j) > rp2_max:
                    continue
                if ang2 * (chi_hi_i * chi_hi_j) < rp2_min:
                    continue
                wpair = wi * (wr[j] if use_weights else 1.0)
                _exact_pair_accumulate_numba(acc[tid], grid, active_idx_l, active_prob_l, active_start_l, active_count_l, cdf_r, lo_l, hi_l, lo_r, hi_r, pi_idx, pj_idx, ang2, rp2_edges, pi_edges, wpair)
        out = np.zeros((nsepp, nsepv), dtype=np.float64)
        for t in range(acc.shape[0]):
            out += acc[t]
        return out
else:
    _exact_auto_counts_block_numba = None
    _exact_cross_counts_block_numba = None


def _run_exact_grid_auto_counts(data: PreparedProjectedSample, *, rp_edges, pi_edges, weight_mode: str, nthreads: int = 1, cntid: str = 'DD', progress_file: str | None = None):
    nsepp = len(rp_edges) - 1
    nsepv = len(pi_edges) - 1
    weighted = _resolve_weight_mode(weight_mode, data.wunit)
    grid = np.asarray(data.pdf_grid, dtype=np.float64)
    prob_lib = np.asarray(data.pdf_prob_lib, dtype=np.float64, order='F')
    cdf_lib = np.asarray(data.pdf_cdf_lib, dtype=np.float64, order='F')
    lo_idx = np.asarray(data.pdf_lo_idx, dtype=np.int32)
    hi_idx = np.asarray(data.pdf_hi_idx, dtype=np.int32)
    pdf_idx = np.asarray(data.pdf_idx, dtype=np.int32)
    active_idx, active_prob, active_start, active_count = _build_exact_active_support(prob_lib, lo_idx, hi_idx)
    active_idx_f = np.asarray(active_idx + 1, dtype=np.int32)
    active_start_f = np.asarray(active_start + 1, dtype=np.int32)
    active_count_f = np.asarray(active_count, dtype=np.int32)
    counts = np.zeros((nsepp, nsepv), dtype=np.float64)
    pimax = float(pi_edges[-1])
    rp_edges = np.asarray(rp_edges, dtype=np.float64)
    rp2_edges = rp_edges * rp_edges
    n = int(data.nrows)
    if n <= 1:
        return counts, _boot_array(nsepp, nsepv), None, None, None
    rv_search = float(max(data.grid_meta.get('pi_search', 0.0), pimax))
    progressf = _progress_arg(progress_file)
    if weighted and _has_kernel('rppi_A_grid_wp_wg'):
        dd = cff.mod.rppi_A_grid_wp_wg(
            set_threads(nthreads), n, data.dec, data.dist, data.dcang, data.weights, data.x, data.y, data.z,
            int(grid.shape[0]), int(prob_lib.shape[1]), grid, prob_lib, cdf_lib,
            lo_idx + 1, hi_idx + 1, pdf_idx,
            active_idx_f, active_start_f, active_count_f,
            nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search, data.sbound,
            data.mxh1, data.mxh2, data.mxh3, 0, cntid, _LOG_SINK, progressf, data.sk, data.ll,
        )
        return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, None
    if (not weighted) and _has_kernel('rppi_A_grid_wp'):
        dd = cff.mod.rppi_A_grid_wp(
            set_threads(nthreads), n, data.dec, data.dist, data.dcang, data.x, data.y, data.z,
            int(grid.shape[0]), int(prob_lib.shape[1]), grid, prob_lib, cdf_lib,
            lo_idx + 1, hi_idx + 1, pdf_idx,
            active_idx_f, active_start_f, active_count_f,
            nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search, data.sbound,
            data.mxh1, data.mxh2, data.mxh3, cntid, _LOG_SINK, progressf, data.sk, data.ll,
        )
        return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, None
    pdf_idx0 = pdf_idx - 1
    use_numba = _exact_auto_counts_block_numba is not None
    block = max(1, min(n, 256 if n < 4096 else 512))
    nblocks = (max(0, n - 1) + block - 1) // block
    _write_exact_progress_header(progress_file, cntid, nblocks)
    if use_numba and _numba_set_num_threads is not None:
        try:
            _numba_set_num_threads(max(1, int(nthreads)))
        except Exception:
            pass
    for ib in range(nblocks):
        start = ib * block
        stop = min(n - 1, start + block)
        if use_numba:
            hist = _exact_auto_counts_block_numba(start, stop, np.asarray(data.x, dtype=np.float64), np.asarray(data.y, dtype=np.float64), np.asarray(data.z, dtype=np.float64), np.asarray(data.weights, dtype=np.float64), bool(weighted), pdf_idx0, grid, prob_lib, cdf_lib, lo_idx, hi_idx, active_idx, active_prob, active_start, active_count, rp2_edges, np.asarray(pi_edges, dtype=np.float64))
        else:
            hist = np.zeros((nsepp, nsepv), dtype=np.float64)
            for i in range(start, stop):
                xi = float(data.x[i]); yi = float(data.y[i]); zi = float(data.z[i])
                idx_i = int(pdf_idx0[i])
                chi_lo_i = float(grid[int(lo_idx[idx_i])]); chi_hi_i = float(grid[int(hi_idx[idx_i])])
                for j in range(i + 1, n):
                    idx_j = int(pdf_idx0[j])
                    chi_lo_j = float(grid[int(lo_idx[idx_j])]); chi_hi_j = float(grid[int(hi_idx[idx_j])])
                    if chi_hi_i < chi_lo_j - pimax or chi_hi_j < chi_lo_i - pimax:
                        continue
                    ang2 = (xi - float(data.x[j])) ** 2 + (yi - float(data.y[j])) ** 2 + (zi - float(data.z[j])) ** 2
                    if ang2 <= 0.0:
                        continue
                    if ang2 * (chi_lo_i * chi_lo_j) > rp2_edges[-1]:
                        continue
                    if ang2 * (chi_hi_i * chi_hi_j) < rp2_edges[0]:
                        continue
                    wpair = float(data.weights[i]) * float(data.weights[j]) if weighted else 1.0
                    hist += wpair * _exact_pair_hist_py(
                            grid, prob_lib, lo_idx, hi_idx, idx_i,
                            grid, cdf_lib, lo_idx, hi_idx, idx_j,
                            ang2, rp_edges, pi_edges,
                            active_idx, active_prob, active_start, active_count,
                        )
        counts += hist
        _write_exact_progress_step(progress_file, cntid, ib + 1, nblocks)
    return counts, _boot_array(nsepp, nsepv), None, None, None


def _run_exact_grid_cross_counts(left: PreparedProjectedSample, right: PreparedProjectedSample, *, rp_edges, pi_edges, weight_mode: str, nthreads: int = 1, cntid: str = 'D1D2', progress_file: str | None = None):
    nsepp = len(rp_edges) - 1
    nsepv = len(pi_edges) - 1
    weighted = _resolve_weight_mode(weight_mode, left.wunit, right.wunit)
    left_grid = np.asarray(left.pdf_grid, dtype=np.float64)
    right_grid = np.asarray(right.pdf_grid, dtype=np.float64)
    if left_grid.shape != right_grid.shape or not np.allclose(left_grid, right_grid):
        raise ValueError("grid_chi_exact cross-correlations require the same shared chi-grid on both samples.")
    grid = left_grid
    prob_l = np.asarray(left.pdf_prob_lib, dtype=np.float64, order='F')
    cdf_r = np.asarray(right.pdf_cdf_lib, dtype=np.float64, order='F')
    lo_l = np.asarray(left.pdf_lo_idx, dtype=np.int32)
    hi_l = np.asarray(left.pdf_hi_idx, dtype=np.int32)
    lo_r = np.asarray(right.pdf_lo_idx, dtype=np.int32)
    hi_r = np.asarray(right.pdf_hi_idx, dtype=np.int32)
    idx_l = np.asarray(left.pdf_idx, dtype=np.int32)
    idx_r = np.asarray(right.pdf_idx, dtype=np.int32)
    active_idx_l, active_prob_l, active_start_l, active_count_l = _build_exact_active_support(prob_l, lo_l, hi_l)
    active_idx_l_f = np.asarray(active_idx_l + 1, dtype=np.int32)
    active_start_l_f = np.asarray(active_start_l + 1, dtype=np.int32)
    active_count_l_f = np.asarray(active_count_l, dtype=np.int32)
    counts = np.zeros((nsepp, nsepv), dtype=np.float64)
    pimax = float(pi_edges[-1])
    rp_edges = np.asarray(rp_edges, dtype=np.float64)
    rp2_edges = rp_edges * rp_edges
    n = int(left.nrows)
    nr = int(right.nrows)
    if n <= 0 or nr <= 0:
        return counts, _boot_array(nsepp, nsepv), None
    rv_search = float(max(left.grid_meta.get('pi_search', 0.0), right.grid_meta.get('pi_search', 0.0), pimax))
    progressf = _progress_arg(progress_file)
    if weighted and _has_kernel('rppi_C_grid_wp_wg'):
        dd = cff.mod.rppi_C_grid_wp_wg(
            set_threads(nthreads), n, left.ra, left.dec, left.dist, left.dcang, left.weights, left.x, left.y, left.z,
            int(grid.shape[0]), int(prob_l.shape[1]), grid, prob_l, lo_l + 1, hi_l + 1, idx_l,
            active_idx_l_f, active_start_l_f, active_count_l_f,
            nr, right.dist, right.dcang, right.weights, right.x, right.y, right.z,
            int(cdf_r.shape[1]), cdf_r, lo_r + 1, hi_r + 1, idx_r,
            nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search, right.sbound,
            right.mxh1, right.mxh2, right.mxh3, 0, cntid, _LOG_SINK, progressf, right.sk, right.ll,
        )
        return _transpose_counts(dd), _boot_array(nsepp, nsepv), None
    if (not weighted) and _has_kernel('rppi_C_grid_wp'):
        dd = cff.mod.rppi_C_grid_wp(
            set_threads(nthreads), n, left.ra, left.dec, left.dist, left.dcang, left.x, left.y, left.z,
            int(grid.shape[0]), int(prob_l.shape[1]), grid, prob_l, lo_l + 1, hi_l + 1, idx_l,
            active_idx_l_f, active_start_l_f, active_count_l_f,
            nr, right.dist, right.dcang, right.x, right.y, right.z,
            int(cdf_r.shape[1]), cdf_r, lo_r + 1, hi_r + 1, idx_r,
            nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search, right.sbound,
            right.mxh1, right.mxh2, right.mxh3, cntid, _LOG_SINK, progressf, right.sk, right.ll,
        )
        return _transpose_counts(dd), _boot_array(nsepp, nsepv), None
    idx_l0 = idx_l - 1
    idx_r0 = idx_r - 1
    use_numba = _exact_cross_counts_block_numba is not None
    block = max(1, min(n, 256 if n < 4096 else 512))
    nblocks = (n + block - 1) // block
    _write_exact_progress_header(progress_file, cntid, nblocks)
    if use_numba and _numba_set_num_threads is not None:
        try:
            _numba_set_num_threads(max(1, int(nthreads)))
        except Exception:
            pass
    for ib in range(nblocks):
        start = ib * block
        stop = min(n, start + block)
        if use_numba:
            hist = _exact_cross_counts_block_numba(start, stop, np.asarray(left.x, dtype=np.float64), np.asarray(left.y, dtype=np.float64), np.asarray(left.z, dtype=np.float64), np.asarray(left.weights, dtype=np.float64), bool(weighted), idx_l0, np.asarray(right.x, dtype=np.float64), np.asarray(right.y, dtype=np.float64), np.asarray(right.z, dtype=np.float64), np.asarray(right.weights, dtype=np.float64), idx_r0, grid, prob_l, cdf_r, lo_l, hi_l, lo_r, hi_r, active_idx_l, active_prob_l, active_start_l, active_count_l, rp2_edges, np.asarray(pi_edges, dtype=np.float64))
        else:
            hist = np.zeros((nsepp, nsepv), dtype=np.float64)
            for i in range(start, stop):
                xi = float(left.x[i]); yi = float(left.y[i]); zi = float(left.z[i])
                idx_i = int(idx_l0[i])
                chi_lo_i = float(grid[int(lo_l[idx_i])]); chi_hi_i = float(grid[int(hi_l[idx_i])])
                for j in range(nr):
                    idx_j = int(idx_r0[j])
                    chi_lo_j = float(grid[int(lo_r[idx_j])]); chi_hi_j = float(grid[int(hi_r[idx_j])])
                    if chi_hi_i < chi_lo_j - pimax or chi_hi_j < chi_lo_i - pimax:
                        continue
                    ang2 = (xi - float(right.x[j])) ** 2 + (yi - float(right.y[j])) ** 2 + (zi - float(right.z[j])) ** 2
                    if ang2 <= 0.0:
                        continue
                    if ang2 * (chi_lo_i * chi_lo_j) > rp2_edges[-1]:
                        continue
                    if ang2 * (chi_hi_i * chi_hi_j) < rp2_edges[0]:
                        continue
                    wpair = float(left.weights[i]) * float(right.weights[j]) if weighted else 1.0
                    hist += wpair * _exact_pair_hist_py(
                            grid, prob_l, lo_l, hi_l, idx_i,
                            grid, cdf_r, lo_r, hi_r, idx_j,
                            ang2, rp_edges, pi_edges,
                            active_idx_l, active_prob_l, active_start_l, active_count_l,
                        )
        counts += hist
        _write_exact_progress_step(progress_file, cntid, ib + 1, nblocks)
    return counts, _boot_array(nsepp, nsepv), None


def _rr_norm_pairs_full(n_random: int) -> float:
    return 0.5 * float(n_random) * float(max(0, n_random - 1))


def _split_random_chunks(n_random: int, *, n_data: int, split_random) -> list[np.ndarray]:
    """Build shuffled index chunks for split-random RR counting.

    Parameters
    ----------
    n_random : int
        Total number of rows in the prepared random catalog.
    n_data : int
        Number of rows in the prepared data catalog. Used by
        ``mode="match_data"`` to choose a chunk size comparable to the data
        sample.
    split_random : object
        Configuration object exposing ``mode``, ``nchunks``, ``chunk_size``,
        and ``seed`` attributes.

    Returns
    -------
    list of numpy.ndarray
        One integer-index array per random chunk. The union of all arrays
        covers the full random catalog exactly once.
    """
    if n_random <= 0:
        return []
    mode = str(getattr(split_random, "mode", "match_data")).strip().lower()
    if mode == "match_data":
        chunk_size = max(1, int(n_data))
        nchunks = int(np.ceil(n_random / chunk_size))
    elif mode == "nchunks":
        nchunks = int(getattr(split_random, "nchunks", 0) or 0)
        if nchunks <= 0:
            raise ValueError("split_random.nchunks must be a positive integer when mode='nchunks'.")
    elif mode == "chunk_size":
        chunk_size = int(getattr(split_random, "chunk_size", 0) or 0)
        if chunk_size <= 0:
            raise ValueError("split_random.chunk_size must be a positive integer when mode='chunk_size'.")
        nchunks = int(np.ceil(n_random / chunk_size))
    else:
        raise ValueError(f"Unsupported split_random.mode={mode!r}.")
    nchunks = max(1, min(int(nchunks), int(n_random)))
    rng = np.random.default_rng(int(getattr(split_random, "seed", 12345)))
    shuffled = rng.permutation(int(n_random))
    return [np.asarray(idx, dtype=np.int64) for idx in np.array_split(shuffled, nchunks) if len(idx) > 0]


def _run_rppi_split_rr_counts(random: PreparedProjectedSample, *, rp_edges, pi_edges, pi_delta, nthreads: int, dojk: bool, nreg: int, bseed: int, progress_file: str | None, split_random, n_data: int, pair_diagnostics: bool = False):
    """Count split-random RR terms for projected auto-correlations.

    Each shuffled random chunk is converted into a chunk-local prepared sample
    with ``regrid=True`` so the RR counter uses a fresh autogrid and pxsort for
    that chunk rather than inheriting the grid of the full random catalog. The
    returned pair normalization is the exact number of within-chunk RR pairs
    included in the accumulated counts.
    """
    chunks = _split_random_chunks(int(random.nrows), n_data=int(n_data), split_random=split_random)
    rr = None
    rr_norm_pairs = 0.0
    chunk_sizes: list[int] = []
    chunk_times: list[float] = []
    chunk_diagnostics: list[dict] = []
    nchunks = len(chunks)
    for ichunk, idx in enumerate(chunks, start=1):
        chunk = subset_prepared_projected_sample(random, idx, pi_edges=pi_edges, regrid=True)
        cntid = f"RR split {ichunk}/{nchunks}" if nchunks > 1 else "RR"
        t0 = time.perf_counter()
        rr_chunk, _, _, _, _ = run_rppi_auto_counts(
            chunk,
            rp_edges=rp_edges,
            pi_edges=pi_edges,
            nthreads=nthreads,
            weight_mode="unweighted",
            doboot=False,
            dojk=dojk,
            nreg=nreg,
            nbts=0,
            bseed=bseed,
            cntid=cntid,
            progress_file=progress_file,
            pair_diagnostics=pair_diagnostics,
        )
        elapsed = time.perf_counter() - t0
        diag = _consume_last_pair_diagnostics()
        chunk_times.append(float(elapsed))
        if diag is not None:
            d = dict(diag)
            d["elapsed_s"] = float(elapsed)
            d["chunk_index"] = int(ichunk)
            d["chunk_size"] = int(len(idx))
            chunk_diagnostics.append(d)
        rr = np.asarray(rr_chunk, dtype=np.float64) if rr is None else rr + np.asarray(rr_chunk, dtype=np.float64)
        n_chunk = int(len(idx))
        chunk_sizes.append(n_chunk)
        rr_norm_pairs += 0.5 * n_chunk * max(0, n_chunk - 1)
    if rr is None:
        rr = np.zeros((len(rp_edges) - 1, len(pi_edges) - 1), dtype=np.float64)
    return rr, rr_norm_pairs, {
        "split_random_enabled": True,
        "split_random_mode": str(getattr(split_random, "mode", "match_data")),
        "split_random_seed": int(getattr(split_random, "seed", 12345)),
        "split_random_nchunks": len(chunk_sizes),
        "split_random_chunk_sizes": chunk_sizes,
        "split_random_rr_times_s": chunk_times,
        "split_random_rr_total_time_s": float(np.sum(chunk_times)) if chunk_times else 0.0,
        "split_random_rr_diagnostics": chunk_diagnostics,
        "intpi_rr_norm_pairs": rr_norm_pairs,
    }

def run_rppi_auto_counts(
    data: PreparedProjectedSample,
    *,
    rp_edges,
    pi_edges,
    nthreads: int,
    weight_mode: str,
    doboot: bool,
    dojk: bool = False,
    nreg: int = 0,
    nbts: int,
    bseed: int,
    cntid: str,
    progress_file: str | None = None,
    pair_diagnostics: bool = False,
):
    _set_last_pair_diagnostics(None)
    nt = set_threads(nthreads)
    npt = int(data.nrows)
    weighted = _resolve_weight_mode(weight_mode, data.wunit)
    progressf = _progress_arg(progress_file)
    nsepp = len(rp_edges) - 1
    nsepv = len(pi_edges) - 1

    # --- PDF mode ------------------------------------------------------------
    pdf_repr = _pdf_repr(data)
    pdf_mode = _has_pdf_payload(data)
    if pdf_mode:
        if pdf_repr == 'quantile_chi':
            qchi = np.asarray(data.pdf_qchi_lib, dtype=np.float64, order='F')
            qlo = np.asarray(data.pdf_qlo_lib, dtype=np.float64)
            qhi = np.asarray(data.pdf_qhi_lib, dtype=np.float64)
            pdf_idx = np.asarray(data.pdf_idx, dtype=np.int32)
            rv_search = float(data.grid_meta.get('pi_search', float(pi_edges[-1]) if len(pi_edges) else 0.0))
            diag_flag = _diag_enabled_flag(pair_diagnostics)
            if dojk and data.region_id is not None and nreg > 0:
                if weighted:
                    raise NotImplementedError("quantile_chi native fast jackknife is currently implemented for unweighted runs only; use the rerun backend for weighted resampling.")
                if not _has_kernel('rppi_Ajk_qchi_wp'):
                    raise RuntimeError("quantile_chi jackknife requested but rppi_Ajk_qchi_wp is unavailable. Rebuild the extension module.")
                regs = np.asarray(data.region_id, dtype=np.int32)
                dd, touch = cff.mod.rppi_Ajk_qchi_wp(
                    nt, npt, data.dec, data.dist, data.dcang, data.x, data.y, data.z,
                    int(qchi.shape[0]), int(qchi.shape[1]), qchi, qlo, qhi, pdf_idx, regs, int(nreg),
                    nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                    data.sbound, data.mxh1, data.mxh2, data.mxh3,
                    cntid, _LOG_SINK, progressf, data.sk, data.ll,
                )
                return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, _transpose_jk_touch(touch)
            if doboot:
                if weighted:
                    raise NotImplementedError("quantile_chi native fast bootstrap is currently implemented for unweighted runs only; use the rerun backend for weighted resampling.")
                if not _has_kernel('rppi_Ab_qchi_wp'):
                    raise RuntimeError("quantile_chi bootstrap requested but rppi_Ab_qchi_wp is unavailable. Rebuild the extension module.")
                dd, bdd = cff.mod.rppi_Ab_qchi_wp(
                    nt, npt, data.dec, data.dist, data.dcang, data.x, data.y, data.z,
                    int(qchi.shape[0]), int(qchi.shape[1]), qchi, qlo, qhi, pdf_idx,
                    nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                    data.sbound, data.mxh1, data.mxh2, data.mxh3, nbts, bseed,
                    cntid, _LOG_SINK, progressf, data.sk, data.ll,
                )
                return _transpose_counts(dd), _transpose_bootstrap(bdd), None, None, None
            if weighted:
                if not _has_kernel('rppi_A_qchi_wp_wg'):
                    raise RuntimeError("quantile_chi weighted counts requested but rppi_A_qchi_wp_wg is unavailable. Rebuild the extension module.")
                out = cff.mod.rppi_A_qchi_wp_wg(
                    nt, npt, data.dec, data.dist, data.dcang, data.weights, data.x, data.y, data.z,
                    int(qchi.shape[0]), int(qchi.shape[1]), qchi, qlo, qhi, pdf_idx,
                    nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                    data.sbound, data.mxh1, data.mxh2, data.mxh3, 0, diag_flag,
                    cntid, _LOG_SINK, progressf, data.sk, data.ll,
                )
            else:
                if not _has_kernel('rppi_A_qchi_wp'):
                    raise RuntimeError("quantile_chi requested but rppi_A_qchi_wp is unavailable. Rebuild the extension module.")
                out = cff.mod.rppi_A_qchi_wp(
                    nt, npt, data.dec, data.dist, data.dcang, data.x, data.y, data.z,
                    int(qchi.shape[0]), int(qchi.shape[1]), qchi, qlo, qhi, pdf_idx,
                    nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                    data.sbound, data.mxh1, data.mxh2, data.mxh3, diag_flag,
                    cntid, _LOG_SINK, progressf, data.sk, data.ll,
                )
            if isinstance(out, tuple):
                dd, qdiag = out[:2]
            else:  # defensive compatibility with old compiled extension
                dd, qdiag = out, None
            if pair_diagnostics and qdiag is not None:
                _set_last_pair_diagnostics(_qchi_diag_dict(
                    "auto", cntid, qdiag, n_left=npt, n_right=npt,
                    nq_left=int(qchi.shape[0]), nq_right=int(qchi.shape[0]),
                ))
            return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, None
        if pdf_repr == 'grid_chi_exact':
            if doboot:
                grid = np.asarray(data.pdf_grid, dtype=np.float64)
                prob_lib = np.asarray(data.pdf_prob_lib, dtype=np.float64, order='F')
                cdf_lib = np.asarray(data.pdf_cdf_lib, dtype=np.float64, order='F')
                lo_idx = np.asarray(data.pdf_lo_idx, dtype=np.int32)
                hi_idx = np.asarray(data.pdf_hi_idx, dtype=np.int32)
                pdf_idx = np.asarray(data.pdf_idx, dtype=np.int32)
                active_idx, active_prob, active_start, active_count = _build_exact_active_support(prob_lib, lo_idx, hi_idx)
                active_idx_f = np.asarray(active_idx + 1, dtype=np.int32)
                active_start_f = np.asarray(active_start + 1, dtype=np.int32)
                active_count_f = np.asarray(active_count, dtype=np.int32)
                rv_search = float(data.grid_meta.get('pi_search', float(pi_edges[-1]) if len(pi_edges) else 0.0))
                if weighted and _has_kernel('rppi_Ab_grid_wp_wg'):
                    out = cff.mod.rppi_Ab_grid_wp_wg(
                        nt, npt, data.dec, data.dist, data.dcang, data.weights, data.x, data.y, data.z,
                        int(grid.shape[0]), int(prob_lib.shape[1]), grid, prob_lib, cdf_lib,
                        lo_idx + 1, hi_idx + 1, pdf_idx, active_idx_f, active_start_f, active_count_f,
                        nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                        data.sbound, data.mxh1, data.mxh2, data.mxh3, nbts, bseed, 0,
                        cntid, _LOG_SINK, progressf, data.sk, data.ll,
                    )
                    dd, bdd, normb, sumwb = out[:4]
                    return (_transpose_counts(dd), _transpose_bootstrap(bdd),
                            np.asarray(normb, dtype=np.float64), np.asarray(sumwb, dtype=np.float64), None)
                if (not weighted) and _has_kernel('rppi_Ab_grid_wp'):
                    dd, bdd = cff.mod.rppi_Ab_grid_wp(
                        nt, npt, data.dec, data.dist, data.dcang, data.x, data.y, data.z,
                        int(grid.shape[0]), int(prob_lib.shape[1]), grid, prob_lib, cdf_lib,
                        lo_idx + 1, hi_idx + 1, pdf_idx, active_idx_f, active_start_f, active_count_f,
                        nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                        data.sbound, data.mxh1, data.mxh2, data.mxh3, nbts, bseed,
                        cntid, _LOG_SINK, progressf, data.sk, data.ll,
                    )
                    return _transpose_counts(dd), _transpose_bootstrap(bdd), None, None, None
                raise RuntimeError('grid_chi_exact bootstrap requested but compiled exact-grid bootstrap kernels are unavailable.')
            if dojk and data.region_id is not None and nreg > 0:
                grid = np.asarray(data.pdf_grid, dtype=np.float64)
                prob_lib = np.asarray(data.pdf_prob_lib, dtype=np.float64, order='F')
                cdf_lib = np.asarray(data.pdf_cdf_lib, dtype=np.float64, order='F')
                lo_idx = np.asarray(data.pdf_lo_idx, dtype=np.int32)
                hi_idx = np.asarray(data.pdf_hi_idx, dtype=np.int32)
                pdf_idx = np.asarray(data.pdf_idx, dtype=np.int32)
                regs = np.asarray(data.region_id, dtype=np.int32)
                active_idx, active_prob, active_start, active_count = _build_exact_active_support(prob_lib, lo_idx, hi_idx)
                active_idx_f = np.asarray(active_idx + 1, dtype=np.int32)
                active_start_f = np.asarray(active_start + 1, dtype=np.int32)
                active_count_f = np.asarray(active_count, dtype=np.int32)
                rv_search = float(data.grid_meta.get('pi_search', float(pi_edges[-1]) if len(pi_edges) else 0.0))
                if weighted and _has_kernel('rppi_Ajk_grid_wp_wg'):
                    dd, touch = cff.mod.rppi_Ajk_grid_wp_wg(
                        nt, npt, data.dec, data.dist, data.dcang, data.weights, data.x, data.y, data.z,
                        int(grid.shape[0]), int(prob_lib.shape[1]), grid, prob_lib, cdf_lib,
                        lo_idx + 1, hi_idx + 1, pdf_idx,
                        active_idx_f, active_start_f, active_count_f, regs, int(nreg),
                        nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64),
                        rv_search, data.sbound, data.mxh1, data.mxh2, data.mxh3, 0,
                        cntid, _LOG_SINK, progressf, data.sk, data.ll,
                    )
                    return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, _transpose_jk_touch(touch)
                if (not weighted) and _has_kernel('rppi_Ajk_grid_wp'):
                    dd, touch = cff.mod.rppi_Ajk_grid_wp(
                        nt, npt, data.dec, data.dist, data.dcang, data.x, data.y, data.z,
                        int(grid.shape[0]), int(prob_lib.shape[1]), grid, prob_lib, cdf_lib,
                        lo_idx + 1, hi_idx + 1, pdf_idx,
                        active_idx_f, active_start_f, active_count_f, regs, int(nreg),
                        nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64),
                        rv_search, data.sbound, data.mxh1, data.mxh2, data.mxh3,
                        cntid, _LOG_SINK, progressf, data.sk, data.ll,
                    )
                    return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, _transpose_jk_touch(touch)
                raise RuntimeError('grid_chi_exact jackknife touch requested but compiled exact-grid JK kernels are unavailable.')
            return _run_exact_grid_auto_counts(data, rp_edges=rp_edges, pi_edges=pi_edges, weight_mode=weight_mode, nthreads=nthreads, cntid=cntid, progress_file=progress_file)
        rv_search = float(data.grid_meta.get('pi_search', float(pi_edges[-1]) if len(pi_edges) else 0.0))
        prob_floor = float(data.grid_meta.get('prob_floor', 1.0e-10))
        pdf_idx = np.asarray(data.pdf_idx, dtype=np.int32)
        nlib = int(getattr(data.pdf_alpha_lib, "shape", (0, 0))[1])
        if dojk and data.region_id is not None and nreg > 0:
            regs = np.asarray(data.region_id, dtype=np.int32)
            if weighted and _has_kernel('rppi_Ajk_gmm_wp_wg'):
                dd, touch = cff.mod.rppi_Ajk_gmm_wp_wg(
                    nt, npt, data.dec, data.dist, data.dcang, data.weights, data.x, data.y, data.z,
                    int(data.pdf_k), nlib, data.pdf_alpha_lib, data.pdf_mu_lib, data.pdf_sig_lib, pdf_idx, regs, nreg,
                    nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, data.sbound,
                    data.mxh1, data.mxh2, data.mxh3, 0, cntid, _LOG_SINK, progressf,
                    data.sk, data.ll,
                )
                return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, _transpose_jk_touch(touch)
            if (not weighted) and _has_kernel('rppi_Ajk_gmm_wp'):
                dd, touch = cff.mod.rppi_Ajk_gmm_wp(
                    nt, npt, data.dec, data.dist, data.dcang, data.x, data.y, data.z,
                    int(data.pdf_k), nlib, data.pdf_alpha_lib, data.pdf_mu_lib, data.pdf_sig_lib, pdf_idx, regs, nreg,
                    nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, data.sbound,
                    data.mxh1, data.mxh2, data.mxh3, cntid, _LOG_SINK, progressf,
                    data.sk, data.ll,
                )
                return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, _transpose_jk_touch(touch)
        if doboot:
            if weighted and _has_kernel('rppi_Ab_gmm_wp_wg'):
                out = cff.mod.rppi_Ab_gmm_wp_wg(
                    nt, npt, data.dec, data.dist, data.dcang, data.weights, data.x, data.y, data.z,
                    int(data.pdf_k), nlib, data.pdf_alpha_lib, data.pdf_mu_lib, data.pdf_sig_lib, pdf_idx,
                    nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, data.sbound,
                    data.mxh1, data.mxh2, data.mxh3, nbts, bseed, 0, cntid, _LOG_SINK, progressf,
                    data.sk, data.ll,
                )
                if isinstance(out, tuple) and len(out) >= 4:
                    dd, bdd, normb, sumwb = out[:4]
                    return _transpose_counts(dd), _transpose_bootstrap(bdd), np.asarray(normb, dtype=np.float64), np.asarray(sumwb, dtype=np.float64), None
                dd, bdd = out
                return _transpose_counts(dd), _transpose_bootstrap(bdd), None, None, None
            if (not weighted) and _has_kernel('rppi_Ab_gmm_wp'):
                dd, bdd = cff.mod.rppi_Ab_gmm_wp(
                    nt, npt, data.dec, data.dist, data.dcang, data.x, data.y, data.z,
                    int(data.pdf_k), nlib, data.pdf_alpha_lib, data.pdf_mu_lib, data.pdf_sig_lib, pdf_idx,
                    nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, data.sbound,
                    data.mxh1, data.mxh2, data.mxh3, nbts, bseed, cntid, _LOG_SINK, progressf,
                    data.sk, data.ll,
                )
                return _transpose_counts(dd), _transpose_bootstrap(bdd), None, None, None
            raise RuntimeError('PDF bootstrap requested but the compiled GMM bootstrap kernels are not available. Rebuild the extension module or use the rerun backend.')
        if weighted and _has_kernel('rppi_A_gmm_wp_wg'):
            dd = cff.mod.rppi_A_gmm_wp_wg(
                nt, npt, data.dec, data.dist, data.dcang, data.weights, data.x, data.y, data.z,
                int(data.pdf_k), nlib, data.pdf_alpha_lib, data.pdf_mu_lib, data.pdf_sig_lib, pdf_idx,
                nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, data.sbound,
                data.mxh1, data.mxh2, data.mxh3, 0, cntid, _LOG_SINK, progressf,
                data.sk, data.ll,
            )
            return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, None
        if (not weighted) and _has_kernel('rppi_A_gmm_wp'):
            dd = cff.mod.rppi_A_gmm_wp(
                nt, npt, data.dec, data.dist, data.dcang, data.x, data.y, data.z,
                int(data.pdf_k), nlib, data.pdf_alpha_lib, data.pdf_mu_lib, data.pdf_sig_lib, pdf_idx,
                nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, data.sbound,
                data.mxh1, data.mxh2, data.mxh3, cntid, _LOG_SINK, progressf,
                data.sk, data.ll,
            )
            return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, None
        raise RuntimeError('PDF mode requested but the compiled GMM kernels are not available. Rebuild the extension module.')
    # ------------------------------------------------------------------------
    if dojk and data.region_id is not None and nreg > 0:
        regs = np.asarray(data.region_id, dtype=np.int32)
        if weighted and _has_kernel("rppi_Ajk_wg"):
            dd, touch = cff.mod.rppi_Ajk_wg(
                nt, npt, data.dec, data.dist, data.weights, data.x, data.y, data.z,
                regs, nreg, nsepp, rp_edges, nsepv, pi_edges, data.sbound,
                data.mxh1, data.mxh2, data.mxh3, 0, cntid, _LOG_SINK, progressf,
                data.sk, data.ll,
            )
            return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, _transpose_jk_touch(touch)
        if (not weighted) and _has_kernel("rppi_Ajk"):
            dd, touch = cff.mod.rppi_Ajk(
                nt, npt, data.dec, data.dist, data.x, data.y, data.z,
                regs, nreg, nsepp, rp_edges, nsepv, pi_edges, data.sbound,
                data.mxh1, data.mxh2, data.mxh3, cntid, _LOG_SINK, progressf,
                data.sk, data.ll,
            )
            return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, _transpose_jk_touch(touch)
    if doboot:
        if weighted:
            out = cff.mod.rppi_Ab_wg(nt, npt, data.dec, data.dist, data.weights, data.x, data.y, data.z, nsepp, rp_edges, nsepv, pi_edges, data.sbound, data.mxh1, data.mxh2, data.mxh3, nbts, bseed, 0, cntid, _LOG_SINK, progressf, data.sk, data.ll)
            if isinstance(out, tuple) and len(out) >= 4:
                dd, bdd, normb, sumwb = out[:4]
                return _transpose_counts(dd), _transpose_bootstrap(bdd), np.asarray(normb, dtype=np.float64), np.asarray(sumwb, dtype=np.float64), None
            dd, bdd = out
            return _transpose_counts(dd), _transpose_bootstrap(bdd), None, None, None
        dd, bdd = cff.mod.rppi_Ab(nt, npt, data.dec, data.dist, data.x, data.y, data.z, nsepp, rp_edges, nsepv, pi_edges, data.sbound, data.mxh1, data.mxh2, data.mxh3, nbts, bseed, cntid, _LOG_SINK, progressf, data.sk, data.ll)
        return _transpose_counts(dd), _transpose_bootstrap(bdd), None, None, None
    if weighted:
        dd = cff.mod.rppi_A_wg(nt, npt, data.dec, data.dist, data.weights, data.x, data.y, data.z, nsepp, rp_edges, nsepv, pi_edges, data.sbound, data.mxh1, data.mxh2, data.mxh3, 0, cntid, _LOG_SINK, progressf, data.sk, data.ll)
    else:
        dd = cff.mod.rppi_A(nt, npt, data.dec, data.dist, data.x, data.y, data.z, nsepp, rp_edges, nsepv, pi_edges, data.sbound, data.mxh1, data.mxh2, data.mxh3, cntid, _LOG_SINK, progressf, data.sk, data.ll)
    return _transpose_counts(dd), _boot_array(nsepp, nsepv), None, None, None


def run_rppi_cross_counts(
    left: PreparedProjectedSample,
    right: PreparedProjectedSample,
    *,
    rp_edges,
    pi_edges,
    nthreads: int,
    weight_mode: str,
    doboot: bool,
    dojk: bool = False,
    nreg: int = 0,
    nbts: int,
    bseed: int,
    cntid: str,
    progress_file: str | None = None,
    pair_diagnostics: bool = False,
):
    _set_last_pair_diagnostics(None)
    nt = set_threads(nthreads)
    npt = int(left.nrows)
    npt1 = int(right.nrows)
    weighted = _resolve_weight_mode(weight_mode, left.wunit, right.wunit)
    progressf = _progress_arg(progress_file)
    nsepp = len(rp_edges) - 1
    nsepv = len(pi_edges) - 1

    # --- PDF mode [cross] ---------------------------------------------------
    left_repr = _pdf_repr(left)
    right_repr = _pdf_repr(right)
    pdf_mode = _has_pdf_payload(left) and _has_pdf_payload(right)
    if pdf_mode:
        if left_repr == 'quantile_chi' or right_repr == 'quantile_chi':
            if left_repr != 'quantile_chi' or right_repr != 'quantile_chi':
                raise NotImplementedError("quantile_chi cross-counts currently require both samples to use quantile_chi.")
            qchi_l = np.asarray(left.pdf_qchi_lib, dtype=np.float64, order='F')
            qlo_l = np.asarray(left.pdf_qlo_lib, dtype=np.float64)
            qhi_l = np.asarray(left.pdf_qhi_lib, dtype=np.float64)
            qchi_r = np.asarray(right.pdf_qchi_lib, dtype=np.float64, order='F')
            qlo_r = np.asarray(right.pdf_qlo_lib, dtype=np.float64)
            qhi_r = np.asarray(right.pdf_qhi_lib, dtype=np.float64)
            idx_l = np.asarray(left.pdf_idx, dtype=np.int32)
            idx_r = np.asarray(right.pdf_idx, dtype=np.int32)
            rv_search = float(max(left.grid_meta.get('pi_search', 0.0), right.grid_meta.get('pi_search', 0.0), float(pi_edges[-1]) if len(pi_edges) else 0.0))
            diag_flag = _diag_enabled_flag(pair_diagnostics)
            if dojk and left.region_id is not None and right.region_id is not None and nreg > 0:
                if weighted:
                    raise NotImplementedError("quantile_chi native fast cross jackknife is currently implemented for unweighted runs only; use the rerun backend for weighted resampling.")
                if not _has_kernel('rppi_Cjk_qchi_wp'):
                    raise RuntimeError("quantile_chi cross jackknife requested but rppi_Cjk_qchi_wp is unavailable. Rebuild the extension module.")
                reg_l = np.asarray(left.region_id, dtype=np.int32)
                reg_r = np.asarray(right.region_id, dtype=np.int32)
                dd, touch = cff.mod.rppi_Cjk_qchi_wp(
                    nt, npt, left.ra, left.dec, left.dist, left.dcang, left.x, left.y, left.z,
                    int(qchi_l.shape[0]), int(qchi_l.shape[1]), qchi_l, qlo_l, qhi_l, idx_l, reg_l,
                    npt1, right.dist, right.dcang, right.x, right.y, right.z,
                    int(qchi_r.shape[0]), int(qchi_r.shape[1]), qchi_r, qlo_r, qhi_r, idx_r, reg_r,
                    int(nreg), nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                    right.sbound, right.mxh1, right.mxh2, right.mxh3,
                    cntid, _LOG_SINK, progressf, right.sk, right.ll,
                )
                return _transpose_counts(dd), _boot_array(nsepp, nsepv), _transpose_jk_touch(touch)
            if doboot:
                if weighted:
                    raise NotImplementedError("quantile_chi native fast cross bootstrap is currently implemented for unweighted runs only; use the rerun backend for weighted resampling.")
                if not _has_kernel('rppi_Cb_qchi_wp'):
                    raise RuntimeError("quantile_chi cross bootstrap requested but rppi_Cb_qchi_wp is unavailable. Rebuild the extension module.")
                dd, bdd = cff.mod.rppi_Cb_qchi_wp(
                    nt, npt, left.ra, left.dec, left.dist, left.dcang, left.x, left.y, left.z,
                    int(qchi_l.shape[0]), int(qchi_l.shape[1]), qchi_l, qlo_l, qhi_l, idx_l,
                    npt1, right.dist, right.dcang, right.x, right.y, right.z,
                    int(qchi_r.shape[0]), int(qchi_r.shape[1]), qchi_r, qlo_r, qhi_r, idx_r,
                    nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                    right.sbound, right.mxh1, right.mxh2, right.mxh3, nbts, bseed,
                    cntid, _LOG_SINK, progressf, right.sk, right.ll,
                )
                return _transpose_counts(dd), _transpose_bootstrap(bdd), None
            if weighted:
                if not _has_kernel('rppi_C_qchi_wp_wg'):
                    raise RuntimeError("quantile_chi weighted cross-counts requested but rppi_C_qchi_wp_wg is unavailable. Rebuild the extension module.")
                out = cff.mod.rppi_C_qchi_wp_wg(
                    nt, npt, left.ra, left.dec, left.dist, left.dcang, left.weights, left.x, left.y, left.z,
                    int(qchi_l.shape[0]), int(qchi_l.shape[1]), qchi_l, qlo_l, qhi_l, idx_l,
                    npt1, right.dist, right.dcang, right.weights, right.x, right.y, right.z,
                    int(qchi_r.shape[0]), int(qchi_r.shape[1]), qchi_r, qlo_r, qhi_r, idx_r,
                    nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                    right.sbound, right.mxh1, right.mxh2, right.mxh3, 0, diag_flag,
                    cntid, _LOG_SINK, progressf, right.sk, right.ll,
                )
            else:
                if not _has_kernel('rppi_C_qchi_wp'):
                    raise RuntimeError("quantile_chi requested but rppi_C_qchi_wp is unavailable. Rebuild the extension module.")
                out = cff.mod.rppi_C_qchi_wp(
                    nt, npt, left.ra, left.dec, left.dist, left.dcang, left.x, left.y, left.z,
                    int(qchi_l.shape[0]), int(qchi_l.shape[1]), qchi_l, qlo_l, qhi_l, idx_l,
                    npt1, right.dist, right.dcang, right.x, right.y, right.z,
                    int(qchi_r.shape[0]), int(qchi_r.shape[1]), qchi_r, qlo_r, qhi_r, idx_r,
                    nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                    right.sbound, right.mxh1, right.mxh2, right.mxh3, diag_flag,
                    cntid, _LOG_SINK, progressf, right.sk, right.ll,
                )
            if isinstance(out, tuple):
                dd, qdiag = out[:2]
            else:  # defensive compatibility with old compiled extension
                dd, qdiag = out, None
            if pair_diagnostics and qdiag is not None:
                _set_last_pair_diagnostics(_qchi_diag_dict(
                    "cross", cntid, qdiag, n_left=npt, n_right=npt1,
                    nq_left=int(qchi_l.shape[0]), nq_right=int(qchi_r.shape[0]),
                ))
            return _transpose_counts(dd), _boot_array(nsepp, nsepv), None
        if left_repr == 'grid_chi_exact' and right_repr == 'grid_chi_exact':
            if doboot:
                left_grid = np.asarray(left.pdf_grid, dtype=np.float64)
                right_grid = np.asarray(right.pdf_grid, dtype=np.float64)
                if left_grid.shape != right_grid.shape or not np.allclose(left_grid, right_grid):
                    raise ValueError('grid_chi_exact cross-correlations require the same shared chi-grid on both samples.')
                grid = left_grid
                prob_l = np.asarray(left.pdf_prob_lib, dtype=np.float64, order='F')
                cdf_r = np.asarray(right.pdf_cdf_lib, dtype=np.float64, order='F')
                lo_l = np.asarray(left.pdf_lo_idx, dtype=np.int32)
                hi_l = np.asarray(left.pdf_hi_idx, dtype=np.int32)
                lo_r = np.asarray(right.pdf_lo_idx, dtype=np.int32)
                hi_r = np.asarray(right.pdf_hi_idx, dtype=np.int32)
                idx_l = np.asarray(left.pdf_idx, dtype=np.int32)
                idx_r = np.asarray(right.pdf_idx, dtype=np.int32)
                active_idx_l, active_prob_l, active_start_l, active_count_l = _build_exact_active_support(prob_l, lo_l, hi_l)
                active_idx_l_f = np.asarray(active_idx_l + 1, dtype=np.int32)
                active_start_l_f = np.asarray(active_start_l + 1, dtype=np.int32)
                active_count_l_f = np.asarray(active_count_l, dtype=np.int32)
                rv_search = float(max(left.grid_meta.get('pi_search', 0.0), right.grid_meta.get('pi_search', 0.0), float(pi_edges[-1]) if len(pi_edges) else 0.0))
                if weighted and _has_kernel('rppi_Cb_grid_wp_wg'):
                    dd, bdd = cff.mod.rppi_Cb_grid_wp_wg(
                        nt, npt, left.ra, left.dec, left.dist, left.dcang, left.weights, left.x, left.y, left.z,
                        int(grid.shape[0]), int(prob_l.shape[1]), grid, prob_l, lo_l + 1, hi_l + 1, idx_l,
                        active_idx_l_f, active_start_l_f, active_count_l_f,
                        npt1, right.dist, right.dcang, right.weights, right.x, right.y, right.z,
                        int(cdf_r.shape[1]), cdf_r, lo_r + 1, hi_r + 1, idx_r,
                        nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                        right.sbound, right.mxh1, right.mxh2, right.mxh3, nbts, bseed, 0,
                        cntid, _LOG_SINK, progressf, right.sk, right.ll,
                    )
                    return _transpose_counts(dd), _transpose_bootstrap(bdd), None
                if (not weighted) and _has_kernel('rppi_Cb_grid_wp'):
                    dd, bdd = cff.mod.rppi_Cb_grid_wp(
                        nt, npt, left.ra, left.dec, left.dist, left.dcang, left.x, left.y, left.z,
                        int(grid.shape[0]), int(prob_l.shape[1]), grid, prob_l, lo_l + 1, hi_l + 1, idx_l,
                        active_idx_l_f, active_start_l_f, active_count_l_f,
                        npt1, right.dist, right.dcang, right.x, right.y, right.z,
                        int(cdf_r.shape[1]), cdf_r, lo_r + 1, hi_r + 1, idx_r,
                        nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64), rv_search,
                        right.sbound, right.mxh1, right.mxh2, right.mxh3, nbts, bseed,
                        cntid, _LOG_SINK, progressf, right.sk, right.ll,
                    )
                    return _transpose_counts(dd), _transpose_bootstrap(bdd), None
                raise RuntimeError('grid_chi_exact bootstrap requested but compiled exact-grid bootstrap kernels are unavailable.')
            if dojk and left.region_id is not None and right.region_id is not None and nreg > 0:
                left_grid = np.asarray(left.pdf_grid, dtype=np.float64)
                right_grid = np.asarray(right.pdf_grid, dtype=np.float64)
                if left_grid.shape != right_grid.shape or not np.allclose(left_grid, right_grid):
                    raise ValueError('grid_chi_exact cross-correlations require the same shared chi-grid on both samples.')
                grid = left_grid
                prob_l = np.asarray(left.pdf_prob_lib, dtype=np.float64, order='F')
                cdf_r = np.asarray(right.pdf_cdf_lib, dtype=np.float64, order='F')
                lo_l = np.asarray(left.pdf_lo_idx, dtype=np.int32)
                hi_l = np.asarray(left.pdf_hi_idx, dtype=np.int32)
                lo_r = np.asarray(right.pdf_lo_idx, dtype=np.int32)
                hi_r = np.asarray(right.pdf_hi_idx, dtype=np.int32)
                idx_l = np.asarray(left.pdf_idx, dtype=np.int32)
                idx_r = np.asarray(right.pdf_idx, dtype=np.int32)
                reg_l = np.asarray(left.region_id, dtype=np.int32)
                reg_r = np.asarray(right.region_id, dtype=np.int32)
                active_idx_l, active_prob_l, active_start_l, active_count_l = _build_exact_active_support(prob_l, lo_l, hi_l)
                active_idx_l_f = np.asarray(active_idx_l + 1, dtype=np.int32)
                active_start_l_f = np.asarray(active_start_l + 1, dtype=np.int32)
                active_count_l_f = np.asarray(active_count_l, dtype=np.int32)
                rv_search = float(max(left.grid_meta.get('pi_search', 0.0), right.grid_meta.get('pi_search', 0.0), float(pi_edges[-1]) if len(pi_edges) else 0.0))
                if weighted and _has_kernel('rppi_Cjk_grid_wp_wg'):
                    dd, touch = cff.mod.rppi_Cjk_grid_wp_wg(
                        nt, npt, left.ra, left.dec, left.dist, left.dcang, left.weights, left.x, left.y, left.z,
                        int(grid.shape[0]), int(prob_l.shape[1]), grid, prob_l, lo_l + 1, hi_l + 1, idx_l,
                        active_idx_l_f, active_start_l_f, active_count_l_f, reg_l,
                        npt1, right.dist, right.dcang, right.weights, right.x, right.y, right.z,
                        int(cdf_r.shape[1]), cdf_r, lo_r + 1, hi_r + 1, idx_r, reg_r,
                        int(nreg), nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64),
                        rv_search, right.sbound, right.mxh1, right.mxh2, right.mxh3, 0,
                        cntid, _LOG_SINK, progressf, right.sk, right.ll,
                    )
                    return _transpose_counts(dd), _boot_array(nsepp, nsepv), _transpose_jk_touch(touch)
                if (not weighted) and _has_kernel('rppi_Cjk_grid_wp'):
                    dd, touch = cff.mod.rppi_Cjk_grid_wp(
                        nt, npt, left.ra, left.dec, left.dist, left.dcang, left.x, left.y, left.z,
                        int(grid.shape[0]), int(prob_l.shape[1]), grid, prob_l, lo_l + 1, hi_l + 1, idx_l,
                        active_idx_l_f, active_start_l_f, active_count_l_f, reg_l,
                        npt1, right.dist, right.dcang, right.x, right.y, right.z,
                        int(cdf_r.shape[1]), cdf_r, lo_r + 1, hi_r + 1, idx_r, reg_r,
                        int(nreg), nsepp, rp_edges, nsepv, np.asarray(pi_edges, dtype=np.float64),
                        rv_search, right.sbound, right.mxh1, right.mxh2, right.mxh3,
                        cntid, _LOG_SINK, progressf, right.sk, right.ll,
                    )
                    return _transpose_counts(dd), _boot_array(nsepp, nsepv), _transpose_jk_touch(touch)
                raise RuntimeError('grid_chi_exact jackknife touch requested but compiled exact-grid JK kernels are unavailable.')
            return _run_exact_grid_cross_counts(left, right, rp_edges=rp_edges, pi_edges=pi_edges, weight_mode=weight_mode, nthreads=nthreads, cntid=cntid, progress_file=progress_file)
        rv_search = float(max(left.grid_meta.get('pi_search', 0.0), right.grid_meta.get('pi_search', 0.0), float(pi_edges[-1]) if len(pi_edges) else 0.0))
        prob_floor = float(max(left.grid_meta.get('prob_floor', 1.0e-10), right.grid_meta.get('prob_floor', 1.0e-10)))
        left_idx = np.asarray(left.pdf_idx, dtype=np.int32)
        right_idx = np.asarray(right.pdf_idx, dtype=np.int32)
        left_nlib = int(getattr(left.pdf_alpha_lib, "shape", (0, 0))[1])
        right_nlib = int(getattr(right.pdf_alpha_lib, "shape", (0, 0))[1])
        if dojk and left.region_id is not None and right.region_id is not None and nreg > 0:
            reg = np.asarray(left.region_id, dtype=np.int32)
            reg1 = np.asarray(right.region_id, dtype=np.int32)
            if weighted and _has_kernel('rppi_Cjk_gmm_wp_wg'):
                dd, touch = cff.mod.rppi_Cjk_gmm_wp_wg(
                    nt, npt, left.ra, left.dec, left.dist, left.dcang, left.weights, left.x, left.y, left.z,
                    int(left.pdf_k), left_nlib, left.pdf_alpha_lib, left.pdf_mu_lib, left.pdf_sig_lib, left_idx, reg,
                    npt1, right.dist, right.dcang, right.weights, right.x, right.y, right.z,
                    int(right.pdf_k), right_nlib, right.pdf_alpha_lib, right.pdf_mu_lib, right.pdf_sig_lib, right_idx, reg1,
                    nreg, nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, right.sbound,
                    right.mxh1, right.mxh2, right.mxh3, 0, cntid, _LOG_SINK, progressf,
                    right.sk, right.ll,
                )
                return _transpose_counts(dd), _boot_array(nsepp, nsepv), _transpose_jk_touch(touch)
            if (not weighted) and _has_kernel('rppi_Cjk_gmm_wp'):
                dd, touch = cff.mod.rppi_Cjk_gmm_wp(
                    nt, npt, left.ra, left.dec, left.dist, left.dcang, left.x, left.y, left.z,
                    int(left.pdf_k), left_nlib, left.pdf_alpha_lib, left.pdf_mu_lib, left.pdf_sig_lib, left_idx, reg,
                    npt1, right.dist, right.dcang, right.x, right.y, right.z,
                    int(right.pdf_k), right_nlib, right.pdf_alpha_lib, right.pdf_mu_lib, right.pdf_sig_lib, right_idx, reg1,
                    nreg, nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, right.sbound,
                    right.mxh1, right.mxh2, right.mxh3, cntid, _LOG_SINK, progressf,
                    right.sk, right.ll,
                )
                return _transpose_counts(dd), _boot_array(nsepp, nsepv), _transpose_jk_touch(touch)
        if doboot:
            if weighted and _has_kernel('rppi_Cb_gmm_wp_wg'):
                dd, bdd = cff.mod.rppi_Cb_gmm_wp_wg(
                    nt, npt, left.ra, left.dec, left.dist, left.dcang, left.weights, left.x, left.y, left.z,
                    int(left.pdf_k), left_nlib, left.pdf_alpha_lib, left.pdf_mu_lib, left.pdf_sig_lib, left_idx,
                    npt1, right.dist, right.dcang, right.weights, right.x, right.y, right.z,
                    int(right.pdf_k), right_nlib, right.pdf_alpha_lib, right.pdf_mu_lib, right.pdf_sig_lib, right_idx,
                    nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, right.sbound,
                    right.mxh1, right.mxh2, right.mxh3, nbts, bseed, 0, cntid, _LOG_SINK, progressf,
                    right.sk, right.ll,
                )
                return _transpose_counts(dd), _transpose_bootstrap(bdd), None
            if (not weighted) and _has_kernel('rppi_Cb_gmm_wp'):
                dd, bdd = cff.mod.rppi_Cb_gmm_wp(
                    nt, npt, left.ra, left.dec, left.dist, left.dcang, left.x, left.y, left.z,
                    int(left.pdf_k), left_nlib, left.pdf_alpha_lib, left.pdf_mu_lib, left.pdf_sig_lib, left_idx,
                    npt1, right.dist, right.dcang, right.x, right.y, right.z,
                    int(right.pdf_k), right_nlib, right.pdf_alpha_lib, right.pdf_mu_lib, right.pdf_sig_lib, right_idx,
                    nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, right.sbound,
                    right.mxh1, right.mxh2, right.mxh3, nbts, bseed, cntid, _LOG_SINK, progressf,
                    right.sk, right.ll,
                )
                return _transpose_counts(dd), _transpose_bootstrap(bdd), None
            raise RuntimeError('PDF bootstrap requested but the compiled GMM bootstrap kernels are not available. Rebuild the extension module or use the rerun backend.')
        if weighted and _has_kernel('rppi_C_gmm_wp_wg'):
            dd = cff.mod.rppi_C_gmm_wp_wg(
                nt, npt, left.ra, left.dec, left.dist, left.dcang, left.weights, left.x, left.y, left.z,
                int(left.pdf_k), left_nlib, left.pdf_alpha_lib, left.pdf_mu_lib, left.pdf_sig_lib, left_idx,
                npt1, right.dist, right.dcang, right.weights, right.x, right.y, right.z,
                int(right.pdf_k), right_nlib, right.pdf_alpha_lib, right.pdf_mu_lib, right.pdf_sig_lib, right_idx,
                nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, right.sbound,
                right.mxh1, right.mxh2, right.mxh3, 0, cntid, _LOG_SINK, progressf,
                right.sk, right.ll,
            )
            return _transpose_counts(dd), _boot_array(nsepp, nsepv), None
        if (not weighted) and _has_kernel('rppi_C_gmm_wp'):
            dd = cff.mod.rppi_C_gmm_wp(
                nt, npt, left.ra, left.dec, left.dist, left.dcang, left.x, left.y, left.z,
                int(left.pdf_k), left_nlib, left.pdf_alpha_lib, left.pdf_mu_lib, left.pdf_sig_lib, left_idx,
                npt1, right.dist, right.dcang, right.x, right.y, right.z,
                int(right.pdf_k), right_nlib, right.pdf_alpha_lib, right.pdf_mu_lib, right.pdf_sig_lib, right_idx,
                nsepp, rp_edges, nsepv, pi_edges, rv_search, prob_floor, right.sbound,
                right.mxh1, right.mxh2, right.mxh3, cntid, _LOG_SINK, progressf,
                right.sk, right.ll,
            )
            return _transpose_counts(dd), _boot_array(nsepp, nsepv), None
        raise RuntimeError('PDF mode requested but the compiled GMM kernels are not available. Rebuild the extension module.')
    # ------------------------------------------------------------------------
    if dojk and left.region_id is not None and right.region_id is not None and nreg > 0:
        reg = np.asarray(left.region_id, dtype=np.int32)
        reg1 = np.asarray(right.region_id, dtype=np.int32)
        if weighted and _has_kernel("rppi_Cjk_wg"):
            dd, touch = cff.mod.rppi_Cjk_wg(
                nt, npt, left.ra, left.dec, left.dist, left.weights, left.x, left.y, left.z, reg,
                npt1, right.dist, right.weights, right.x, right.y, right.z, reg1,
                nreg, nsepp, rp_edges, nsepv, pi_edges, right.sbound,
                right.mxh1, right.mxh2, right.mxh3, 0, cntid, _LOG_SINK, progressf,
                right.sk, right.ll,
            )
            return _transpose_counts(dd), _boot_array(nsepp, nsepv), _transpose_jk_touch(touch)
        if (not weighted) and _has_kernel("rppi_Cjk"):
            dd, touch = cff.mod.rppi_Cjk(
                nt, npt, left.ra, left.dec, left.dist, left.x, left.y, left.z, reg,
                npt1, right.dist, right.x, right.y, right.z, reg1,
                nreg, nsepp, rp_edges, nsepv, pi_edges, right.sbound,
                right.mxh1, right.mxh2, right.mxh3, cntid, _LOG_SINK, progressf,
                right.sk, right.ll,
            )
            return _transpose_counts(dd), _boot_array(nsepp, nsepv), _transpose_jk_touch(touch)
    if doboot:
        if weighted:
            dd, bdd = cff.mod.rppi_Cb_wg(nt, npt, left.ra, left.dec, left.dist, left.weights, left.x, left.y, left.z, npt1, right.dist, right.weights, right.x, right.y, right.z, nsepp, rp_edges, nsepv, pi_edges, right.sbound, right.mxh1, right.mxh2, right.mxh3, nbts, bseed, 0, cntid, _LOG_SINK, progressf, right.sk, right.ll)
        else:
            dd, bdd = cff.mod.rppi_Cb(nt, npt, left.ra, left.dec, left.dist, left.x, left.y, left.z, npt1, right.dist, right.x, right.y, right.z, nsepp, rp_edges, nsepv, pi_edges, right.sbound, right.mxh1, right.mxh2, right.mxh3, nbts, bseed, cntid, _LOG_SINK, progressf, right.sk, right.ll)
        return _transpose_counts(dd), _transpose_bootstrap(bdd), None
    if weighted:
        dd = cff.mod.rppi_C_wg(nt, npt, left.ra, left.dec, left.dist, left.weights, left.x, left.y, left.z, npt1, right.dist, right.weights, right.x, right.y, right.z, nsepp, rp_edges, nsepv, pi_edges, right.sbound, right.mxh1, right.mxh2, right.mxh3, 0, cntid, _LOG_SINK, progressf, right.sk, right.ll)
    else:
        dd = cff.mod.rppi_C(nt, npt, left.ra, left.dec, left.dist, left.x, left.y, left.z, npt1, right.dist, right.x, right.y, right.z, nsepp, rp_edges, nsepv, pi_edges, right.sbound, right.mxh1, right.mxh2, right.mxh3, cntid, _LOG_SINK, progressf, right.sk, right.ll)
    return _transpose_counts(dd), _boot_array(nsepp, nsepv), None


def build_auto_count_result(data: PreparedProjectedSample, *, rp_edges, rp_centers, pi_edges, pi_centers, pi_delta, nthreads: int, weight_mode: str, doboot: bool, dojk: bool = False, nreg: int = 0, nbts: int, bseed: int, progress_file: str | None = None, pair_diagnostics: bool = False):
    t0 = time.perf_counter()
    dd, bdd, normb, sumwb, dd_touch = run_rppi_auto_counts(data, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=doboot, dojk=dojk, nreg=nreg, nbts=nbts, bseed=bseed, cntid="DD", progress_file=progress_file, pair_diagnostics=pair_diagnostics)
    dd_elapsed = time.perf_counter() - t0
    dd_diag = _consume_last_pair_diagnostics()
    metadata = {
        "n_data": int(data.nrows),
        "data_weighted": not data.wunit,
        "jk_nregions": int(nreg) if dojk else 0,
        "jk_touch_available": dd_touch is not None,
        "pdf_repr": _pdf_repr(data),
        "pdf_nquant": int(getattr(data, "pdf_k", 0) or 0) if _pdf_repr(data) == "quantile_chi" else None,
        "qchi_prepare": _qchi_prepare_metadata(data),
        "qchi_diagnostics_enabled": bool(pair_diagnostics and _pdf_repr(data) == "quantile_chi"),
    }
    _attach_timing_and_diag("DD", dd_elapsed, dd_diag, metadata)
    return ProjectedAutoCountsResult(
        rp_edges=np.asarray(rp_edges),
        rp_centers=np.asarray(rp_centers),
        pi_edges=np.asarray(pi_edges),
        pi_centers=np.asarray(pi_centers),
        dd=dd,
        dd_boot=bdd,
        norm_dd_boot=normb,
        sum_w_data_boot=sumwb,
        intpi_dd=_integrate_pi(dd, pi_delta),
        metadata=metadata,
    )

def build_auto_counts(data: PreparedProjectedSample, random: PreparedProjectedSample, *, rp_edges, rp_centers, pi_edges, pi_centers, pi_delta, nthreads: int, estimator: str, weight_mode: str, doboot: bool, dojk: bool = False, nreg: int = 0, nbts: int, bseed: int, progress_file: str | None = None, split_random=None, pair_diagnostics: bool = False):
    """Build projected auto-correlation count terms.

    Parameters
    ----------
    data, random : PreparedProjectedSample
        Prepared data and random catalogs.
    rp_edges, rp_centers, pi_edges, pi_centers, pi_delta : array-like
        Projected and line-of-sight binning arrays.
    nthreads : int
        Number of OpenMP threads for the Fortran kernels.
    estimator : str
        Pair-count estimator family. ``split_random`` is honored only for
        ``estimator="LS"``.
    weight_mode : str
        Weighting mode forwarded to the kernels.
    doboot, dojk : bool
        Enable bootstrap or jackknife bookkeeping. Split-random RR currently
        disables the jackknife touch fast path.
    nreg, nbts, bseed : int
        Resampling controls forwarded to the kernels.
    progress_file : str or None, optional
        Optional progress sink used by the notebook/CLI progress layer.
    split_random : SplitRandomSpec or None, optional
        Optional split-random RR configuration. When enabled, ``DR`` remains a
        full count while ``RR`` is accumulated over shuffled, re-autogridded
        random chunks.

    Returns
    -------
    ProjectedAutoCounts
        Count container with ``dd``, ``rr``, ``dr`` and metadata. In
        split-random mode the metadata includes the chunk sizes and the exact
        ``rr_norm_pairs`` value used by the LS estimator.
    """
    t0 = time.perf_counter()
    dd, bdd, normb, sumwb, dd_touch = run_rppi_auto_counts(data, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=doboot, dojk=dojk, nreg=nreg, nbts=nbts, bseed=bseed, cntid="DD", progress_file=progress_file, pair_diagnostics=pair_diagnostics)
    dd_elapsed = time.perf_counter() - t0
    dd_diag = _consume_last_pair_diagnostics()
    rr = dr = rr_touch = dr_touch = None
    rr_elapsed = dr_elapsed = None
    rr_diag = dr_diag = None
    split_meta = {}
    rr_norm_pairs = _rr_norm_pairs_full(int(random.nrows))
    if estimator in {"NAT", "LS"}:
        use_split_rr = bool(getattr(split_random, "enabled", False))
        if use_split_rr:
            rr, rr_norm_pairs, split_meta = _run_rppi_split_rr_counts(random, rp_edges=rp_edges, pi_edges=pi_edges, pi_delta=pi_delta, nthreads=nthreads, dojk=dojk, nreg=nreg, bseed=bseed, progress_file=progress_file, split_random=split_random, n_data=int(data.nrows), pair_diagnostics=pair_diagnostics)
            rr_elapsed = float(split_meta.get("split_random_rr_total_time_s", 0.0))
            rr_diag = None
            rr_touch = None
        else:
            t0 = time.perf_counter()
            rr, _, _, _, rr_touch = run_rppi_auto_counts(random, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode="unweighted", doboot=False, dojk=dojk, nreg=nreg, nbts=0, bseed=bseed, cntid="RR", progress_file=progress_file, pair_diagnostics=pair_diagnostics)
            rr_elapsed = time.perf_counter() - t0
            rr_diag = _consume_last_pair_diagnostics()
    if estimator in {"DP", "LS"}:
        t0 = time.perf_counter()
        dr, _, dr_touch = run_rppi_cross_counts(data, random, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=False, dojk=dojk, nreg=nreg, nbts=0, bseed=bseed, cntid="DR", progress_file=progress_file, pair_diagnostics=pair_diagnostics)
        dr_elapsed = time.perf_counter() - t0
        dr_diag = _consume_last_pair_diagnostics()
    metadata = {
        "n_data": int(data.nrows),
        "n_random": int(random.nrows),
        "data_weighted": not data.wunit,
        "jk_nregions": int(nreg) if dojk else 0,
        "jk_touch_available": (dd_touch is not None and not bool(getattr(split_random, "enabled", False))),
        "rr_norm_pairs": rr_norm_pairs,
        "pdf_repr": _pdf_repr(data),
        "pdf_nquant": int(getattr(data, "pdf_k", 0) or 0) if _pdf_repr(data) == "quantile_chi" else None,
        "qchi_prepare_data": _qchi_prepare_metadata(data),
        "qchi_prepare_random": _qchi_prepare_metadata(random),
        "qchi_diagnostics_enabled": bool(pair_diagnostics and (_pdf_repr(data) == "quantile_chi" or _pdf_repr(random) == "quantile_chi")),
    }
    _attach_timing_and_diag("DD", dd_elapsed, dd_diag, metadata)
    if rr_elapsed is not None:
        _attach_timing_and_diag("RR", rr_elapsed, rr_diag, metadata)
    if dr_elapsed is not None:
        _attach_timing_and_diag("DR", dr_elapsed, dr_diag, metadata)
    metadata.update(split_meta)
    return ProjectedAutoCounts(rp_edges=np.asarray(rp_edges), rp_centers=np.asarray(rp_centers), pi_edges=np.asarray(pi_edges), pi_centers=np.asarray(pi_centers), dd=dd, rr=rr, dr=dr, dd_boot=bdd, norm_dd_boot=normb, sum_w_data_boot=sumwb, dd_jk_touch=dd_touch, rr_jk_touch=rr_touch, dr_jk_touch=dr_touch, intpi_dd=_integrate_pi(dd, pi_delta), intpi_rr=_integrate_pi(rr, pi_delta), intpi_dr=_integrate_pi(dr, pi_delta), metadata=metadata)


def build_cross_count_result(data1: PreparedProjectedSample, data2: PreparedProjectedSample, *, rp_edges, rp_centers, pi_edges, pi_centers, pi_delta, nthreads: int, weight_mode: str, doboot: bool, dojk: bool = False, nreg: int = 0, nbts: int, bseed: int, primary: str = "data1", progress_file: str | None = None, pair_diagnostics: bool = False):
    if primary == "data2":
        data1, data2 = data2, data1
    t0 = time.perf_counter()
    d1d2, b_d1d2, d1d2_touch = run_rppi_cross_counts(data1, data2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=doboot, dojk=dojk, nreg=nreg, nbts=nbts, bseed=bseed, cntid="12", progress_file=progress_file, pair_diagnostics=pair_diagnostics)
    elapsed = time.perf_counter() - t0
    diag = _consume_last_pair_diagnostics()
    metadata = {
        "n_data1": int(data1.nrows),
        "n_data2": int(data2.nrows),
        "primary": primary,
        "jk_nregions": int(nreg) if dojk else 0,
        "jk_touch_available": d1d2_touch is not None,
        "pdf_repr1": _pdf_repr(data1),
        "pdf_repr2": _pdf_repr(data2),
        "qchi_prepare_data1": _qchi_prepare_metadata(data1),
        "qchi_prepare_data2": _qchi_prepare_metadata(data2),
        "qchi_diagnostics_enabled": bool(pair_diagnostics and (_pdf_repr(data1) == "quantile_chi" or _pdf_repr(data2) == "quantile_chi")),
    }
    _attach_timing_and_diag("12", elapsed, diag, metadata)
    return ProjectedCrossCountsResult(rp_edges=np.asarray(rp_edges), rp_centers=np.asarray(rp_centers), pi_edges=np.asarray(pi_edges), pi_centers=np.asarray(pi_centers), d1d2=d1d2, d1d2_boot=b_d1d2, intpi_d1d2=_integrate_pi(d1d2, pi_delta), metadata=metadata)

def build_cross_counts(data1: PreparedProjectedSample, random1: PreparedProjectedSample | None, data2: PreparedProjectedSample, random2: PreparedProjectedSample | None, *, rp_edges, rp_centers, pi_edges, pi_centers, pi_delta, nthreads: int, estimator: str, weight_mode: str, doboot: bool, dojk: bool = False, nreg: int = 0, nbts: int, bseed: int, primary: str = "data1", progress_file: str | None = None, pair_diagnostics: bool = False):
    if primary == "data2":
        data1, random1, data2, random2 = data2, random2, data1, random1
    t0 = time.perf_counter()
    d1d2, b_d1d2, d1d2_touch = run_rppi_cross_counts(data1, data2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=doboot, dojk=dojk, nreg=nreg, nbts=nbts, bseed=bseed, cntid="12", progress_file=progress_file, pair_diagnostics=pair_diagnostics)
    d1d2_elapsed = time.perf_counter() - t0
    d1d2_diag = _consume_last_pair_diagnostics()
    d1r2 = r1d2 = r1r2 = b_d1r2 = None
    d1r2_elapsed = r1d2_elapsed = r1r2_elapsed = None
    d1r2_diag = r1d2_diag = r1r2_diag = None
    d1r2_touch = r1d2_touch = r1r2_touch = None
    if estimator in {"DP", "LS"}:
        if random2 is None:
            raise ValueError("random2 is required for projected cross DP/LS in the estimator-aware contract.")
        t0 = time.perf_counter()
        d1r2, b_d1r2, d1r2_touch = run_rppi_cross_counts(data1, random2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=doboot, dojk=dojk, nreg=nreg, nbts=nbts, bseed=bseed, cntid="1R", progress_file=progress_file, pair_diagnostics=pair_diagnostics)
        d1r2_elapsed = time.perf_counter() - t0
        d1r2_diag = _consume_last_pair_diagnostics()
    if estimator == "LS":
        if random1 is None or random2 is None:
            raise ValueError("random1 and random2 are required for projected cross LS.")
        t0 = time.perf_counter()
        r1d2, _, r1d2_touch = run_rppi_cross_counts(random1, data2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode="unweighted", doboot=False, dojk=dojk, nreg=nreg, nbts=0, bseed=bseed, cntid="R2", progress_file=progress_file, pair_diagnostics=pair_diagnostics)
        r1d2_elapsed = time.perf_counter() - t0
        r1d2_diag = _consume_last_pair_diagnostics()
        t0 = time.perf_counter()
        r1r2, _, r1r2_touch = run_rppi_cross_counts(random1, random2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode="unweighted", doboot=False, dojk=dojk, nreg=nreg, nbts=0, bseed=bseed, cntid="RR", progress_file=progress_file, pair_diagnostics=pair_diagnostics)
        r1r2_elapsed = time.perf_counter() - t0
        r1r2_diag = _consume_last_pair_diagnostics()
    elif estimator == "NAT":
        if random1 is None or random2 is None:
            raise ValueError("random1 and random2 are required for projected cross NAT.")
        t0 = time.perf_counter()
        r1r2, _, r1r2_touch = run_rppi_cross_counts(random1, random2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode="unweighted", doboot=False, dojk=dojk, nreg=nreg, nbts=0, bseed=bseed, cntid="RR", progress_file=progress_file, pair_diagnostics=pair_diagnostics)
        r1r2_elapsed = time.perf_counter() - t0
        r1r2_diag = _consume_last_pair_diagnostics()
    metadata = {"n_data1": int(data1.nrows), "n_random1": 0 if random1 is None else int(random1.nrows), "n_data2": int(data2.nrows), "n_random2": 0 if random2 is None else int(random2.nrows), "primary": primary, "jk_nregions": int(nreg) if dojk else 0, "jk_touch_available": d1d2_touch is not None,
                "pdf_repr1": _pdf_repr(data1), "pdf_repr2": _pdf_repr(data2),
                "qchi_prepare_data1": _qchi_prepare_metadata(data1), "qchi_prepare_data2": _qchi_prepare_metadata(data2),
                "qchi_prepare_random1": None if random1 is None else _qchi_prepare_metadata(random1),
                "qchi_prepare_random2": None if random2 is None else _qchi_prepare_metadata(random2),
                "qchi_diagnostics_enabled": bool(pair_diagnostics and any(_pdf_repr(s) == "quantile_chi" for s in (data1, data2, random1, random2) if s is not None))}
    _attach_timing_and_diag("12", d1d2_elapsed, d1d2_diag, metadata)
    if d1r2_elapsed is not None:
        _attach_timing_and_diag("1R", d1r2_elapsed, d1r2_diag, metadata)
    if r1d2_elapsed is not None:
        _attach_timing_and_diag("R2", r1d2_elapsed, r1d2_diag, metadata)
    if r1r2_elapsed is not None:
        _attach_timing_and_diag("RR", r1r2_elapsed, r1r2_diag, metadata)
    return ProjectedCrossCounts(rp_edges=np.asarray(rp_edges), rp_centers=np.asarray(rp_centers), pi_edges=np.asarray(pi_edges), pi_centers=np.asarray(pi_centers), d1d2=d1d2, d1r2=d1r2, r1d2=r1d2, r1r2=r1r2, d1d2_boot=b_d1d2, d1r2_boot=b_d1r2, d1d2_jk_touch=d1d2_touch, d1r2_jk_touch=d1r2_touch, r1d2_jk_touch=r1d2_touch, r1r2_jk_touch=r1r2_touch, intpi_d1d2=_integrate_pi(d1d2, pi_delta), intpi_d1r2=_integrate_pi(d1r2, pi_delta), intpi_r1d2=_integrate_pi(r1d2, pi_delta), intpi_r1r2=_integrate_pi(r1r2, pi_delta), metadata=metadata)

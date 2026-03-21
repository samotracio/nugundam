"""Adapters between prepared projected samples and the compiled pair counters."""
from __future__ import annotations

import os

import numpy as np

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


def _run_rppi_split_rr_counts(random: PreparedProjectedSample, *, rp_edges, pi_edges, pi_delta, nthreads: int, dojk: bool, nreg: int, bseed: int, progress_file: str | None, split_random, n_data: int):
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
    nchunks = len(chunks)
    for ichunk, idx in enumerate(chunks, start=1):
        chunk = subset_prepared_projected_sample(random, idx, pi_edges=pi_edges, regrid=True)
        cntid = f"RR split {ichunk}/{nchunks}" if nchunks > 1 else "RR"
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
        )
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
):
    nt = set_threads(nthreads)
    npt = int(data.nrows)
    weighted = _resolve_weight_mode(weight_mode, data.wunit)
    progressf = _progress_arg(progress_file)
    nsepp = len(rp_edges) - 1
    nsepv = len(pi_edges) - 1
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
):
    nt = set_threads(nthreads)
    npt = int(left.nrows)
    npt1 = int(right.nrows)
    weighted = _resolve_weight_mode(weight_mode, left.wunit, right.wunit)
    progressf = _progress_arg(progress_file)
    nsepp = len(rp_edges) - 1
    nsepv = len(pi_edges) - 1
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


def build_auto_count_result(data: PreparedProjectedSample, *, rp_edges, rp_centers, pi_edges, pi_centers, pi_delta, nthreads: int, weight_mode: str, doboot: bool, dojk: bool = False, nreg: int = 0, nbts: int, bseed: int, progress_file: str | None = None):
    dd, bdd, normb, sumwb, dd_touch = run_rppi_auto_counts(data, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=doboot, dojk=dojk, nreg=nreg, nbts=nbts, bseed=bseed, cntid="DD", progress_file=progress_file)
    return ProjectedAutoCountsResult(rp_edges=np.asarray(rp_edges), rp_centers=np.asarray(rp_centers), pi_edges=np.asarray(pi_edges), pi_centers=np.asarray(pi_centers), dd=dd, dd_boot=bdd, norm_dd_boot=normb, sum_w_data_boot=sumwb, intpi_dd=_integrate_pi(dd, pi_delta), metadata={"n_data": int(data.nrows), "data_weighted": not data.wunit, "jk_nregions": int(nreg) if dojk else 0, "jk_touch_available": dd_touch is not None})


def build_auto_counts(data: PreparedProjectedSample, random: PreparedProjectedSample, *, rp_edges, rp_centers, pi_edges, pi_centers, pi_delta, nthreads: int, estimator: str, weight_mode: str, doboot: bool, dojk: bool = False, nreg: int = 0, nbts: int, bseed: int, progress_file: str | None = None, split_random=None):
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
    dd, bdd, normb, sumwb, dd_touch = run_rppi_auto_counts(data, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=doboot, dojk=dojk, nreg=nreg, nbts=nbts, bseed=bseed, cntid="DD", progress_file=progress_file)
    rr = dr = rr_touch = dr_touch = None
    split_meta = {}
    rr_norm_pairs = _rr_norm_pairs_full(int(random.nrows))
    if estimator in {"NAT", "LS"}:
        use_split_rr = bool(getattr(split_random, "enabled", False))
        if use_split_rr:
            rr, rr_norm_pairs, split_meta = _run_rppi_split_rr_counts(random, rp_edges=rp_edges, pi_edges=pi_edges, pi_delta=pi_delta, nthreads=nthreads, dojk=dojk, nreg=nreg, bseed=bseed, progress_file=progress_file, split_random=split_random, n_data=int(data.nrows))
            rr_touch = None
        else:
            rr, _, _, _, rr_touch = run_rppi_auto_counts(random, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode="unweighted", doboot=False, dojk=dojk, nreg=nreg, nbts=0, bseed=bseed, cntid="RR", progress_file=progress_file)
    if estimator in {"DP", "LS"}:
        dr, _, dr_touch = run_rppi_cross_counts(data, random, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=False, dojk=dojk, nreg=nreg, nbts=0, bseed=bseed, cntid="DR", progress_file=progress_file)
    metadata = {"n_data": int(data.nrows), "n_random": int(random.nrows), "data_weighted": not data.wunit, "jk_nregions": int(nreg) if dojk else 0, "jk_touch_available": (dd_touch is not None and not bool(getattr(split_random, "enabled", False))), "rr_norm_pairs": rr_norm_pairs}
    metadata.update(split_meta)
    return ProjectedAutoCounts(rp_edges=np.asarray(rp_edges), rp_centers=np.asarray(rp_centers), pi_edges=np.asarray(pi_edges), pi_centers=np.asarray(pi_centers), dd=dd, rr=rr, dr=dr, dd_boot=bdd, norm_dd_boot=normb, sum_w_data_boot=sumwb, dd_jk_touch=dd_touch, rr_jk_touch=rr_touch, dr_jk_touch=dr_touch, intpi_dd=_integrate_pi(dd, pi_delta), intpi_rr=_integrate_pi(rr, pi_delta), intpi_dr=_integrate_pi(dr, pi_delta), metadata=metadata)


def build_cross_count_result(data1: PreparedProjectedSample, data2: PreparedProjectedSample, *, rp_edges, rp_centers, pi_edges, pi_centers, pi_delta, nthreads: int, weight_mode: str, doboot: bool, dojk: bool = False, nreg: int = 0, nbts: int, bseed: int, primary: str = "data1", progress_file: str | None = None):
    if primary == "data2":
        data1, data2 = data2, data1
    d1d2, b_d1d2, d1d2_touch = run_rppi_cross_counts(data1, data2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=doboot, dojk=dojk, nreg=nreg, nbts=nbts, bseed=bseed, cntid="12", progress_file=progress_file)
    return ProjectedCrossCountsResult(rp_edges=np.asarray(rp_edges), rp_centers=np.asarray(rp_centers), pi_edges=np.asarray(pi_edges), pi_centers=np.asarray(pi_centers), d1d2=d1d2, d1d2_boot=b_d1d2, intpi_d1d2=_integrate_pi(d1d2, pi_delta), metadata={"n_data1": int(data1.nrows), "n_data2": int(data2.nrows), "primary": primary, "jk_nregions": int(nreg) if dojk else 0, "jk_touch_available": d1d2_touch is not None})


def build_cross_counts(data1: PreparedProjectedSample, random1: PreparedProjectedSample | None, data2: PreparedProjectedSample, random2: PreparedProjectedSample | None, *, rp_edges, rp_centers, pi_edges, pi_centers, pi_delta, nthreads: int, estimator: str, weight_mode: str, doboot: bool, dojk: bool = False, nreg: int = 0, nbts: int, bseed: int, primary: str = "data1", progress_file: str | None = None):
    if primary == "data2":
        data1, random1, data2, random2 = data2, random2, data1, random1
    d1d2, b_d1d2, d1d2_touch = run_rppi_cross_counts(data1, data2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=doboot, dojk=dojk, nreg=nreg, nbts=nbts, bseed=bseed, cntid="12", progress_file=progress_file)
    d1r2 = r1d2 = r1r2 = b_d1r2 = None
    d1r2_touch = r1d2_touch = r1r2_touch = None
    if estimator in {"DP", "LS"}:
        if random2 is None:
            raise ValueError("random2 is required for projected cross DP/LS in the estimator-aware contract.")
        d1r2, b_d1r2, d1r2_touch = run_rppi_cross_counts(data1, random2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode=weight_mode, doboot=doboot, dojk=dojk, nreg=nreg, nbts=nbts, bseed=bseed, cntid="1R", progress_file=progress_file)
    if estimator == "LS":
        if random1 is None or random2 is None:
            raise ValueError("random1 and random2 are required for projected cross LS.")
        r1d2, _, r1d2_touch = run_rppi_cross_counts(random1, data2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode="unweighted", doboot=False, dojk=dojk, nreg=nreg, nbts=0, bseed=bseed, cntid="R2", progress_file=progress_file)
        r1r2, _, r1r2_touch = run_rppi_cross_counts(random1, random2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode="unweighted", doboot=False, dojk=dojk, nreg=nreg, nbts=0, bseed=bseed, cntid="RR", progress_file=progress_file)
    elif estimator == "NAT":
        if random1 is None or random2 is None:
            raise ValueError("random1 and random2 are required for projected cross NAT.")
        r1r2, _, r1r2_touch = run_rppi_cross_counts(random1, random2, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=nthreads, weight_mode="unweighted", doboot=False, dojk=dojk, nreg=nreg, nbts=0, bseed=bseed, cntid="RR", progress_file=progress_file)
    return ProjectedCrossCounts(rp_edges=np.asarray(rp_edges), rp_centers=np.asarray(rp_centers), pi_edges=np.asarray(pi_edges), pi_centers=np.asarray(pi_centers), d1d2=d1d2, d1r2=d1r2, r1d2=r1d2, r1r2=r1r2, d1d2_boot=b_d1d2, d1r2_boot=b_d1r2, d1d2_jk_touch=d1d2_touch, d1r2_jk_touch=d1r2_touch, r1d2_jk_touch=r1d2_touch, r1r2_jk_touch=r1r2_touch, intpi_d1d2=_integrate_pi(d1d2, pi_delta), intpi_d1r2=_integrate_pi(d1r2, pi_delta), intpi_r1d2=_integrate_pi(r1d2, pi_delta), intpi_r1r2=_integrate_pi(r1r2, pi_delta), metadata={"n_data1": int(data1.nrows), "n_random1": 0 if random1 is None else int(random1.nrows), "n_data2": int(data2.nrows), "n_random2": 0 if random2 is None else int(random2.nrows), "primary": primary, "jk_nregions": int(nreg) if dojk else 0, "jk_touch_available": d1d2_touch is not None})

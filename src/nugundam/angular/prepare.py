"""Angular sample preparation, gridding, and sorting utilities."""
from __future__ import annotations

from typing import Iterable

import numpy as np

import nugundam.cflibfor as cff

from ..core.catalogs import catalog_get_column, catalog_has_column, catalog_nrows
from ..core.common import bound2d, cross0guess, makebins, radec2xyz
from ..core.jackknife import build_shared_sky_regions, choose_default_nregions, normalize_region_labels
from .models import AngularAutoConfig, AngularCrossConfig, PreparedAngularSample


def _resolve_autogrid_mode(value) -> str:
    """
    Resolve autogrid mode.
    
    Parameters
    ----------
    value : object
        Value for ``value``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if value is False:
        return "manual"
    if value is True or value is None:
        return "legacy"
    mode = str(value).strip().lower()
    if mode in {"manual", "false", "none"}:
        return "manual"
    if mode == "legacy":
        return "legacy"
    if mode in {"adaptive", "countbox", "runtime"}:
        return "adaptive"
    raise ValueError(f"Unknown autogrid mode {value!r}. Use False, True/'legacy', or 'adaptive'.")


def _normalize_pxorder(value) -> str:
    """
    Normalize pxorder.
    
    Parameters
    ----------
    value : object
        Value for ``value``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if value is None:
        return "none"
    mode = str(value).strip().lower()
    if mode in {"none", "null"}:
        return "none"
    if mode in {"natural", "grid", "cell"}:
        return "natural"
    if mode in {"cell-dec", "cell_dec", "celldec"}:
        return "cell-dec"
    raise NameError(
        f"Order {value!r} not implemented. Use 'none', 'natural', or 'cell-dec'."
    )


def _col(table, name: str, *, dtype=None):
    return catalog_get_column(table, name, dtype=dtype)


def _shared_user_region_ids(pairs):
    arrays = []
    labels = []
    for table, region_col in pairs:
        arr = np.asarray(_col(table, region_col))
        arrays.append(arr)
        labels.append(arr)
    full = normalize_region_labels(np.concatenate(labels))
    out = []
    start = 0
    for arr in arrays:
        stop = start + len(arr)
        out.append(full[start:stop])
        start = stop
    return out


def _auto_region_ids(catalogs, *, nregions: int, seed: int, geometry_from: str = "auto"):
    random_catalogs = [item for item in catalogs if item[3] == "random"]
    data_catalogs = [item for item in catalogs if item[3] == "data"]
    if geometry_from == "randoms":
        geometry = random_catalogs or data_catalogs
    elif geometry_from == "data":
        geometry = data_catalogs or random_catalogs
    else:
        geometry = random_catalogs or data_catalogs
    geometry_pairs = [(_col(table, ra_col), _col(table, dec_col)) for table, ra_col, dec_col, _kind in geometry]
    assign_pairs = [(_col(table, ra_col), _col(table, dec_col)) for table, ra_col, dec_col, _kind in catalogs]
    assignments, centers = build_shared_sky_regions(geometry_pairs, assign_pairs, nregions=nregions, seed=seed)
    return assignments, centers


def _sort_index_2d_arrays(ra, dec, *, sbound, mxh1: int, mxh2: int, pxorder: str = "natural") -> np.ndarray:
    pxmode = _normalize_pxorder(pxorder)
    if pxmode == "none":
        return np.arange(len(ra), dtype=np.int64)
    qdec, qra = _grid_cell_indices(np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64), sbound=sbound, mxh1=mxh1, mxh2=mxh2)
    if pxmode == "natural":
        return np.lexsort((qra, qdec))
    if pxmode == "cell-dec":
        return np.lexsort((np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64), qra, qdec))
    raise NameError(
        f"Order {pxorder!r} not implemented. Use 'none', 'natural', or 'cell-dec'."
    )


def subset_prepared_angular_sample(sample: PreparedAngularSample, keep, *, regrid: bool = False, theta_edges=None) -> PreparedAngularSample:
    idx = np.asarray(keep)
    if idx.dtype == bool:
        idx = np.flatnonzero(idx)
    else:
        idx = np.asarray(idx, dtype=np.int64)
    ra = np.asarray(sample.ra[idx], dtype=np.float64)
    dec = np.asarray(sample.dec[idx], dtype=np.float64)
    weights = np.asarray(sample.weights[idx], dtype=np.float32)
    x = np.asarray(sample.x[idx], dtype=np.float64)
    y = np.asarray(sample.y[idx], dtype=np.float64)
    z = np.asarray(sample.z[idx], dtype=np.float64)
    region_id = None if sample.region_id is None else np.asarray(sample.region_id[idx], dtype=np.int32)

    mxh1 = int(sample.mxh1)
    mxh2 = int(sample.mxh2)
    grid_meta = dict(sample.grid_meta)
    if regrid and len(idx) > 0:
        mode = _resolve_autogrid_mode(grid_meta.get("autogrid_mode", True))
        dens = grid_meta.get("dens", None)
        if mode == "legacy":
            mxh1, mxh2, _, _info = best_skgrid_2d(len(ra), ra, mode="legacy", dens=dens)
        elif mode == "adaptive":
            theta_max = float(np.asarray(theta_edges, dtype=np.float64)[-1]) if theta_edges is not None and len(theta_edges) else float(grid_meta.get("theta_max", 0.0))
            coarse_bins = int(grid_meta.get("coarse_bins", 32))
            nthreads = int(grid_meta.get("nthreads", -1))
            mxh1, mxh2, _, _info = best_skgrid_2d(
                len(ra),
                ra,
                dec,
                dens=dens,
                mode="adaptive",
                sample_ras=ra,
                sample_decs=dec,
                coarse_bins=coarse_bins,
                nthreads=nthreads,
                count_sbound=sample.sbound,
                theta_max=theta_max,
                include_auto=True,
                include_cross=False,
            )
        sidx = _sort_index_2d_arrays(ra, dec, sbound=sample.sbound, mxh1=mxh1, mxh2=mxh2, pxorder=grid_meta.get("pxorder", "natural"))
        ra = np.asarray(ra[sidx], dtype=np.float64)
        dec = np.asarray(dec[sidx], dtype=np.float64)
        weights = np.asarray(weights[sidx], dtype=np.float32)
        x = np.asarray(x[sidx], dtype=np.float64)
        y = np.asarray(y[sidx], dtype=np.float64)
        z = np.asarray(z[sidx], dtype=np.float64)
        if region_id is not None:
            region_id = np.asarray(region_id[sidx], dtype=np.int32)
    sk, ll = cff.mod.skll2d(mxh1, mxh2, len(idx), ra, dec, sample.sbound)
    return PreparedAngularSample(
        table=None,
        ra=ra,
        dec=dec,
        weights=weights,
        x=x,
        y=y,
        z=z,
        sk=np.asarray(sk),
        ll=np.asarray(ll),
        wunit=bool(np.all(weights == 1.0)),
        sbound=sample.sbound,
        mxh1=mxh1,
        mxh2=mxh2,
        region_id=region_id,
        grid_meta=grid_meta,
        nrows=int(len(idx)),
    )


def _as_1d_float64(values) -> np.ndarray:
    """
    Convert the supplied value to 1d float64.
    
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
    arr = np.asarray(values, dtype=np.float64)
    return np.ravel(arr)


def _concatenate_float64(arrays: Iterable) -> np.ndarray:
    """
    Concatenate float64.
    
    Parameters
    ----------
    arrays : object
        Value for ``arrays``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    seq = [_as_1d_float64(a) for a in arrays]
    if not seq:
        return np.empty(0, dtype=np.float64)
    if len(seq) == 1:
        return seq[0]
    return np.concatenate(seq)


def _sample_width_from_ra(ras) -> tuple[float, float, float]:
    """
    Sample width from ra.
    
    Parameters
    ----------
    ras : object
        Value for ``ras``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ras = _as_1d_float64(ras)
    if ras.size == 0:
        return 0.0, 360.0, 360.0
    ramin = float(np.min(ras))
    ramax = float(np.max(ras))
    samplewidth = ramax - ramin
    if samplewidth <= 0:
        samplewidth = 360.0
    return ramin, ramax, float(samplewidth)


def best_skgrid_2d_legacy(npts, ras, *, dens=None):
    """
    Choose skgrid 2d legacy.
    
    Parameters
    ----------
    npts : object
        Value for ``npts``.
    ras : object
        Value for ``ras``.
    dens : object, optional
        Value for ``dens``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    if isinstance(npts, (list, tuple)):
        npts_eff = int(sum(int(v) for v in npts))
        ras_eff = _concatenate_float64(ras)
    else:
        npts_eff = int(npts)
        ras_eff = _as_1d_float64(ras)
    if dens is None:
        dens = 22.0 if npts_eff > 50000 else 16.0
    _, _, samplewidth = _sample_width_from_ra(ras_eff)
    h1h2 = npts_eff / dens
    h1 = max(int(np.rint(10.75 + 0.075 * np.sqrt(npts_eff))), 1)
    h2 = int(np.rint(h1h2 / h1) * (360.0 / samplewidth))
    h2 = max(h2, 1)
    return h1, h2, float(dens)


def _histogram_in_count_box(ras, decs, *, sbound, n_dec: int, n_ra: int) -> np.ndarray:
    """
    Histogram in count box.
    
    Parameters
    ----------
    ras : object
        Value for ``ras``.
    decs : object
        Value for ``decs``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    n_dec : object
        Value for ``n_dec``. This argument is keyword-only.
    n_ra : object
        Value for ``n_ra``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ras = _as_1d_float64(ras)
    decs = _as_1d_float64(decs)
    if ras.size == 0 or decs.size == 0:
        return np.zeros((max(int(n_dec), 1), max(int(n_ra), 1)), dtype=np.int64)

    ramin, ramax, decmin, decmax = [float(v) for v in sbound]
    ra_span = ramax - ramin
    dec_span = decmax - decmin
    if ra_span <= 0.0:
        ra_span = 360.0
    if dec_span <= 0.0:
        dec_span = 1.0

    n_dec = max(int(n_dec), 1)
    n_ra = max(int(n_ra), 1)

    ra_work = np.mod(ras - ramin, ra_span) + ramin
    ra_idx = np.floor((ra_work - ramin) / ra_span * n_ra).astype(np.int64)
    dec_idx = np.floor((decs - decmin) / dec_span * n_dec).astype(np.int64)
    np.clip(ra_idx, 0, n_ra - 1, out=ra_idx)
    np.clip(dec_idx, 0, n_dec - 1, out=dec_idx)

    hist = np.zeros((n_dec, n_ra), dtype=np.int64)
    np.add.at(hist, (dec_idx, ra_idx), 1)
    return hist



def _count_box_probe_stats(ras, decs, *, sbound, coarse_bins: int = 32):
    """
    Count box probe stats.
    
    Parameters
    ----------
    ras : object
        Value for ``ras``.
    decs : object
        Value for ``decs``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    coarse_bins : object, optional
        Value for ``coarse_bins``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ras = _as_1d_float64(ras)
    decs = _as_1d_float64(decs)
    ramin, ramax, decmin, decmax = [float(v) for v in sbound]
    ra_span = max(ramax - ramin, 360.0)
    dec_span = max(decmax - decmin, 1.0e-6)

    coarse_bins = max(int(coarse_bins), 8)
    n_dec = coarse_bins
    approx_cell_deg = max(dec_span / n_dec, 1.0e-6)
    n_ra = int(np.rint(ra_span / approx_cell_deg))
    n_ra = max(16, min(n_ra, max(8 * coarse_bins, 1024)))

    hist = _histogram_in_count_box(ras, decs, sbound=sbound, n_dec=n_dec, n_ra=n_ra)
    occ = hist > 0
    occupied = int(np.count_nonzero(occ))
    total = int(occ.size)
    row_counts = np.count_nonzero(occ, axis=1)
    active_rows = row_counts > 0
    fill_frac = occupied / total if total > 0 else 1.0
    active_dec_frac = float(np.count_nonzero(active_rows) / n_dec) if n_dec > 0 else 1.0
    mean_row_fill = float(np.mean(row_counts[active_rows] / n_ra)) if np.any(active_rows) else fill_frac

    return {
        "count_box_ra_span": float(ra_span),
        "count_box_dec_span": float(dec_span),
        "fill_frac": max(float(fill_frac), 1.0 / max(total, 1)),
        "active_dec_frac": max(active_dec_frac, 1.0 / n_dec),
        "mean_row_fill": max(mean_row_fill, 1.0 / n_ra),
        "n_ra": int(n_ra),
        "n_dec": int(n_dec),
        "occupied_cells": occupied,
        "total_cells": total,
    }



def _subsample_autogrid_arrays(ras, decs, *, max_points: int = 250_000) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Subsample autogrid arrays.
    
    Parameters
    ----------
    ras : object
        Value for ``ras``.
    decs : object
        Value for ``decs``.
    max_points : object, optional
        Value for ``max_points``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ras = _as_1d_float64(ras)
    decs = _as_1d_float64(decs)
    n = len(ras)
    if n == 0:
        return ras, decs, 1.0
    if n <= int(max_points):
        return ras, decs, 1.0
    idx = np.linspace(0, n - 1, int(max_points), dtype=np.int64)
    sample_n = len(idx)
    scale = float(n) / float(sample_n)
    return ras[idx], decs[idx], scale



def _local_mean_by_radius(values: np.ndarray, radius: int) -> np.ndarray:
    """
    Local mean by radius.
    
    Parameters
    ----------
    values : object
        Value for ``values``.
    radius : object
        Value for ``radius``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    r = max(int(radius), 0)
    if r == 0:
        return arr.copy()
    cs = np.concatenate(([0.0], np.cumsum(arr, dtype=np.float64)))
    idx = np.arange(arr.size, dtype=np.int64)
    lo = np.maximum(idx - r, 0)
    hi = np.minimum(idx + r + 1, arr.size)
    return (cs[hi] - cs[lo]) / (hi - lo)



def _row_summary_for_candidate(ras, decs, *, sbound, max_points: int = 250_000):
    """
    Row summary for candidate.
    
    Parameters
    ----------
    ras : object
        Value for ``ras``.
    decs : object
        Value for ``decs``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    max_points : object, optional
        Value for ``max_points``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ras_s, decs_s, scale = _subsample_autogrid_arrays(ras, decs, max_points=max_points)
    return {
        "ras": ras_s,
        "decs": decs_s,
        "scale": float(scale),
        "n_input": int(len(_as_1d_float64(ras))),
        "sbound": tuple(float(v) for v in sbound),
    }



def _candidate_grid_row_stats(summary: dict, *, h1: int, h2: int) -> dict:
    """
    Candidate grid row stats.
    
    Parameters
    ----------
    summary : object
        Value for ``summary``.
    h1 : object
        Value for ``h1``. This argument is keyword-only.
    h2 : object
        Value for ``h2``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ras = summary["ras"]
    decs = summary["decs"]
    sbound = summary["sbound"]
    qdec, qra = _grid_cell_indices(ras, decs, sbound=sbound, mxh1=h1, mxh2=h2)
    qdec = np.clip(qdec.astype(np.int64), 1, h1)
    qra = np.clip(qra.astype(np.int64), 1, h2)
    rows = qdec - 1
    row_counts = np.bincount(rows, minlength=h1).astype(np.float64) * summary["scale"]

    cell_ids = rows * h2 + (qra - 1)
    uniq, counts = np.unique(cell_ids, return_counts=True)
    occ_rows = (uniq // h2).astype(np.int64)
    occ_cols_row = np.bincount(occ_rows, minlength=h1).astype(np.float64)
    row_fill = occ_cols_row / float(max(h2, 1))
    objects_per_cell_all = row_counts / float(max(h2, 1))
    mean_occ_occupied = np.divide(
        row_counts,
        occ_cols_row,
        out=np.zeros_like(row_counts),
        where=occ_cols_row > 0,
    )
    p95_occ = float(np.percentile(counts.astype(np.float64) * summary["scale"], 95.0)) if counts.size else 0.0

    return {
        "row_counts": row_counts,
        "row_fill": row_fill,
        "objects_per_cell_all": objects_per_cell_all,
        "mean_occ_occupied": mean_occ_occupied,
        "occupied_cols_row": occ_cols_row,
        "p95_occ": p95_occ,
    }



def _neighbor_window_counts(*, sbound, h1: int, h2: int, theta_max: float) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Neighbor window counts.
    
    Parameters
    ----------
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    h1 : object
        Value for ``h1``. This argument is keyword-only.
    h2 : object
        Value for ``h2``. This argument is keyword-only.
    theta_max : object
        Value for ``theta_max``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ramin, ramax, decmin, decmax = [float(v) for v in sbound]
    ra_span = max(ramax - ramin, 1.0e-9)
    dec_span = max(decmax - decmin, 1.0e-9)
    dec_cell = dec_span / float(max(h1, 1))
    ra_cell = ra_span / float(max(h2, 1))
    k_dec = int(np.ceil(theta_max / max(dec_cell, 1.0e-9)))
    row_mid = decmin + (np.arange(h1, dtype=np.float64) + 0.5) * dec_cell
    cos_term = np.abs(np.cos(np.deg2rad(row_mid)))
    cos_floor = np.cos(np.deg2rad(85.0))
    cos_term = np.clip(cos_term, cos_floor, None)
    k_ra = np.ceil(theta_max / np.maximum(ra_cell * cos_term, 1.0e-9)).astype(np.int64)
    k_ra = np.clip(k_ra, 0, max(h2 // 2, 0))
    neighbor_cells = float(2 * k_dec + 1) * (2 * k_ra + 1).astype(np.float64)
    return k_dec, k_ra, neighbor_cells



def _score_auto_rows(summary: dict, *, sbound, h1: int, h2: int, theta_max: float, dens: float) -> tuple[float, dict]:
    """
    Score auto rows.
    
    Parameters
    ----------
    summary : object
        Value for ``summary``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    h1 : object
        Value for ``h1``. This argument is keyword-only.
    h2 : object
        Value for ``h2``. This argument is keyword-only.
    theta_max : object
        Value for ``theta_max``. This argument is keyword-only.
    dens : object
        Value for ``dens``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    row_stats = _candidate_grid_row_stats(summary, h1=h1, h2=h2)
    row_counts = row_stats["row_counts"]
    k_dec, k_ra, neighbor_cells = _neighbor_window_counts(sbound=sbound, h1=h1, h2=h2, theta_max=theta_max)
    local_fill = _local_mean_by_radius(row_stats["row_fill"], k_dec)
    local_occ = _local_mean_by_radius(row_stats["mean_occ_occupied"], k_dec)
    expected_points = neighbor_cells * local_fill * local_occ
    score = float(np.sum(row_counts * (neighbor_cells + expected_points)))
    occ_penalty = 1.0 + 0.05 * max(row_stats["p95_occ"] / max(float(dens), 1.0) - 1.0, 0.0)
    return score * occ_penalty, {**row_stats, "k_dec": int(k_dec), "mean_k_ra": float(np.mean(k_ra) if k_ra.size else 0.0)}



def _score_cross_rows(left_summary: dict, right_summary: dict, *, sbound, h1: int, h2: int, theta_max: float, dens: float) -> tuple[float, dict]:
    """
    Score cross rows.
    
    Parameters
    ----------
    left_summary : object
        Value for ``left_summary``.
    right_summary : object
        Value for ``right_summary``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    h1 : object
        Value for ``h1``. This argument is keyword-only.
    h2 : object
        Value for ``h2``. This argument is keyword-only.
    theta_max : object
        Value for ``theta_max``. This argument is keyword-only.
    dens : object
        Value for ``dens``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    left_rows = _candidate_grid_row_stats(left_summary, h1=h1, h2=h2)
    right_rows = _candidate_grid_row_stats(right_summary, h1=h1, h2=h2)
    row_counts_left = left_rows["row_counts"]
    k_dec, k_ra, neighbor_cells = _neighbor_window_counts(sbound=sbound, h1=h1, h2=h2, theta_max=theta_max)
    local_fill_right = _local_mean_by_radius(right_rows["row_fill"], k_dec)
    local_occ_right = _local_mean_by_radius(right_rows["mean_occ_occupied"], k_dec)
    expected_points = neighbor_cells * local_fill_right * local_occ_right
    score = float(np.sum(row_counts_left * (neighbor_cells + expected_points)))
    occ_penalty = 1.0 + 0.05 * max(right_rows["p95_occ"] / max(float(dens), 1.0) - 1.0, 0.0)
    diag = {
        "left": left_rows,
        "right": right_rows,
        "k_dec": int(k_dec),
        "mean_k_ra": float(np.mean(k_ra) if k_ra.size else 0.0),
    }
    return score * occ_penalty, diag



def _candidate_pairs_for_runtime_search(
    legacy_h1: int,
    legacy_h2: int,
    est_h1: int,
    est_h2: int,
    *,
    target_total_cells: float,
    nthreads: int = -1,
) -> list[tuple[int, int]]:
    """
    Candidate pairs for runtime search.
    
    Parameters
    ----------
    legacy_h1 : object
        Value for ``legacy_h1``.
    legacy_h2 : object
        Value for ``legacy_h2``.
    est_h1 : object
        Value for ``est_h1``.
    est_h2 : object
        Value for ``est_h2``.
    target_total_cells : object
        Value for ``target_total_cells``. This argument is keyword-only.
    nthreads : object, optional
        Value for ``nthreads``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    thread_floor = 1
    if nthreads is not None and int(nthreads) > 0:
        thread_floor = max(1, min(int(nthreads), 64))

    legacy_total = max(float(legacy_h1) * float(legacy_h2), 1.0)
    est_total = max(float(target_total_cells), 1.0)
    mixed_total = np.sqrt(legacy_total * est_total)

    est_h1_clamped = int(np.clip(est_h1, max(1, legacy_h1 // 3), max(legacy_h1 * 3, 3)))
    h1_bases = [legacy_h1, est_h1_clamped, int(np.rint(np.sqrt(max(legacy_h1 * est_h1_clamped, 1))))]
    total_bases = [legacy_total, mixed_total]
    h1_scales = (0.75, 1.0, 1.33)
    h2_scales = (0.9, 1.0, 1.1)

    candidates: set[tuple[int, int]] = set()
    for hb in h1_bases:
        for s1 in h1_scales:
            h1 = max(int(np.rint(hb * s1)), thread_floor)
            for total in total_bases:
                h2_base = max(int(np.rint(total / max(h1, 1))), 1)
                for s2 in h2_scales:
                    h2 = max(int(np.rint(h2_base * s2)), 1)
                    candidates.add((h1, h2))

    candidates.add((max(int(legacy_h1), thread_floor), max(int(legacy_h2), 1)))
    candidates.add((max(int(est_h1_clamped), thread_floor), max(int(est_h2), 1)))

    filtered = [
        (h1, h2)
        for h1, h2 in candidates
        if h1 > 0 and h2 > 0 and h1 <= 4096 and h2 <= 250000
    ]
    return sorted(filtered)



def best_skgrid_2d_adaptive(
    npts,
    ras,
    decs,
    *,
    dens=None,
    sample_ras=None,
    sample_decs=None,
    coarse_bins: int = 32,
    nthreads: int = -1,
    count_sbound=None,
    theta_max: float | None = None,
    left_ras=None,
    left_decs=None,
    include_auto: bool = True,
    include_cross: bool = False,
):
    """
    Choose skgrid 2d adaptive.
    
    Parameters
    ----------
    npts : object
        Value for ``npts``.
    ras : object
        Value for ``ras``.
    decs : object
        Value for ``decs``.
    dens : object, optional
        Value for ``dens``. This argument is keyword-only.
    sample_ras : object, optional
        Value for ``sample_ras``. This argument is keyword-only.
    sample_decs : object, optional
        Value for ``sample_decs``. This argument is keyword-only.
    coarse_bins : object, optional
        Value for ``coarse_bins``. This argument is keyword-only.
    nthreads : object, optional
        Value for ``nthreads``. This argument is keyword-only.
    count_sbound : object, optional
        Value for ``count_sbound``. This argument is keyword-only.
    theta_max : object, optional
        Value for ``theta_max``. This argument is keyword-only.
    left_ras : object, optional
        Value for ``left_ras``. This argument is keyword-only.
    left_decs : object, optional
        Value for ``left_decs``. This argument is keyword-only.
    include_auto : object, optional
        Value for ``include_auto``. This argument is keyword-only.
    include_cross : object, optional
        Value for ``include_cross``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    npts_eff = int(npts)
    ras_eff = _as_1d_float64(ras)
    decs_eff = _as_1d_float64(decs)
    if sample_ras is None or sample_decs is None:
        fp_ras = ras_eff
        fp_decs = decs_eff
    else:
        fp_ras = _as_1d_float64(sample_ras)
        fp_decs = _as_1d_float64(sample_decs)

    if count_sbound is None:
        count_sbound = bound2d([decs_eff])
    count_sbound = tuple(float(v) for v in count_sbound)
    theta_max = float(theta_max or 0.0)

    legacy_h1, legacy_h2, dens = best_skgrid_2d_legacy(npts_eff, fp_ras, dens=dens)
    stats = _count_box_probe_stats(fp_ras, fp_decs, sbound=count_sbound, coarse_bins=coarse_bins)

    target_occupied_cells = max(npts_eff / dens, 1.0)
    target_total_cells = target_occupied_cells / stats["fill_frac"]
    effective_height = max(stats["count_box_dec_span"] * stats["active_dec_frac"], 1.0e-6)
    effective_width = max(stats["count_box_ra_span"] * stats["mean_row_fill"], 1.0e-6)
    aspect = max(effective_width / effective_height, 1.0e-6)
    est_h1 = max(int(np.rint(np.sqrt(target_total_cells / aspect))), 1)
    est_h2 = max(int(np.rint(target_total_cells / max(est_h1, 1))), 1)
    if nthreads is not None and int(nthreads) > 0:
        est_h1 = max(est_h1, min(int(nthreads), 64))

    indexed_summary = _row_summary_for_candidate(ras_eff, decs_eff, sbound=count_sbound)
    left_summary = None
    if include_cross and left_ras is not None and left_decs is not None:
        left_summary = _row_summary_for_candidate(left_ras, left_decs, sbound=count_sbound)

    best_score = None
    best_pair = (legacy_h1, legacy_h2)
    candidates = _candidate_pairs_for_runtime_search(
        legacy_h1,
        legacy_h2,
        est_h1,
        est_h2,
        target_total_cells=target_total_cells,
        nthreads=nthreads,
    )

    for h1, h2 in candidates:
        score = 0.0
        if include_auto:
            auto_score, _ = _score_auto_rows(
                indexed_summary,
                sbound=count_sbound,
                h1=h1,
                h2=h2,
                theta_max=theta_max,
                dens=dens,
            )
            score += auto_score
        if include_cross and left_summary is not None:
            cross_score, _ = _score_cross_rows(
                left_summary,
                indexed_summary,
                sbound=count_sbound,
                h1=h1,
                h2=h2,
                theta_max=theta_max,
                dens=dens,
            )
            score += cross_score
        if best_score is None or score < best_score:
            best_score = score
            best_pair = (h1, h2)

    h1, h2 = best_pair
    return h1, h2, float(dens), {
        "legacy_h1": int(legacy_h1),
        "legacy_h2": int(legacy_h2),
        "est_h1": int(est_h1),
        "est_h2": int(est_h2),
        "target_total_cells": float(target_total_cells),
        "candidate_count": int(len(candidates)),
        "best_score": float(best_score if best_score is not None else 0.0),
        "theta_max": float(theta_max),
        "count_sbound": tuple(float(v) for v in count_sbound),
        **stats,
    }



def best_skgrid_2d(
    npts,
    ras,
    decs=None,
    *,
    dens=None,
    mode="legacy",
    sample_ras=None,
    sample_decs=None,
    coarse_bins: int = 32,
    nthreads: int = -1,
    count_sbound=None,
    theta_max: float | None = None,
    left_ras=None,
    left_decs=None,
    include_auto: bool = True,
    include_cross: bool = False,
):
    """
    Choose skgrid 2d.
    
    Parameters
    ----------
    npts : object
        Value for ``npts``.
    ras : object
        Value for ``ras``.
    decs : object, optional
        Value for ``decs``.
    dens : object, optional
        Value for ``dens``. This argument is keyword-only.
    mode : object, optional
        Value for ``mode``. This argument is keyword-only.
    sample_ras : object, optional
        Value for ``sample_ras``. This argument is keyword-only.
    sample_decs : object, optional
        Value for ``sample_decs``. This argument is keyword-only.
    coarse_bins : object, optional
        Value for ``coarse_bins``. This argument is keyword-only.
    nthreads : object, optional
        Value for ``nthreads``. This argument is keyword-only.
    count_sbound : object, optional
        Value for ``count_sbound``. This argument is keyword-only.
    theta_max : object, optional
        Value for ``theta_max``. This argument is keyword-only.
    left_ras : object, optional
        Value for ``left_ras``. This argument is keyword-only.
    left_decs : object, optional
        Value for ``left_decs``. This argument is keyword-only.
    include_auto : object, optional
        Value for ``include_auto``. This argument is keyword-only.
    include_cross : object, optional
        Value for ``include_cross``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    mode = _resolve_autogrid_mode(mode)
    if mode == "manual":
        raise ValueError("best_skgrid_2d() does not support manual mode; use the user-supplied mxh1/mxh2 directly.")
    if mode == "legacy":
        h1, h2, dens = best_skgrid_2d_legacy(npts, ras, dens=dens)
        return h1, h2, dens, {"mode": "legacy"}
    if decs is None:
        raise ValueError("Adaptive autogrid requires declinations.")
    h1, h2, dens, info = best_skgrid_2d_adaptive(
        npts,
        ras,
        decs,
        dens=dens,
        sample_ras=sample_ras,
        sample_decs=sample_decs,
        coarse_bins=coarse_bins,
        nthreads=nthreads,
        count_sbound=count_sbound,
        theta_max=theta_max,
        left_ras=left_ras,
        left_decs=left_decs,
        include_auto=include_auto,
        include_cross=include_cross,
    )
    info["mode"] = "adaptive"
    return h1, h2, dens, info


def _grid_cell_indices(ra: np.ndarray, dec: np.ndarray, *, sbound, mxh1: int, mxh2: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Grid cell indices.
    
    Parameters
    ----------
    ra : object
        Value for ``ra``.
    dec : object
        Value for ``dec``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    mxh1 : object
        Value for ``mxh1``. This argument is keyword-only.
    mxh2 : object
        Value for ``mxh2``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ramin, ramax, decmin, decmax = [float(v) for v in sbound]
    dec_span = decmax - decmin
    ra_span = ramax - ramin

    if mxh1 <= 0 or dec_span <= 0.0:
        qdec = np.ones(len(dec), dtype=np.int64)
    else:
        hc1 = dec_span / float(mxh1)
        qdec = np.floor((dec - decmin) / hc1).astype(np.int64) + 1

    if mxh2 <= 0 or ra_span <= 0.0:
        qra = np.ones(len(ra), dtype=np.int64)
    else:
        hc2 = ra_span / float(mxh2)
        qra = np.floor((ra - ramin) / hc2).astype(np.int64) + 1
        hi = qra > mxh2
        lo = qra < 1
        if np.any(hi):
            qra[hi] -= mxh2
        if np.any(lo):
            qra[lo] += mxh2

    return qdec, qra


def pixsort(table, ra_col: str, dec_col: str, *, sbound, mxh1: int, mxh2: int, pxorder: str = "natural"):
    """
    Sort a catalog into the counter-friendly cell order used by the linked-list builder.
    
    Parameters
    ----------
    table : object
        Value for ``table``.
    ra_col : object
        Value for ``ra_col``.
    dec_col : object
        Value for ``dec_col``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    mxh1 : object
        Value for ``mxh1``. This argument is keyword-only.
    mxh2 : object
        Value for ``mxh2``. This argument is keyword-only.
    pxorder : object, optional
        Value for ``pxorder``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    ra = _col(table, ra_col, dtype=np.float64)
    dec = _col(table, dec_col, dtype=np.float64)
    pxmode = _normalize_pxorder(pxorder)

    if pxmode == "none":
        return np.arange(catalog_nrows(table), dtype=np.int64)

    qdec, qra = _grid_cell_indices(ra, dec, sbound=sbound, mxh1=mxh1, mxh2=mxh2)
    if pxmode == "natural":
        return np.lexsort((qra, qdec))
    if pxmode == "cell-dec":
        return np.lexsort((ra, dec, qra, qdec))
    raise NameError(
        f"Order {pxorder!r} not implemented. Use 'none', 'natural', or 'cell-dec'."
    )


def _ensure_weights(table, colname: str):
    """Return the weight column when present, otherwise unit weights."""
    if catalog_has_column(table, colname):
        return np.asarray(_col(table, colname), dtype=np.float32)
    return np.ones(catalog_nrows(table), dtype=np.float32)


def _prepare_sample(table, *, ra_col, dec_col, wei_col, sbound, mxh1, mxh2, pxorder, region_id=None, grid_meta=None):
    """
    Prepare sample.
    
    Parameters
    ----------
    table : object
        Value for ``table``.
    ra_col : object
        Value for ``ra_col``. This argument is keyword-only.
    dec_col : object
        Value for ``dec_col``. This argument is keyword-only.
    wei_col : object
        Value for ``wei_col``. This argument is keyword-only.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    mxh1 : object
        Value for ``mxh1``. This argument is keyword-only.
    mxh2 : object
        Value for ``mxh2``. This argument is keyword-only.
    pxorder : object
        Value for ``pxorder``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ra = _col(table, ra_col, dtype=np.float64)
    dec = _col(table, dec_col, dtype=np.float64)
    weights = _ensure_weights(table, wei_col)
    sidx = _sort_index_2d_arrays(ra, dec, sbound=sbound, mxh1=mxh1, mxh2=mxh2, pxorder=pxorder)
    ra = np.asarray(ra[sidx], dtype=np.float64)
    dec = np.asarray(dec[sidx], dtype=np.float64)
    weights = np.asarray(weights[sidx], dtype=np.float32)
    if region_id is not None:
        region_id = np.asarray(region_id, dtype=np.int32)[sidx]
    x, y, z = radec2xyz(np.deg2rad(ra), np.deg2rad(dec))
    sk, ll = cff.mod.skll2d(mxh1, mxh2, len(ra), ra, dec, sbound)
    return PreparedAngularSample(
        table=None,
        ra=ra,
        dec=dec,
        weights=weights,
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        z=np.asarray(z, dtype=np.float64),
        sk=np.asarray(sk),
        ll=np.asarray(ll),
        wunit=bool(np.all(weights == 1.0)),
        sbound=sbound,
        mxh1=mxh1,
        mxh2=mxh2,
        region_id=None if region_id is None else np.asarray(region_id, dtype=np.int32),
        grid_meta={} if grid_meta is None else dict(grid_meta),
        nrows=int(len(ra)),
    )


def _autogrid_auto_pair(data, random, config: AngularAutoConfig):
    """
    Autogrid auto pair.
    
    Parameters
    ----------
    data : object
        Value for ``data``.
    random : object
        Value for ``random``.
    config : object
        Value for ``config``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    mode = _resolve_autogrid_mode(config.grid.autogrid)
    if mode == "manual":
        return (
            config.grid.mxh1,
            config.grid.mxh2,
            {"mode": "manual", "sample": "data"},
        ), (
            config.grid.mxh1,
            config.grid.mxh2,
            {"mode": "manual", "sample": "random"},
        )

    data_ra = _col(data, config.columns_data.ra, dtype=np.float64)
    data_dec = _col(data, config.columns_data.dec, dtype=np.float64)
    rand_ra = _col(random, config.columns_random.ra, dtype=np.float64)
    rand_dec = _col(random, config.columns_random.dec, dtype=np.float64)
    count_sbound = bound2d([data_dec, rand_dec])
    theta_edges, _ = makebins(config.binning.nsep, config.binning.sepmin, config.binning.dsep, config.binning.logsep)
    theta_max = float(theta_edges[-1]) if len(theta_edges) else 0.0

    if mode == "legacy":
        dd_h1, dd_h2, dens, dd_info = best_skgrid_2d(
            len(data),
            data_ra,
            mode="legacy",
            dens=config.grid.dens,
        )
        rr_h1, rr_h2, _, rr_info = best_skgrid_2d(
            len(random),
            rand_ra,
            mode="legacy",
            dens=dens,
        )
        dd_info.update({"sample": "data", "dens": dens})
        rr_info.update({"sample": "random", "dens": dens})
        return (dd_h1, dd_h2, dd_info), (rr_h1, rr_h2, rr_info)

    dd_h1, dd_h2, dens, dd_info = best_skgrid_2d(
        len(data),
        data_ra,
        data_dec,
        dens=config.grid.dens,
        mode="adaptive",
        sample_ras=data_ra,
        sample_decs=data_dec,
        coarse_bins=config.grid.coarse_bins,
        nthreads=config.nthreads,
        count_sbound=count_sbound,
        theta_max=theta_max,
        include_auto=True,
        include_cross=False,
    )
    rr_h1, rr_h2, _, rr_info = best_skgrid_2d(
        len(random),
        rand_ra,
        rand_dec,
        dens=dens,
        mode="adaptive",
        sample_ras=rand_ra,
        sample_decs=rand_dec,
        coarse_bins=config.grid.coarse_bins,
        nthreads=config.nthreads,
        count_sbound=count_sbound,
        theta_max=theta_max,
        left_ras=data_ra,
        left_decs=data_dec,
        include_auto=True,
        include_cross=True,
    )
    dd_info.update({"sample": "data", "grid_probe_source": "data", "dens": dens, "objective": "DD"})
    rr_info.update({"sample": "random", "grid_probe_source": "random", "dens": dens, "objective": "RR+DR"})
    return (dd_h1, dd_h2, dd_info), (rr_h1, rr_h2, rr_info)



def _autogrid_cross_common(data1, random1, data2, random2, config: AngularCrossConfig):
    """
    Autogrid cross common.
    
    Parameters
    ----------
    data1 : object
        Value for ``data1``.
    random1 : object
        Value for ``random1``.
    data2 : object
        Value for ``data2``.
    random2 : object
        Value for ``random2``.
    config : object
        Value for ``config``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    mode = _resolve_autogrid_mode(config.grid.autogrid)
    if mode == "manual":
        return config.grid.mxh1, config.grid.mxh2, {"mode": "manual", "sample": "shared"}

    tables = [data1, data2]
    ra_cols = [config.columns_data1.ra, config.columns_data2.ra]
    dec_cols = [config.columns_data1.dec, config.columns_data2.dec]
    if random1 is not None:
        tables.append(random1)
        ra_cols.append(config.columns_random1.ra)
        dec_cols.append(config.columns_random1.dec)
    if random1 is not None and random2 is not None:
        tables.append(random2)
        ra_cols.append(config.columns_random2.ra)
        dec_cols.append(config.columns_random2.dec)
    sizes = [catalog_nrows(t) for t in tables]

    if mode == "legacy":
        order = np.argsort(sizes)[::-1][:2]
        ras = [_col(tables[i], ra_cols[i], dtype=np.float64) for i in order]
        h1, h2, dens, info = best_skgrid_2d(
            [sizes[i] for i in order],
            ras,
            dens=config.grid.dens,
            mode="legacy",
        )
        info.update({"sample": "shared", "dens": dens})
        return h1, h2, info

    all_decs = [_col(tables[i], dec_cols[i], dtype=np.float64) for i in range(len(tables))]
    count_sbound = bound2d(all_decs)
    theta_edges, _ = makebins(config.binning.nsep, config.binning.sepmin, config.binning.dsep, config.binning.logsep)
    theta_max = float(theta_edges[-1]) if len(theta_edges) else 0.0

    if random2 is not None:
        fp_ras = _concatenate_float64([
            _col(random1, config.columns_random1.ra, dtype=np.float64),
            _col(random2, config.columns_random2.ra, dtype=np.float64),
        ])
        fp_decs = _concatenate_float64([
            _col(random1, config.columns_random1.dec, dtype=np.float64),
            _col(random2, config.columns_random2.dec, dtype=np.float64),
        ])
        left_ras = _concatenate_float64([
            _col(data1, config.columns_data1.ra, dtype=np.float64),
            _col(random1, config.columns_random1.ra, dtype=np.float64),
        ])
        left_decs = _concatenate_float64([
            _col(data1, config.columns_data1.dec, dtype=np.float64),
            _col(random1, config.columns_random1.dec, dtype=np.float64),
        ])
        right_ras = _concatenate_float64([
            _col(data2, config.columns_data2.ra, dtype=np.float64),
            _col(random2, config.columns_random2.ra, dtype=np.float64),
        ])
        right_decs = _concatenate_float64([
            _col(data2, config.columns_data2.dec, dtype=np.float64),
            _col(random2, config.columns_random2.dec, dtype=np.float64),
        ])
        grid_probe_source = "random1+random2"
    elif random1 is not None:
        fp_ras = _concatenate_float64([
            _col(random1, config.columns_random1.ra, dtype=np.float64),
            _col(data2, config.columns_data2.ra, dtype=np.float64),
        ])
        fp_decs = _concatenate_float64([
            _col(random1, config.columns_random1.dec, dtype=np.float64),
            _col(data2, config.columns_data2.dec, dtype=np.float64),
        ])
        left_ras = _as_1d_float64(_col(data1, config.columns_data1.ra, dtype=np.float64))
        left_decs = _as_1d_float64(_col(data1, config.columns_data1.dec, dtype=np.float64))
        right_ras = _concatenate_float64([
            _col(data2, config.columns_data2.ra, dtype=np.float64),
            _col(random1, config.columns_random1.ra, dtype=np.float64),
        ])
        right_decs = _concatenate_float64([
            _col(data2, config.columns_data2.dec, dtype=np.float64),
            _col(random1, config.columns_random1.dec, dtype=np.float64),
        ])
        grid_probe_source = "random1+data2"
    elif random2 is not None:
        fp_ras = _concatenate_float64([
            _col(data2, config.columns_data2.ra, dtype=np.float64),
            _col(random2, config.columns_random2.ra, dtype=np.float64),
        ])
        fp_decs = _concatenate_float64([
            _col(data2, config.columns_data2.dec, dtype=np.float64),
            _col(random2, config.columns_random2.dec, dtype=np.float64),
        ])
        left_ras = _as_1d_float64(_col(data1, config.columns_data1.ra, dtype=np.float64))
        left_decs = _as_1d_float64(_col(data1, config.columns_data1.dec, dtype=np.float64))
        right_ras = _concatenate_float64([
            _col(data2, config.columns_data2.ra, dtype=np.float64),
            _col(random2, config.columns_random2.ra, dtype=np.float64),
        ])
        right_decs = _concatenate_float64([
            _col(data2, config.columns_data2.dec, dtype=np.float64),
            _col(random2, config.columns_random2.dec, dtype=np.float64),
        ])
        grid_probe_source = "data2+random2"
    else:
        fp_ras = _concatenate_float64([_col(data1, config.columns_data1.ra, dtype=np.float64), _col(data2, config.columns_data2.ra, dtype=np.float64)])
        fp_decs = _concatenate_float64([_col(data1, config.columns_data1.dec, dtype=np.float64), _col(data2, config.columns_data2.dec, dtype=np.float64)])
        left_ras = _as_1d_float64(_col(data1, config.columns_data1.ra, dtype=np.float64))
        left_decs = _as_1d_float64(_col(data1, config.columns_data1.dec, dtype=np.float64))
        right_ras = _as_1d_float64(_col(data2, config.columns_data2.ra, dtype=np.float64))
        right_decs = _as_1d_float64(_col(data2, config.columns_data2.dec, dtype=np.float64))
        grid_probe_source = "data1+data2"

    h1, h2, dens, info = best_skgrid_2d(
        len(right_ras),
        right_ras,
        right_decs,
        dens=config.grid.dens,
        mode="adaptive",
        sample_ras=fp_ras,
        sample_decs=fp_decs,
        coarse_bins=config.grid.coarse_bins,
        nthreads=config.nthreads,
        count_sbound=count_sbound,
        theta_max=theta_max,
        left_ras=left_ras,
        left_decs=left_decs,
        include_auto=False,
        include_cross=True,
    )
    info.update({"sample": "shared", "dens": dens, "grid_probe_source": grid_probe_source, "objective": "cross"})
    return h1, h2, info


def prepare_angular_auto(data, random, config: AngularAutoConfig):
    """
    Prepare angular auto.
    
    Parameters
    ----------
    data : object
        Value for ``data``.
    random : object
        Value for ``random``.
    config : object
        Value for ``config``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    sep_edges, (_, sep_centers, sep_delta) = makebins(
        config.binning.nsep,
        config.binning.sepmin,
        config.binning.dsep,
        config.binning.logsep,
    )
    sbound = bound2d([
        _col(data, config.columns_data.dec, dtype=np.float64),
        _col(random, config.columns_random.dec, dtype=np.float64),
    ])
    (dd_h1, dd_h2, dd_grid_info), (rr_h1, rr_h2, rr_grid_info) = _autogrid_auto_pair(data, random, config)
    data_region = random_region = None
    jk_meta = {"jk_enabled": bool(config.jackknife.enabled), "jk_region_source": None, "jk_nregions": None, "jk_geometry_from": None}
    if config.jackknife.enabled:
        dreg = config.columns_data.region
        rreg = config.columns_random.region
        if dreg is not None and rreg is not None:
            data_region, random_region = _shared_user_region_ids([(data, dreg), (random, rreg)])
            jk_meta.update({"jk_region_source": "user", "jk_nregions": int(np.max(np.concatenate([data_region, random_region])) + 1)})
        elif dreg is not None or rreg is not None:
            raise ValueError("For jackknife auto-correlations, either both data/random region columns must be supplied or neither.")
        else:
            nregions = config.jackknife.nregions
            if nregions is None:
                nregions = choose_default_nregions(config.binning.nsep)
            assignments, _centers = _auto_region_ids([
                (data, config.columns_data.ra, config.columns_data.dec, "data"),
                (random, config.columns_random.ra, config.columns_random.dec, "random"),
            ], nregions=min(int(nregions), max(1, catalog_nrows(data), catalog_nrows(random))), seed=config.jackknife.seed, geometry_from=config.jackknife.geometry_from)
            data_region, random_region = assignments
            jk_meta.update({"jk_region_source": "auto", "jk_nregions": int(np.max(np.concatenate([data_region, random_region])) + 1), "jk_geometry_from": config.jackknife.geometry_from})
    grid_mode = _resolve_autogrid_mode(config.grid.autogrid)
    data_prepared = _prepare_sample(
        data,
        ra_col=config.columns_data.ra,
        dec_col=config.columns_data.dec,
        wei_col=config.columns_data.weight,
        sbound=sbound,
        mxh1=dd_h1,
        mxh2=dd_h2,
        pxorder=config.grid.pxorder,
        region_id=data_region,
        grid_meta={**dd_grid_info, "autogrid_mode": grid_mode, "pxorder": config.grid.pxorder, "coarse_bins": config.grid.coarse_bins, "nthreads": config.nthreads, "theta_max": float(sep_edges[-1]) if len(sep_edges) else 0.0},
    )
    random_prepared = _prepare_sample(
        random,
        ra_col=config.columns_random.ra,
        dec_col=config.columns_random.dec,
        wei_col=config.columns_random.weight,
        sbound=sbound,
        mxh1=rr_h1,
        mxh2=rr_h2,
        pxorder=config.grid.pxorder,
        region_id=random_region,
        grid_meta={**rr_grid_info, "autogrid_mode": grid_mode, "pxorder": config.grid.pxorder, "coarse_bins": config.grid.coarse_bins, "nthreads": config.nthreads, "theta_max": float(sep_edges[-1]) if len(sep_edges) else 0.0},
    )
    grid_data = dict(dd_grid_info)
    grid_random = dict(rr_grid_info)
    grid_data.setdefault("sbound", tuple(float(v) for v in sbound))
    grid_random.setdefault("sbound", tuple(float(v) for v in sbound))
    meta = {
        "theta_edges": np.asarray(sep_edges, dtype=np.float64),
        "theta_centers": np.asarray(sep_centers, dtype=np.float64),
        "theta_delta": np.asarray(sep_delta, dtype=np.float64),
        "sbound": tuple(float(v) for v in sbound),
        "grid_data": grid_data,
        "grid_random": grid_random,
    }
    meta.update(jk_meta)
    return data_prepared, random_prepared, meta


# def prepare_angular_auto(data, random, config: AngularAutoConfig):
#     sep_edges, (_, sep_centers, sep_delta) = makebins(config.binning.nsep, config.binning.sepmin, config.binning.dsep, config.binning.logsep)
#     sbound_data = bound2d([data[config.columns_data.dec].data], [data[config.columns_data.ra].data])
#     sbound_random = bound2d([random[config.columns_random.dec].data], [random[config.columns_random.ra].data])
#     (dd_h1, dd_h2, dd_grid_info), (rr_h1, rr_h2, rr_grid_info) = _autogrid_auto_pair(data, random, config)
#     data_prepared = _prepare_sample(data, ra_col=config.columns_data.ra, dec_col=config.columns_data.dec, wei_col=config.columns_data.weight, sbound=sbound_data, mxh1=dd_h1, mxh2=dd_h2, pxorder=config.grid.pxorder)
#     random_prepared = _prepare_sample(random, ra_col=config.columns_random.ra, dec_col=config.columns_random.dec, wei_col=config.columns_random.weight, sbound=sbound_random, mxh1=rr_h1, mxh2=rr_h2, pxorder=config.grid.pxorder)
#     metadata = {
#         "theta_edges": np.asarray(sep_edges, dtype=np.float64),
#         "theta_centers": np.asarray(sep_centers, dtype=np.float64),
#         "theta_delta": np.asarray(sep_delta, dtype=np.float64),
#         "cross0": cross0guess(np.asarray(data[config.columns_data.ra].data)),
#         "grid_data": {"mxh1": dd_h1, "mxh2": dd_h2, "sbound": tuple(float(v) for v in sbound_data), **dd_grid_info},
#         "grid_random": {"mxh1": rr_h1, "mxh2": rr_h2, "sbound": tuple(float(v) for v in sbound_random), **rr_grid_info},
#     }
#     return data_prepared, random_prepared, metadata


# def prepare_angular_auto(data, random, config: AngularAutoConfig):
#     sep_edges, (_, sep_centers, sep_delta) = makebins(config.binning.nsep, config.binning.sepmin, config.binning.dsep, config.binning.logsep)
#     sbound = bound2d([data[config.columns_data.dec].data, random[config.columns_random.dec].data], [data[config.columns_data.ra].data, random[config.columns_random.ra].data])
#     (dd_h1, dd_h2, dd_grid_info), (rr_h1, rr_h2, rr_grid_info) = _autogrid_auto_pair(data, random, config)
#     data_prepared = _prepare_sample(data, ra_col=config.columns_data.ra, dec_col=config.columns_data.dec, wei_col=config.columns_data.weight, sbound=sbound, mxh1=dd_h1, mxh2=dd_h2, pxorder=config.grid.pxorder)
#     random_prepared = _prepare_sample(random, ra_col=config.columns_random.ra, dec_col=config.columns_random.dec, wei_col=config.columns_random.weight, sbound=sbound, mxh1=rr_h1, mxh2=rr_h2, pxorder=config.grid.pxorder)
#     metadata = {
#         "theta_edges": np.asarray(sep_edges, dtype=np.float64),
#         "theta_centers": np.asarray(sep_centers, dtype=np.float64),
#         "theta_delta": np.asarray(sep_delta, dtype=np.float64),
#         "cross0": cross0guess(np.asarray(data[config.columns_data.ra].data)),
#         "grid_data": {"mxh1": dd_h1, "mxh2": dd_h2, **dd_grid_info},
#         "grid_random": {"mxh1": rr_h1, "mxh2": rr_h2, **rr_grid_info},
#     }
#     return data_prepared, random_prepared, metadata


def prepare_angular_cross(data1, random1, data2, random2, config: AngularCrossConfig):
    """
    Prepare angular cross.
    
    Parameters
    ----------
    data1 : object
        Value for ``data1``.
    random1 : object
        Value for ``random1``.
    data2 : object
        Value for ``data2``.
    random2 : object
        Value for ``random2``.
    config : object
        Value for ``config``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    sep_edges, (_, sep_centers, sep_delta) = makebins(
        config.binning.nsep,
        config.binning.sepmin,
        config.binning.dsep,
        config.binning.logsep,
    )
    dec_arrays = [_col(data1, config.columns_data1.dec, dtype=np.float64), _col(data2, config.columns_data2.dec, dtype=np.float64)]
    if random1 is not None:
        dec_arrays.append(_col(random1, config.columns_random1.dec, dtype=np.float64))
    if random2 is not None:
        dec_arrays.append(_col(random2, config.columns_random2.dec, dtype=np.float64))
    sbound = bound2d(dec_arrays)
    h1, h2, grid_info = _autogrid_cross_common(data1, random1, data2, random2, config)
    jk_meta = {"jk_enabled": bool(config.jackknife.enabled), "jk_region_source": None, "jk_nregions": None, "jk_geometry_from": None}
    reg1 = reg_r1 = reg2 = reg_r2 = None
    if config.jackknife.enabled:
        pairs = [(data1, config.columns_data1.region), (data2, config.columns_data2.region)]
        if random1 is not None:
            pairs.append((random1, config.columns_random1.region))
        if random2 is not None:
            pairs.append((random2, config.columns_random2.region))
        provided = [col is not None for _tab, col in pairs]
        if all(provided):
            regions = _shared_user_region_ids(pairs)
            idx = 0
            reg1, reg2 = regions[0], regions[1]
            idx = 2
            if random1 is not None:
                reg_r1 = regions[idx]; idx += 1
            if random2 is not None:
                reg_r2 = regions[idx]
            all_regs = np.concatenate(regions) if regions else np.empty(0, dtype=np.int32)
            jk_meta.update({"jk_region_source": "user", "jk_nregions": int(np.max(all_regs) + 1) if all_regs.size else 0})
        elif any(provided):
            raise ValueError("For jackknife cross-correlations, region columns must be supplied for all participating catalogs or for none of them.")
        else:
            nregions = config.jackknife.nregions
            if nregions is None:
                nregions = choose_default_nregions(config.binning.nsep)
            catalogs = [
                (data1, config.columns_data1.ra, config.columns_data1.dec, "data"),
                (data2, config.columns_data2.ra, config.columns_data2.dec, "data"),
            ]
            if random1 is not None:
                catalogs.append((random1, config.columns_random1.ra, config.columns_random1.dec, "random"))
            if random2 is not None:
                catalogs.append((random2, config.columns_random2.ra, config.columns_random2.dec, "random"))
            assignments, _centers = _auto_region_ids(catalogs, nregions=min(int(nregions), max(1, sum(catalog_nrows(cat[0]) for cat in catalogs))), seed=config.jackknife.seed, geometry_from=config.jackknife.geometry_from)
            reg1, reg2 = assignments[0], assignments[1]
            idx = 2
            if random1 is not None:
                reg_r1 = assignments[idx]; idx += 1
            if random2 is not None:
                reg_r2 = assignments[idx]
            all_regs = np.concatenate(assignments) if assignments else np.empty(0, dtype=np.int32)
            jk_meta.update({"jk_region_source": "auto", "jk_nregions": int(np.max(all_regs) + 1) if all_regs.size else 0, "jk_geometry_from": config.jackknife.geometry_from})
    prep1 = _prepare_sample(data1, ra_col=config.columns_data1.ra, dec_col=config.columns_data1.dec, wei_col=config.columns_data1.weight, sbound=sbound, mxh1=h1, mxh2=h2, pxorder=config.grid.pxorder, region_id=reg1)
    prep_r1 = None if random1 is None else _prepare_sample(random1, ra_col=config.columns_random1.ra, dec_col=config.columns_random1.dec, wei_col=config.columns_random1.weight, sbound=sbound, mxh1=h1, mxh2=h2, pxorder=config.grid.pxorder, region_id=reg_r1)
    prep2 = _prepare_sample(data2, ra_col=config.columns_data2.ra, dec_col=config.columns_data2.dec, wei_col=config.columns_data2.weight, sbound=sbound, mxh1=h1, mxh2=h2, pxorder=config.grid.pxorder, region_id=reg2)
    prep_r2 = None if random2 is None else _prepare_sample(random2, ra_col=config.columns_random2.ra, dec_col=config.columns_random2.dec, wei_col=config.columns_random2.weight, sbound=sbound, mxh1=h1, mxh2=h2, pxorder=config.grid.pxorder, region_id=reg_r2)
    grid_common = dict(grid_info)
    grid_common.setdefault("sbound", tuple(float(v) for v in sbound))
    meta = {
        "theta_edges": np.asarray(sep_edges, dtype=np.float64),
        "theta_centers": np.asarray(sep_centers, dtype=np.float64),
        "theta_delta": np.asarray(sep_delta, dtype=np.float64),
        "sbound": tuple(float(v) for v in sbound),
        "grid_common": grid_common,
    }
    meta.update(jk_meta)
    return prep1, prep_r1, prep2, prep_r2, meta


# def prepare_angular_cross(data1, random1, data2, random2, config: AngularCrossConfig):
#     sep_edges, (_, sep_centers, sep_delta) = makebins(config.binning.nsep, config.binning.sepmin, config.binning.dsep, config.binning.logsep)
#     ra_arrays = [np.asarray(_col(data1, config.columns_data1.ra, dtype=np.float64)), np.asarray(_col(random1, config.columns_random1.ra, dtype=np.float64)), np.asarray(_col(data2, config.columns_data2.ra, dtype=np.float64))]
#     dec_arrays = [np.asarray(_col(data1, config.columns_data1.dec, dtype=np.float64)), np.asarray(_col(random1, config.columns_random1.dec, dtype=np.float64)), np.asarray(_col(data2, config.columns_data2.dec, dtype=np.float64))]
#     if random2 is not None:
#         ra_arrays.append(np.asarray(_col(random2, config.columns_random2.ra, dtype=np.float64)))
#         dec_arrays.append(_col(random2, config.columns_random2.dec, dtype=np.float64))
#     sbound = bound2d(dec_arrays, ra_arrays)
#     h1, h2, grid_info = _autogrid_cross_common(data1, random1, data2, random2, config)
#     prep1 = _prepare_sample(data1, ra_col=config.columns_data1.ra, dec_col=config.columns_data1.dec, wei_col=config.columns_data1.weight, sbound=sbound, mxh1=h1, mxh2=h2, pxorder=config.grid.pxorder)
#     prepr1 = _prepare_sample(random1, ra_col=config.columns_random1.ra, dec_col=config.columns_random1.dec, wei_col=config.columns_random1.weight, sbound=sbound, mxh1=h1, mxh2=h2, pxorder=config.grid.pxorder)
#     prep2 = _prepare_sample(data2, ra_col=config.columns_data2.ra, dec_col=config.columns_data2.dec, wei_col=config.columns_data2.weight, sbound=sbound, mxh1=h1, mxh2=h2, pxorder=config.grid.pxorder)
#     prepr2 = None
#     if random2 is not None:
#         prepr2 = _prepare_sample(random2, ra_col=config.columns_random2.ra, dec_col=config.columns_random2.dec, wei_col=config.columns_random2.weight, sbound=sbound, mxh1=h1, mxh2=h2, pxorder=config.grid.pxorder)
#     metadata = {
#         "theta_edges": np.asarray(sep_edges, dtype=np.float64),
#         "theta_centers": np.asarray(sep_centers, dtype=np.float64),
#         "theta_delta": np.asarray(sep_delta, dtype=np.float64),
#         "cross0": cross0guess(np.asarray(_col(data1, config.columns_data1.ra, dtype=np.float64))),
#         "grid_shared": {"mxh1": h1, "mxh2": h2, **grid_info},
#     }
#     return prep1, prepr1, prep2, prepr2, metadata

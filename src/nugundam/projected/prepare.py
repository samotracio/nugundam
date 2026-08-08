"""Projected sample preparation, distance handling, and 3D gridding utilities."""
from __future__ import annotations

from typing import Iterable
import time

import numpy as np
import nugundam.cflibfor as cff

from ..core.catalogs import catalog_get_column, catalog_has_column, catalog_nrows
from ..core.common import makebins, radec2xyz
from ..core.jackknife import build_shared_sky_regions, choose_default_nregions, normalize_region_labels
from .models import (
    PreparedProjectedSample,
    ProjectedAutoConfig,
    ProjectedCrossConfig,
    ProjectedCatalogColumns,
    ProjectedGridSpec,
    ProjectedPdfSpec,
    ProjectedPdfSourceSpec,
)
from .gmm_compress import compress_pdf_segments
from .quantile_compress import compress_pdf_quantiles
from .pdf_common import load_pdf_matrix, resolve_common_chi_grid, resolve_common_chi_edges



def _refine_edge_histogram_pdfs(matrix: np.ndarray, chi_edges: np.ndarray, nsub: int, *, label: str = "pdf"):
    """Approximate edge-grid histogram PDFs on a refined chi-center grid.

    Each original PDF value is interpreted as probability mass uniformly
    distributed inside the corresponding chi-edge interval.  The interval is
    split into ``nsub`` equal-width sub-bins and the mass is distributed
    equally among them.  This is a controlled quadrature/refinement option for
    exact/ePDF multi-pi runs with pi bins finer than the original chi grid.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    chi_edges = np.asarray(chi_edges, dtype=np.float64)
    nsub = int(nsub)
    if nsub <= 1:
        centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])
        return np.asarray(centers, dtype=np.float64), matrix
    if matrix.ndim != 2:
        raise ValueError(f"{label} PDF matrix must be two-dimensional for edge refinement.")
    if chi_edges.ndim != 1 or chi_edges.size != matrix.shape[1] + 1:
        raise ValueError(
            f"{label} edge refinement requires chi_edges length = matrix.shape[1] + 1 "
            f"({chi_edges.size} != {matrix.shape[1]} + 1)."
        )
    if np.any(np.diff(chi_edges) <= 0.0):
        raise ValueError(f"{label} chi edge grid must be strictly increasing for edge refinement.")

    left = chi_edges[:-1]
    width = np.diff(chi_edges)
    frac = (np.arange(nsub, dtype=np.float64) + 0.5) / float(nsub)
    refined_grid = (left[:, None] + width[:, None] * frac[None, :]).reshape(-1)
    refined_prob = np.repeat(matrix / float(nsub), nsub, axis=1)
    return np.asarray(refined_grid, dtype=np.float64), np.asarray(refined_prob, dtype=np.float64)

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
    return np.ravel(np.asarray(values, dtype=np.float64))


def _col(table, name: str, *, dtype=np.float64) -> np.ndarray:
    return np.asarray(catalog_get_column(table, name, dtype=dtype), dtype=dtype)


def _resolve_pdf_columns(pdf: ProjectedPdfSpec):
    """Resolve GMM column names from the PDF spec."""
    k = int(pdf.k)
    if k <= 0:
        raise ValueError("pdf.k must be a positive integer.")

    def _build(prefix: str | None, cols: list[str] | None, label: str) -> list[str]:
        if cols is not None:
            if len(cols) != k:
                raise ValueError(f"pdf.{label}_cols must have length k={k}.")
            return [str(c) for c in cols]
        if prefix is None:
            raise ValueError(f"Either pdf.{label}_cols or pdf.{label}_prefix must be provided.")
        base = int(getattr(pdf, 'index_base', 0) or 0)
        return [f"{prefix}{i}" for i in range(base, base + k)]

    alpha_cols = _build(getattr(pdf, 'alpha_prefix', None), getattr(pdf, 'alpha_cols', None), 'alpha')
    mu_cols = _build(getattr(pdf, 'mu_prefix', None), getattr(pdf, 'mu_cols', None), 'mu')
    sig_cols = _build(getattr(pdf, 'sigma_prefix', None), getattr(pdf, 'sigma_cols', None), 'sigma')
    return alpha_cols, mu_cols, sig_cols


def _validate_pdf_source_config(config, *, cross: bool) -> None:
    spec = getattr(config, "pdf_source", None)
    if spec is None or not bool(getattr(spec, "enabled", False)):
        return
    if not bool(getattr(getattr(config, "pdf", None), "enabled", False)):
        raise ValueError("config.pdf_source.enabled requires config.pdf.enabled=True.")
    pdf_kind = str(getattr(config.pdf, "kind", "gmm_chi")).strip().lower()
    if pdf_kind not in {"gmm_chi", "grid_chi_exact", "quantile_chi"}:
        raise ValueError("config.pdf_source currently supports pdf.kind='gmm_chi', 'grid_chi_exact', or 'quantile_chi'.")
    if pdf_kind == "gmm_chi" and str(getattr(spec, "compressor", "segments_equal_mass")).strip().lower() != "segments_equal_mass":
        raise ValueError("config.pdf_source.compressor must be 'segments_equal_mass'.")
    if getattr(spec, "pdf_random", None) is not None or getattr(spec, "pdf_random1", None) is not None or getattr(spec, "pdf_random2", None) is not None:
        raise NotImplementedError("Compiled projected pdf_source mode currently supports random PDFs only through pdf.random_pdf_policy='inherit'.")
    if cross:
        if getattr(spec, "pdf_data1", None) is None or getattr(spec, "pdf_data2", None) is None:
            raise ValueError("Compiled projected cross pdf_source mode requires both pdf_source.pdf_data1 and pdf_source.pdf_data2.")
    else:
        if getattr(spec, "pdf_data", None) is None:
            raise ValueError("Compiled projected auto pdf_source mode requires pdf_source.pdf_data.")

def _read_pdf_gmm_chi(table, pdf: ProjectedPdfSpec, *, config=None, pdf_source: ProjectedPdfSourceSpec | None = None, source=None, label: str = "pdf"):
    """Read or build per-object chi-space GMM parameters for projected PDF mode."""
    source_matrix = source
    source_meta = pdf_source
    if source_meta is None and source is not None and hasattr(source, "chi_grid"):
        source_meta = source
    if source_matrix is None and source_meta is not None and bool(getattr(source_meta, "enabled", False)):
        source_matrix = source_meta
    if source_matrix is not None:
        if config is None:
            raise ValueError("config is required when building projected GMM PDFs from empirical pdf_source inputs.")
        meta_spec = source_meta if source_meta is not None else source_matrix
        grid_kind = str(getattr(meta_spec, "grid_kind", "centers")).strip().lower()
        chi_grid = resolve_common_chi_grid(
            z_grid=getattr(meta_spec, "z_grid", None),
            chi_grid=getattr(meta_spec, "chi_grid", None),
            config=config,
            grid_kind=grid_kind,
            label=label,
        )
        chi_edges = resolve_common_chi_edges(
            z_grid=getattr(meta_spec, "z_grid", None),
            chi_grid=getattr(meta_spec, "chi_grid", None),
            config=config,
            grid_kind=grid_kind,
            label=label,
        )
        matrix = load_pdf_matrix(source_matrix, table, nrows=catalog_nrows(table), label=label)
        if matrix.shape[1] != chi_grid.size:
            raise ValueError(
                f"{label} matrix column count {matrix.shape[1]} does not match the resolved chi-grid length {chi_grid.size}."
            )
        edge_moments = bool(getattr(meta_spec, "edge_moments", True)) and grid_kind == "edges"
        return compress_pdf_segments(
            matrix,
            chi_grid,
            k=int(pdf.k),
            compressor=str(getattr(meta_spec, "compressor", "segments_equal_mass")),
            eps=float(getattr(meta_spec, "eps", 0.0)),
            sigma_floor=float(getattr(meta_spec, "sigma_floor", 1.0e-6)),
            chi_edges=chi_edges,
            edge_moments=edge_moments,
        )

    alpha_cols, mu_cols, sig_cols = _resolve_pdf_columns(pdf)
    alpha = np.vstack([_col(table, c, dtype=np.float64) for c in alpha_cols])
    mu = np.vstack([_col(table, c, dtype=np.float64) for c in mu_cols])
    sig = np.vstack([_col(table, c, dtype=np.float64) for c in sig_cols])

    if np.any(sig < 0.0):
        raise ValueError('PDF sigma columns must be non-negative.')
    asum = np.sum(alpha, axis=0)
    bad = asum <= 0.0
    if np.any(bad):
        raise ValueError('PDF alpha columns must sum to > 0 for all objects.')
    alpha = alpha / asum[None, :]
    return alpha, mu, sig


def _gmm_mean_sigma_eff(alpha: np.ndarray, mu: np.ndarray, sig: np.ndarray):
    """Return mixture mean and effective sigma per object."""
    mean = np.sum(alpha * mu, axis=0)
    # Var = E[sig^2 + mu^2] - mean^2
    ex2 = np.sum(alpha * (sig * sig + mu * mu), axis=0)
    var = np.maximum(0.0, ex2 - mean * mean)
    sigma_eff = np.sqrt(var)
    return mean, sigma_eff


def _gmm_dcang(mu: np.ndarray, sig: np.ndarray, *, nsigma: float = 3.0):
    """Conservative distance scale for angular search (lower tail clip)."""
    dcang = np.min(mu - float(nsigma) * sig, axis=0)
    dcang = np.maximum(dcang, 1.0e-6)
    return dcang



def _read_pdf_quantile_chi(table, pdf: ProjectedPdfSpec, *, config=None, pdf_source: ProjectedPdfSourceSpec | None = None, source=None, label: str = "pdf"):
    """Read empirical per-object PDFs and compress them to chi quantiles."""
    source_matrix = source
    source_meta = pdf_source
    if source_meta is None and source is not None and hasattr(source, "chi_grid"):
        source_meta = source
    if source_matrix is None and source_meta is not None and bool(getattr(source_meta, "enabled", False)):
        source_matrix = source_meta
    if source_matrix is None:
        raise ValueError("quantile_chi mode requires config.pdf_source inputs; precomputed quantile columns are not supported in this branch.")
    if config is None:
        raise ValueError("config is required when building projected quantile PDFs from empirical pdf_source inputs.")

    meta_spec = source_meta if source_meta is not None else source_matrix
    grid_kind = str(getattr(meta_spec, "grid_kind", "centers")).strip().lower()
    chi_grid = resolve_common_chi_grid(
        z_grid=getattr(meta_spec, "z_grid", None),
        chi_grid=getattr(meta_spec, "chi_grid", None),
        config=config,
        grid_kind=grid_kind,
        label=label,
    )
    chi_edges = resolve_common_chi_edges(
        z_grid=getattr(meta_spec, "z_grid", None),
        chi_grid=getattr(meta_spec, "chi_grid", None),
        config=config,
        grid_kind=grid_kind,
        label=label,
    )
    matrix = load_pdf_matrix(source_matrix, table, nrows=catalog_nrows(table), label=label)
    expected = (chi_edges.size - 1) if chi_edges is not None else chi_grid.size
    if matrix.shape[1] != expected:
        grid_label = "chi-edge bins" if chi_edges is not None else "chi-grid centers"
        raise ValueError(
            f"{label} matrix column count {matrix.shape[1]} does not match the resolved {grid_label} length {expected}."
        )
    storage = str(getattr(pdf, "quantile_storage", "float32")).strip().lower()
    if storage not in {"float32", "float64"}:
        raise NotImplementedError("quantile_storage currently supports only 'float32' and 'float64' in this branch; uint16 is reserved for the optimized backend.")
    return compress_pdf_quantiles(
        matrix,
        chi_grid,
        nquant=int(getattr(pdf, "nquant", 16)),
        chi_edges=chi_edges,
        eps=float(getattr(meta_spec, "eps", 0.0)),
        positions=str(getattr(pdf, "quantile_positions", "midpoint")),
        dtype=storage,
        label=label,
    )

def _read_pdf_grid_exact(table, pdf: ProjectedPdfSpec, *, config=None, pdf_source: ProjectedPdfSourceSpec | None = None, source=None, label: str = "pdf"):
    """Read the real empirical per-object PDF rows on a shared chi grid."""
    source_matrix = source
    source_meta = pdf_source
    if source_meta is None and source is not None and hasattr(source, "chi_grid"):
        source_meta = source
    if source_matrix is None and source_meta is not None and bool(getattr(source_meta, "enabled", False)):
        source_matrix = source_meta
    if source_matrix is None:
        raise ValueError("grid_chi_exact mode requires config.pdf_source inputs; precomputed exact-grid PDF columns are not supported.")
    if config is None:
        raise ValueError("config is required when building projected exact-grid PDFs from empirical pdf_source inputs.")
    meta_spec = source_meta if source_meta is not None else source_matrix
    grid_kind = str(getattr(meta_spec, "grid_kind", "centers")).strip().lower()
    edge_refine = int(getattr(meta_spec, "edge_refine", 1))
    if edge_refine < 1:
        raise ValueError("config.pdf_source.edge_refine must be >= 1.")

    matrix = load_pdf_matrix(source_matrix, table, nrows=catalog_nrows(table), label=label)

    if grid_kind == "edges" and edge_refine > 1:
        chi_edges = resolve_common_chi_edges(
            z_grid=getattr(meta_spec, "z_grid", None),
            chi_grid=getattr(meta_spec, "chi_grid", None),
            config=config,
            grid_kind=grid_kind,
            label=label,
        )
        if matrix.shape[1] != chi_edges.size - 1:
            raise ValueError(
                f"{label} matrix column count {matrix.shape[1]} does not match the supplied edge-grid "
                f"length {chi_edges.size} (expected {chi_edges.size - 1})."
            )
        chi_grid, matrix = _refine_edge_histogram_pdfs(matrix, chi_edges, edge_refine, label=label)
    else:
        chi_grid = resolve_common_chi_grid(
            z_grid=getattr(meta_spec, "z_grid", None),
            chi_grid=getattr(meta_spec, "chi_grid", None),
            config=config,
            grid_kind=grid_kind,
            label=label,
        )
        if matrix.shape[1] != chi_grid.size:
            raise ValueError(
                f"{label} matrix column count {matrix.shape[1]} does not match the resolved chi-grid length {chi_grid.size}."
            )

    return np.asarray(chi_grid, dtype=np.float64), np.asarray(matrix, dtype=np.float64)


def _grid_exact_support(prob: np.ndarray, chi_grid: np.ndarray, *, prob_floor: float):
    """Return PDF means, support bounds, and search proxies for exact-grid PDFs."""
    prob = np.asarray(prob, dtype=np.float64)
    chi_grid = np.asarray(chi_grid, dtype=np.float64)
    if prob.ndim != 2:
        raise ValueError("Empirical PDF matrix must be two-dimensional.")
    if chi_grid.ndim != 1 or chi_grid.size != prob.shape[1]:
        raise ValueError("Shared chi-grid must be one-dimensional and match the PDF matrix width.")
    row_sum = np.sum(prob, axis=1)
    if np.any(row_sum <= 0.0):
        raise ValueError("Empirical PDF rows must sum to a positive value.")
    means = np.sum(prob * chi_grid[None, :], axis=1)
    lo = np.zeros(prob.shape[0], dtype=np.int32)
    hi = np.zeros(prob.shape[0], dtype=np.int32)
    chi_lo = np.empty(prob.shape[0], dtype=np.float64)
    chi_hi = np.empty(prob.shape[0], dtype=np.float64)
    floor = float(max(prob_floor, 0.0))
    for i in range(prob.shape[0]):
        active = np.flatnonzero(prob[i] > floor)
        if active.size == 0:
            active = np.array([int(np.argmax(prob[i]))], dtype=np.int64)
        lo[i] = int(active[0])
        hi[i] = int(active[-1])
        chi_lo[i] = float(chi_grid[lo[i]])
        chi_hi[i] = float(chi_grid[hi[i]])
    halfspan = np.maximum(means - chi_lo, chi_hi - means)
    dcang = np.maximum(chi_lo, 1.0e-6)
    return means, lo, hi, chi_lo, chi_hi, halfspan, dcang


def _has_any_pdf_payload(sample) -> bool:
    return bool(getattr(sample, "pdf_idx", None) is not None and str(getattr(sample, "pdf_repr", "none")) != "none")


def _pdf_summary_from_source(table, config, *, source=None, label: str = "pdf"):
    """Return summary statistics used to size the projected PDF search grid."""
    kind = str(getattr(config.pdf, "kind", "gmm_chi")).strip().lower()
    if kind == "gmm_chi":
        alpha, mu, sig = _read_pdf_gmm_chi(table, config.pdf, config=config, pdf_source=getattr(config, "pdf_source", None), source=source, label=label)
        dist, sig_eff = _gmm_mean_sigma_eff(alpha, mu, sig)
        return {
            "kind": kind,
            "dist": np.asarray(dist, dtype=np.float64),
            "halfspan": np.asarray(sig_eff, dtype=np.float64),
            "dcang": _gmm_dcang(mu, sig, nsigma=3.0),
        }
    if kind == "quantile_chi":
        qchi, qmean, qlo, qhi = _read_pdf_quantile_chi(table, config.pdf, config=config, pdf_source=getattr(config, "pdf_source", None), source=source, label=label)
        return {
            "kind": kind,
            "dist": np.asarray(qmean, dtype=np.float64),
            "halfspan": np.maximum(np.asarray(qmean, dtype=np.float64) - np.asarray(qlo, dtype=np.float64), np.asarray(qhi, dtype=np.float64) - np.asarray(qmean, dtype=np.float64)),
            "dcang": np.maximum(np.asarray(qlo, dtype=np.float64), 1.0e-6),
        }
    if kind == "grid_chi_exact":
        chi_grid, prob = _read_pdf_grid_exact(table, config.pdf, config=config, pdf_source=getattr(config, "pdf_source", None), source=source, label=label)
        dist, lo, hi, chi_lo, chi_hi, halfspan, dcang = _grid_exact_support(prob, chi_grid, prob_floor=float(getattr(config.pdf, "prob_floor", 0.0)))
        return {
            "kind": kind,
            "dist": np.asarray(dist, dtype=np.float64),
            "halfspan": np.asarray(halfspan, dtype=np.float64),
            "chi_lo": np.asarray(chi_lo, dtype=np.float64),
            "chi_hi": np.asarray(chi_hi, dtype=np.float64),
            "dcang": np.asarray(dcang, dtype=np.float64),
        }
    raise ValueError(f"Unsupported pdf.kind={config.pdf.kind!r}.")


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


def _prepared_projected_from_arrays(
    *,
    ra,
    dec,
    dist,
    weights,
    sbound,
    mxh1: int,
    mxh2: int,
    mxh3: int,
    pi_edges,
    grid_meta: dict,
    region_id=None,
    dcang=None,
    pdf_repr: str = "none",
    pdf_k: int = 0,
    pdf_alpha_lib=None,
    pdf_mu_lib=None,
    pdf_sig_lib=None,
    pdf_prob_lib=None,
    pdf_cdf_lib=None,
    pdf_grid=None,
    pdf_lo_idx=None,
    pdf_hi_idx=None,
    pdf_qchi_lib=None,
    pdf_qlo_lib=None,
    pdf_qhi_lib=None,
    pdf_idx=None,
    sort_rows: bool = False,
) -> PreparedProjectedSample:
    """Build a prepared projected sample from already materialized row arrays."""
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float32)
    region_id = None if region_id is None else np.asarray(region_id, dtype=np.int32)
    dcang = None if dcang is None else np.asarray(dcang, dtype=np.float64)
    pdf_idx = None if pdf_idx is None else np.asarray(pdf_idx, dtype=np.int32)
    if sort_rows and len(ra) > 0:
        sidx = _sort_index_3d(
            ra,
            dec,
            dist,
            sbound=sbound,
            mxh1=int(mxh1),
            mxh2=int(mxh2),
            mxh3=int(mxh3),
            pxorder=grid_meta.get("pxorder", "natural"),
        )
        ra, dec, dist = ra[sidx], dec[sidx], dist[sidx]
        weights = np.asarray(weights[sidx], dtype=np.float32)
        if region_id is not None:
            region_id = np.asarray(region_id[sidx], dtype=np.int32)
        if dcang is not None:
            dcang = np.asarray(dcang[sidx], dtype=np.float64)
        if pdf_idx is not None:
            pdf_idx = np.asarray(pdf_idx[sidx], dtype=np.int32)
    pi_edges_search = grid_meta.get('pi_edges_search', None)
    sk_pi_edges = np.asarray(pi_edges_search if pi_edges_search is not None else pi_edges, dtype=np.float64)
    sk, ll = _build_skll3d(int(mxh1), int(mxh2), int(mxh3), ra, dec, dist, np.asarray(sbound, dtype=np.float64), sk_pi_edges)
    x, y, z = radec2xyz(np.deg2rad(ra), np.deg2rad(dec))
    return PreparedProjectedSample(
        table=None,
        ra=np.asarray(ra, dtype=np.float64),
        dec=np.asarray(dec, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float32),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        z=np.asarray(z, dtype=np.float64),
        sk=np.asarray(sk),
        ll=np.asarray(ll),
        wunit=bool(np.allclose(weights, 1.0)),
        sbound=tuple(float(v) for v in sbound),
        mxh1=int(mxh1),
        mxh2=int(mxh2),
        mxh3=int(mxh3),
        dcang=None if dcang is None else np.asarray(dcang, dtype=np.float64),
        pdf_repr=str(pdf_repr),
        pdf_k=int(pdf_k),
        pdf_alpha_lib=pdf_alpha_lib,
        pdf_mu_lib=pdf_mu_lib,
        pdf_sig_lib=pdf_sig_lib,
        pdf_prob_lib=pdf_prob_lib,
        pdf_cdf_lib=pdf_cdf_lib,
        pdf_grid=pdf_grid,
        pdf_lo_idx=None if pdf_lo_idx is None else np.asarray(pdf_lo_idx, dtype=np.int32),
        pdf_hi_idx=None if pdf_hi_idx is None else np.asarray(pdf_hi_idx, dtype=np.int32),
        pdf_qchi_lib=None if pdf_qchi_lib is None else np.asarray(pdf_qchi_lib, dtype=np.float64, order='F'),
        pdf_qlo_lib=None if pdf_qlo_lib is None else np.asarray(pdf_qlo_lib, dtype=np.float64),
        pdf_qhi_lib=None if pdf_qhi_lib is None else np.asarray(pdf_qhi_lib, dtype=np.float64),
        pdf_idx=None if pdf_idx is None else np.asarray(pdf_idx, dtype=np.int32),
        region_id=None if region_id is None else np.asarray(region_id, dtype=np.int32),
        grid_meta=dict(grid_meta),
        nrows=int(len(ra)),
    )


def subset_prepared_projected_sample(sample: PreparedProjectedSample, keep, *, pi_edges, regrid: bool = False) -> PreparedProjectedSample:
    idx = np.asarray(keep)
    if idx.dtype == bool:
        idx = np.flatnonzero(idx)
    else:
        idx = np.asarray(idx, dtype=np.int64)
    ra = np.asarray(sample.ra[idx], dtype=np.float64)
    dec = np.asarray(sample.dec[idx], dtype=np.float64)
    dist = np.asarray(sample.dist[idx], dtype=np.float64)
    dcang = None if sample.dcang is None else np.asarray(sample.dcang[idx], dtype=np.float64)
    pdf_idx = None if sample.pdf_idx is None else np.asarray(sample.pdf_idx[idx], dtype=np.int32)
    weights = np.asarray(sample.weights[idx], dtype=np.float32)
    region_id = None if sample.region_id is None else np.asarray(sample.region_id[idx], dtype=np.int32)

    mxh1 = int(sample.mxh1)
    mxh2 = int(sample.mxh2)
    mxh3 = int(sample.mxh3)
    grid_meta = dict(sample.grid_meta)
    sort_rows = False
    if regrid and len(idx) > 0 and bool(grid_meta.get("autogrid", True)):
        mxh1, mxh2, mxh3, _ = best_skgrid_3d_legacy(
            len(ra),
            ra,
            sbound=sample.sbound,
            nsepv=int((len(grid_meta.get('pi_edges_search')) - 1) if grid_meta.get('pi_edges_search', None) is not None else grid_meta.get('nsepv', max(1, len(pi_edges) - 1))),
            dsepv=float((np.asarray(grid_meta.get('pi_edges_search'), dtype=np.float64)[1] - np.asarray(grid_meta.get('pi_edges_search'), dtype=np.float64)[0]) if grid_meta.get('pi_edges_search', None) is not None else grid_meta.get('dsepv', np.asarray(pi_edges, dtype=np.float64)[1] - np.asarray(pi_edges, dtype=np.float64)[0] if len(pi_edges) > 1 else 1.0)),
            dens=grid_meta.get("dens", None),
        )
        sort_rows = True
    return _prepared_projected_from_arrays(
        ra=ra,
        dec=dec,
        dist=dist,
        weights=weights,
        sbound=sample.sbound,
        mxh1=mxh1,
        mxh2=mxh2,
        mxh3=mxh3,
        pi_edges=pi_edges,
        grid_meta=grid_meta,
        region_id=region_id,
        dcang=dcang,
        pdf_repr=str(getattr(sample, 'pdf_repr', 'none')),
        pdf_k=int(getattr(sample, 'pdf_k', 0) or 0),
        pdf_alpha_lib=getattr(sample, 'pdf_alpha_lib', None),
        pdf_mu_lib=getattr(sample, 'pdf_mu_lib', None),
        pdf_sig_lib=getattr(sample, 'pdf_sig_lib', None),
        pdf_prob_lib=getattr(sample, 'pdf_prob_lib', None),
        pdf_cdf_lib=getattr(sample, 'pdf_cdf_lib', None),
        pdf_grid=getattr(sample, 'pdf_grid', None),
        pdf_lo_idx=getattr(sample, 'pdf_lo_idx', None),
        pdf_hi_idx=getattr(sample, 'pdf_hi_idx', None),
        pdf_qchi_lib=getattr(sample, 'pdf_qchi_lib', None),
        pdf_qlo_lib=getattr(sample, 'pdf_qlo_lib', None),
        pdf_qhi_lib=getattr(sample, 'pdf_qhi_lib', None),
        pdf_idx=pdf_idx,
        sort_rows=sort_rows,
    )


def resample_prepared_projected_sample(sample: PreparedProjectedSample, draw_idx, *, pi_edges, regrid: bool = False) -> PreparedProjectedSample:
    """Bootstrap-style resampling of a prepared projected sample.

    Unlike :func:`subset_prepared_projected_sample`, ``draw_idx`` may contain
    repeated indices. The returned sample is re-sorted in 3D cell order so the
    linked-list layout remains valid for the compiled counters.
    """
    idx = np.asarray(draw_idx, dtype=np.int64)
    ra = np.asarray(sample.ra[idx], dtype=np.float64)
    dec = np.asarray(sample.dec[idx], dtype=np.float64)
    dist = np.asarray(sample.dist[idx], dtype=np.float64)
    dcang = None if sample.dcang is None else np.asarray(sample.dcang[idx], dtype=np.float64)
    pdf_idx = None if sample.pdf_idx is None else np.asarray(sample.pdf_idx[idx], dtype=np.int32)
    weights = np.asarray(sample.weights[idx], dtype=np.float32)
    region_id = None if sample.region_id is None else np.asarray(sample.region_id[idx], dtype=np.int32)

    mxh1 = int(sample.mxh1)
    mxh2 = int(sample.mxh2)
    mxh3 = int(sample.mxh3)
    grid_meta = dict(sample.grid_meta)
    if regrid and len(idx) > 0 and bool(grid_meta.get("autogrid", True)):
        mxh1, mxh2, mxh3, _ = best_skgrid_3d_legacy(
            len(ra),
            ra,
            sbound=sample.sbound,
            nsepv=int((len(grid_meta.get('pi_edges_search')) - 1) if grid_meta.get('pi_edges_search', None) is not None else grid_meta.get('nsepv', max(1, len(pi_edges) - 1))),
            dsepv=float((np.asarray(grid_meta.get('pi_edges_search'), dtype=np.float64)[1] - np.asarray(grid_meta.get('pi_edges_search'), dtype=np.float64)[0]) if grid_meta.get('pi_edges_search', None) is not None else grid_meta.get('dsepv', np.asarray(pi_edges, dtype=np.float64)[1] - np.asarray(pi_edges, dtype=np.float64)[0] if len(pi_edges) > 1 else 1.0)),
            dens=grid_meta.get("dens", None),
        )
    return _prepared_projected_from_arrays(
        ra=ra,
        dec=dec,
        dist=dist,
        weights=weights,
        sbound=sample.sbound,
        mxh1=mxh1,
        mxh2=mxh2,
        mxh3=mxh3,
        pi_edges=pi_edges,
        grid_meta=grid_meta,
        region_id=region_id,
        dcang=dcang,
        pdf_repr=str(getattr(sample, 'pdf_repr', 'none')),
        pdf_k=int(getattr(sample, 'pdf_k', 0) or 0),
        pdf_alpha_lib=getattr(sample, 'pdf_alpha_lib', None),
        pdf_mu_lib=getattr(sample, 'pdf_mu_lib', None),
        pdf_sig_lib=getattr(sample, 'pdf_sig_lib', None),
        pdf_prob_lib=getattr(sample, 'pdf_prob_lib', None),
        pdf_cdf_lib=getattr(sample, 'pdf_cdf_lib', None),
        pdf_grid=getattr(sample, 'pdf_grid', None),
        pdf_lo_idx=getattr(sample, 'pdf_lo_idx', None),
        pdf_hi_idx=getattr(sample, 'pdf_hi_idx', None),
        pdf_qchi_lib=getattr(sample, 'pdf_qchi_lib', None),
        pdf_qlo_lib=getattr(sample, 'pdf_qlo_lib', None),
        pdf_qhi_lib=getattr(sample, 'pdf_qhi_lib', None),
        pdf_idx=pdf_idx,
        sort_rows=True,
    )


def rebuild_pdf_random_inheritance_from_prepared(random_sample: PreparedProjectedSample, data_lib: PreparedProjectedSample, config, *, pi_edges, seed: int | None = None, regrid: bool = False) -> PreparedProjectedSample:
    """Rebuild inherited PDF assignments for a prepared random sample."""
    if not _has_any_pdf_payload(data_lib):
        raise ValueError('data_lib must carry a valid PDF library in order to inherit PDFs.')
    nrand = int(random_sample.nrows)
    rng_seed = int(config.pdf.seed if seed is None else seed)
    rng = np.random.default_rng(rng_seed)
    pick = rng.integers(0, max(1, int(data_lib.nrows)), size=nrand, dtype=np.int64) if nrand else np.empty(0, dtype=np.int64)
    pdf_idx = (pick + 1).astype(np.int32)
    dist = np.asarray(data_lib.dist[pick], dtype=np.float64) if nrand else np.empty(0, dtype=np.float64)
    dcang = np.asarray(data_lib.dcang[pick], dtype=np.float64) if (nrand and data_lib.dcang is not None) else None

    mxh1 = int(random_sample.mxh1)
    mxh2 = int(random_sample.mxh2)
    mxh3 = int(random_sample.mxh3)
    grid_meta = dict(random_sample.grid_meta)
    if regrid and nrand > 0 and bool(grid_meta.get("autogrid", True)):
        mxh1, mxh2, mxh3, _ = best_skgrid_3d_legacy(
            nrand,
            np.asarray(random_sample.ra, dtype=np.float64),
            sbound=random_sample.sbound,
            nsepv=int((len(grid_meta.get('pi_edges_search')) - 1) if grid_meta.get('pi_edges_search', None) is not None else grid_meta.get('nsepv', max(1, len(pi_edges) - 1))),
            dsepv=float((np.asarray(grid_meta.get('pi_edges_search'), dtype=np.float64)[1] - np.asarray(grid_meta.get('pi_edges_search'), dtype=np.float64)[0]) if grid_meta.get('pi_edges_search', None) is not None else grid_meta.get('dsepv', np.asarray(pi_edges, dtype=np.float64)[1] - np.asarray(pi_edges, dtype=np.float64)[0] if len(pi_edges) > 1 else 1.0)),
            dens=grid_meta.get("dens", None),
        )
    return _prepared_projected_from_arrays(
        ra=random_sample.ra,
        dec=random_sample.dec,
        dist=dist,
        weights=np.ones(nrand, dtype=np.float32),
        sbound=random_sample.sbound,
        mxh1=mxh1,
        mxh2=mxh2,
        mxh3=mxh3,
        pi_edges=pi_edges,
        grid_meta=grid_meta,
        region_id=random_sample.region_id,
        dcang=dcang,
        pdf_repr=str(getattr(data_lib, 'pdf_repr', 'none')),
        pdf_k=int(getattr(data_lib, 'pdf_k', 0) or 0),
        pdf_alpha_lib=getattr(data_lib, 'pdf_alpha_lib', None),
        pdf_mu_lib=getattr(data_lib, 'pdf_mu_lib', None),
        pdf_sig_lib=getattr(data_lib, 'pdf_sig_lib', None),
        pdf_prob_lib=getattr(data_lib, 'pdf_prob_lib', None),
        pdf_cdf_lib=getattr(data_lib, 'pdf_cdf_lib', None),
        pdf_grid=getattr(data_lib, 'pdf_grid', None),
        pdf_lo_idx=getattr(data_lib, 'pdf_lo_idx', None),
        pdf_hi_idx=getattr(data_lib, 'pdf_hi_idx', None),
        pdf_qchi_lib=getattr(data_lib, 'pdf_qchi_lib', None),
        pdf_qlo_lib=getattr(data_lib, 'pdf_qlo_lib', None),
        pdf_qhi_lib=getattr(data_lib, 'pdf_qhi_lib', None),
        pdf_idx=pdf_idx,
        sort_rows=True,
    )


def bound3d(dec_arrays, dist_arrays):
    """
    Bound3d.
    
    Parameters
    ----------
    dec_arrays : object
        Value for ``dec_arrays``.
    dist_arrays : object
        Value for ``dist_arrays``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    decs = [_as_1d_float64(a) for a in dec_arrays]
    dists = [_as_1d_float64(a) for a in dist_arrays]
    delta = 1.0e-3
    decmin = max(min(float(np.min(a)) for a in decs) - delta, -90.0)
    decmax = min(max(float(np.max(a)) for a in decs) + delta, 90.0)
    dcmin = max(min(float(np.min(a)) for a in dists) - delta, 0.0)
    dcmax = max(float(np.max(a)) for a in dists) + delta
    return (0.0, 360.0, decmin, decmax, dcmin, dcmax)


def _distance_array(table, columns: ProjectedCatalogColumns, config: ProjectedAutoConfig | ProjectedCrossConfig) -> np.ndarray:
    """
    Distance array.
    
    Parameters
    ----------
    table : object
        Value for ``table``.
    columns : object
        Value for ``columns``.
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
    if config.distance.calcdist:
        try:
            from astropy.cosmology import LambdaCDM
        except Exception as exc:  # pragma: no cover - exercised only when optional dependency is missing
            raise ImportError(
                "Projected correlations with distance.calcdist=True require astropy to be installed. "
                "Either install astropy or provide a precomputed comoving-distance column and set calcdist=False."
            ) from exc
        z = _col(table, columns.redshift)
        cosmo = LambdaCDM(H0=config.distance.h0, Om0=config.distance.omegam, Ode0=config.distance.omegal)
        return _as_1d_float64(cosmo.comoving_distance(z).value)
    return _col(table, columns.distance)


def best_skgrid_3d_legacy(npts, ras, *, sbound, nsepv: int, dsepv: float, dens=None):
    """
    Choose skgrid 3d legacy.
    
    Parameters
    ----------
    npts : object
        Value for ``npts``.
    ras : object
        Value for ``ras``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    nsepv : object
        Value for ``nsepv``. This argument is keyword-only.
    dsepv : object
        Value for ``dsepv``. This argument is keyword-only.
    dens : object, optional
        Value for ``dens``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    if isinstance(npts, (list, tuple)):
        npts_eff = int(sum(int(v) for v in npts))
        ras_eff = np.concatenate([_as_1d_float64(r) for r in ras])
    else:
        npts_eff = int(npts)
        ras_eff = _as_1d_float64(ras)
    if dens is None:
        dens = 18.0 if npts_eff > 100000 else 8.0
    if ras_eff.size == 0:
        samplewidth = 360.0
    else:
        samplewidth = float(np.max(ras_eff) - np.min(ras_eff))
        if samplewidth <= 0.0:
            samplewidth = 360.0
    dcmin, dcmax = float(sbound[4]), float(sbound[5])
    rvmax = float(nsepv) * float(dsepv)
    h3 = max(int((dcmax - dcmin) / max(rvmax, 1.0e-9)), 1)
    h1h2 = npts_eff / (float(dens) * h3)
    h1 = max(int(np.rint(2.92 + 0.05 * np.sqrt(max(npts_eff, 1)))), 1)
    h2 = max(int(np.rint(h1h2 / h1) * (360.0 / samplewidth)), 1)
    return h1, h2, h3, float(dens)


def _cell_indices_3d(ra, dec, dist, *, sbound, mxh1: int, mxh2: int, mxh3: int):
    """
    Cell indices 3d.
    
    Parameters
    ----------
    ra : object
        Value for ``ra``.
    dec : object
        Value for ``dec``.
    dist : object
        Value for ``dist``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    mxh1 : object
        Value for ``mxh1``. This argument is keyword-only.
    mxh2 : object
        Value for ``mxh2``. This argument is keyword-only.
    mxh3 : object
        Value for ``mxh3``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ramin, ramax, decmin, decmax, dmin, dmax = [float(v) for v in sbound]
    ra_span = max(ramax - ramin, 360.0)
    dec_span = max(decmax - decmin, 1.0e-12)
    dist_span = max(dmax - dmin, 1.0e-12)
    qra = np.floor((np.mod(ra - ramin, ra_span)) / ra_span * mxh2).astype(np.int64)
    qdec = np.floor((dec - decmin) / dec_span * mxh1).astype(np.int64)
    qdist = np.floor((dist - dmin) / dist_span * mxh3).astype(np.int64)
    np.clip(qra, 0, mxh2 - 1, out=qra)
    np.clip(qdec, 0, mxh1 - 1, out=qdec)
    np.clip(qdist, 0, mxh3 - 1, out=qdist)
    return qra, qdec, qdist


def _sort_index_3d(ra, dec, dist, *, sbound, mxh1: int, mxh2: int, mxh3: int, pxorder: str) -> np.ndarray:
    """
    Sort index 3d.
    
    Parameters
    ----------
    ra : object
        Value for ``ra``.
    dec : object
        Value for ``dec``.
    dist : object
        Value for ``dist``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    mxh1 : object
        Value for ``mxh1``. This argument is keyword-only.
    mxh2 : object
        Value for ``mxh2``. This argument is keyword-only.
    mxh3 : object
        Value for ``mxh3``. This argument is keyword-only.
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
    mode = str(pxorder or "none").strip().lower()
    if mode in {"none", "null"}:
        return np.arange(len(ra), dtype=np.int64)
    if mode != "natural":
        raise NameError(f"Projected pxorder {pxorder!r} not implemented. Use 'none' or 'natural'.")
    qra, qdec, qdist = _cell_indices_3d(ra, dec, dist, sbound=sbound, mxh1=mxh1, mxh2=mxh2, mxh3=mxh3)
    return np.lexsort((qra, qdec, qdist))


def _build_skll3d_python(mxh1: int, mxh2: int, mxh3: int, ra, dec, dist, sbound, pi_edges):
    """
    Build skll3d python.
    
    Parameters
    ----------
    mxh1 : object
        Value for ``mxh1``.
    mxh2 : object
        Value for ``mxh2``.
    mxh3 : object
        Value for ``mxh3``.
    ra : object
        Value for ``ra``.
    dec : object
        Value for ``dec``.
    dist : object
        Value for ``dist``.
    sbound : object
        Value for ``sbound``.
    pi_edges : object
        Value for ``pi_edges``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    qra, qdec, qdist = _cell_indices_3d(ra, dec, dist, sbound=sbound, mxh1=mxh1, mxh2=mxh2, mxh3=mxh3)
    sk = np.zeros((mxh3, mxh2, mxh1), dtype=np.int32)
    ll = np.zeros(len(ra), dtype=np.int32)
    for i in range(len(ra)):
        c3, c2, c1 = int(qdist[i]), int(qra[i]), int(qdec[i])
        prev = sk[c3, c2, c1]
        ll[i] = prev
        sk[c3, c2, c1] = i + 1
    return sk, ll


def _build_skll3d(mxh1: int, mxh2: int, mxh3: int, ra, dec, dist, sbound, pi_edges):
    """
    Build skll3d.
    
    Parameters
    ----------
    mxh1 : object
        Value for ``mxh1``.
    mxh2 : object
        Value for ``mxh2``.
    mxh3 : object
        Value for ``mxh3``.
    ra : object
        Value for ``ra``.
    dec : object
        Value for ``dec``.
    dist : object
        Value for ``dist``.
    sbound : object
        Value for ``sbound``.
    pi_edges : object
        Value for ``pi_edges``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    try:
        return cff.mod.skll3d(mxh1, mxh2, mxh3, len(ra), ra, dec, dist, sbound, pi_edges, len(pi_edges) - 1)
    except Exception:
        return _build_skll3d_python(mxh1, mxh2, mxh3, ra, dec, dist, sbound, pi_edges)


def _prepare_sample(table, columns: ProjectedCatalogColumns, config, *, sbound, mxh1: int, mxh2: int, mxh3: int, pi_edges, use_weights: bool, region_id=None, grid_meta=None):
    """
    Prepare sample.
    
    Parameters
    ----------
    table : object
        Value for ``table``.
    columns : object
        Value for ``columns``.
    config : object
        Value for ``config``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    mxh1 : object
        Value for ``mxh1``. This argument is keyword-only.
    mxh2 : object
        Value for ``mxh2``. This argument is keyword-only.
    mxh3 : object
        Value for ``mxh3``. This argument is keyword-only.
    pi_edges : object
        Value for ``pi_edges``. This argument is keyword-only.
    use_weights : object
        Value for ``use_weights``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ra = _col(table, columns.ra)
    dec = _col(table, columns.dec)
    dist = _distance_array(table, columns, config)
    if use_weights:
        if catalog_has_column(table, columns.weight):
            weights = _col(table, columns.weight, dtype=np.float64)
        else:
            if config.weights.weight_mode == "weighted":
                raise KeyError(columns.weight)
            weights = np.ones(len(ra), dtype=np.float64)
    else:
        weights = np.ones(len(ra), dtype=np.float64)

    sidx = _sort_index_3d(ra, dec, dist, sbound=sbound, mxh1=mxh1, mxh2=mxh2, mxh3=mxh3, pxorder=config.grid.pxorder)
    ra, dec, dist, weights = ra[sidx], dec[sidx], dist[sidx], weights[sidx]
    if region_id is not None:
        region_id = np.asarray(region_id, dtype=np.int32)[sidx]
    sk, ll = _build_skll3d(mxh1, mxh2, mxh3, ra, dec, dist, np.asarray(sbound, dtype=np.float64), np.asarray(pi_edges, dtype=np.float64))
    x, y, z = radec2xyz(np.deg2rad(ra), np.deg2rad(dec))
    return PreparedProjectedSample(
        table=None,
        ra=ra,
        dec=dec,
        dist=dist,
        weights=np.asarray(weights, dtype=np.float32),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        z=np.asarray(z, dtype=np.float64),
        sk=np.asarray(sk),
        ll=np.asarray(ll),
        wunit=bool(np.allclose(weights, 1.0)),
        sbound=tuple(float(v) for v in sbound),
        mxh1=int(mxh1),
        mxh2=int(mxh2),
        mxh3=int(mxh3),
        region_id=None if region_id is None else np.asarray(region_id, dtype=np.int32),
        grid_meta={} if grid_meta is None else dict(grid_meta),
        nrows=int(len(ra)),
    )


def _grid_for_sample(table, columns, config, *, sbound):
    """
    Grid for sample.
    
    Parameters
    ----------
    table : object
        Value for ``table``.
    columns : object
        Value for ``columns``.
    config : object
        Value for ``config``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ra = _col(table, columns.ra)
    if config.grid.autogrid:
        return best_skgrid_3d_legacy(len(ra), ra, sbound=sbound, nsepv=config.binning.nsepv, dsepv=config.binning.dsepv, dens=config.grid.dens)[:3]
    return int(config.grid.mxh1), int(config.grid.mxh2), int(config.grid.mxh3)



def _prepare_sample_pdf_data(table, columns: ProjectedCatalogColumns, config, *, sbound, mxh1: int, mxh2: int, mxh3: int,
                             pi_edges_search, use_weights: bool, region_id=None, grid_meta=None, pdf_source=None, pdf_label: str = "pdf"):
    """Prepare a data sample with per-object PDF payloads in comoving distance."""
    pdf = config.pdf
    kind = str(pdf.kind).strip().lower()

    ra = _col(table, columns.ra)
    dec = _col(table, columns.dec)

    if use_weights:
        if catalog_has_column(table, columns.weight):
            weights = _col(table, columns.weight, dtype=np.float64)
        else:
            if config.weights.weight_mode == 'weighted':
                raise KeyError(columns.weight)
            weights = np.ones(len(ra), dtype=np.float64)
    else:
        weights = np.ones(len(ra), dtype=np.float64)

    if kind == 'gmm_chi':
        alpha, mu, sig = _read_pdf_gmm_chi(table, pdf, config=config, pdf_source=getattr(config, "pdf_source", None), source=pdf_source, label=pdf_label)
        dist_mean, _sig_eff = _gmm_mean_sigma_eff(alpha, mu, sig)
        dcang = _gmm_dcang(mu, sig, nsigma=3.0)
        sidx = _sort_index_3d(ra, dec, dist_mean, sbound=sbound, mxh1=mxh1, mxh2=mxh2, mxh3=mxh3, pxorder=config.grid.pxorder)
        ra, dec = ra[sidx], dec[sidx]
        dist_mean = dist_mean[sidx]
        weights = weights[sidx]
        dcang = dcang[sidx]
        alpha = np.asarray(alpha[:, sidx], dtype=np.float64, order='F')
        mu = np.asarray(mu[:, sidx], dtype=np.float64, order='F')
        sig = np.asarray(sig[:, sidx], dtype=np.float64, order='F')
        if region_id is not None:
            region_id = np.asarray(region_id, dtype=np.int32)[sidx]
        sk, ll = _build_skll3d(mxh1, mxh2, mxh3, ra, dec, dist_mean, np.asarray(sbound, dtype=np.float64), np.asarray(pi_edges_search, dtype=np.float64))
        x, y, z = radec2xyz(np.deg2rad(ra), np.deg2rad(dec))
        pdf_idx = (np.arange(len(ra), dtype=np.int32) + 1)
        return PreparedProjectedSample(
            table=None,
            ra=np.asarray(ra, dtype=np.float64),
            dec=np.asarray(dec, dtype=np.float64),
            dist=np.asarray(dist_mean, dtype=np.float64),
            dcang=np.asarray(dcang, dtype=np.float64),
            pdf_repr='gmm_chi',
            pdf_k=int(alpha.shape[0]),
            pdf_alpha_lib=alpha,
            pdf_mu_lib=mu,
            pdf_sig_lib=sig,
            pdf_idx=pdf_idx,
            weights=np.asarray(weights, dtype=np.float32),
            x=np.asarray(x, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            z=np.asarray(z, dtype=np.float64),
            sk=np.asarray(sk),
            ll=np.asarray(ll),
            wunit=bool(np.allclose(weights, 1.0)),
            sbound=tuple(float(v) for v in sbound),
            mxh1=int(mxh1),
            mxh2=int(mxh2),
            mxh3=int(mxh3),
            region_id=None if region_id is None else np.asarray(region_id, dtype=np.int32),
            grid_meta={} if grid_meta is None else dict(grid_meta),
            nrows=int(len(ra)),
        )


    if kind == 'quantile_chi':
        _qchi_t0 = time.perf_counter()
        qchi, dist_mean, qlo, qhi = _read_pdf_quantile_chi(table, pdf, config=config, pdf_source=getattr(config, "pdf_source", None), source=pdf_source, label=pdf_label)
        _qchi_compress_time = time.perf_counter() - _qchi_t0
        _qchi_dtype = str(np.asarray(qchi).dtype)
        _qchi_bytes = int(np.asarray(qchi).nbytes + np.asarray(qlo).nbytes + np.asarray(qhi).nbytes)
        _qchi_span = np.asarray(qhi, dtype=np.float64) - np.asarray(qlo, dtype=np.float64)
        dcang = np.maximum(qlo, 1.0e-6)
        sidx = _sort_index_3d(ra, dec, dist_mean, sbound=sbound, mxh1=mxh1, mxh2=mxh2, mxh3=mxh3, pxorder=config.grid.pxorder)
        ra, dec = ra[sidx], dec[sidx]
        dist_mean = np.asarray(dist_mean[sidx], dtype=np.float64)
        weights = weights[sidx]
        dcang = np.asarray(dcang[sidx], dtype=np.float64)
        qchi = np.asarray(qchi[:, sidx], dtype=np.float64, order='F')
        qlo = np.asarray(qlo[sidx], dtype=np.float64)
        qhi = np.asarray(qhi[sidx], dtype=np.float64)
        if region_id is not None:
            region_id = np.asarray(region_id, dtype=np.int32)[sidx]
        sk, ll = _build_skll3d(mxh1, mxh2, mxh3, ra, dec, dist_mean, np.asarray(sbound, dtype=np.float64), np.asarray(pi_edges_search, dtype=np.float64))
        x, y, z = radec2xyz(np.deg2rad(ra), np.deg2rad(dec))
        pdf_idx = (np.arange(len(ra), dtype=np.int32) + 1)
        return PreparedProjectedSample(
            table=None,
            ra=np.asarray(ra, dtype=np.float64),
            dec=np.asarray(dec, dtype=np.float64),
            dist=np.asarray(dist_mean, dtype=np.float64),
            dcang=np.asarray(dcang, dtype=np.float64),
            pdf_repr='quantile_chi',
            pdf_k=int(qchi.shape[0]),
            pdf_qchi_lib=qchi,
            pdf_qlo_lib=qlo,
            pdf_qhi_lib=qhi,
            pdf_idx=pdf_idx,
            weights=np.asarray(weights, dtype=np.float32),
            x=np.asarray(x, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            z=np.asarray(z, dtype=np.float64),
            sk=np.asarray(sk),
            ll=np.asarray(ll),
            wunit=bool(np.allclose(weights, 1.0)),
            sbound=tuple(float(v) for v in sbound),
            mxh1=int(mxh1),
            mxh2=int(mxh2),
            mxh3=int(mxh3),
            region_id=None if region_id is None else np.asarray(region_id, dtype=np.int32),
            grid_meta={**({} if grid_meta is None else dict(grid_meta)),
                       "qchi_prepare_compress_time_s": float(_qchi_compress_time),
                       "qchi_storage_dtype": _qchi_dtype,
                       "qchi_library_nbytes": _qchi_bytes,
                       "qchi_nquant": int(qchi.shape[0]),
                       "qchi_nlib": int(qchi.shape[1]),
                       "qchi_span_min": float(np.min(_qchi_span)) if _qchi_span.size else 0.0,
                       "qchi_span_median": float(np.median(_qchi_span)) if _qchi_span.size else 0.0,
                       "qchi_span_max": float(np.max(_qchi_span)) if _qchi_span.size else 0.0},
            nrows=int(len(ra)),
        )

    if kind == 'grid_chi_exact':
        chi_grid, prob = _read_pdf_grid_exact(table, pdf, config=config, pdf_source=getattr(config, "pdf_source", None), source=pdf_source, label=pdf_label)
        dist_mean, lo_idx, hi_idx, _chi_lo, _chi_hi, _halfspan, dcang = _grid_exact_support(prob, chi_grid, prob_floor=float(getattr(pdf, 'prob_floor', 0.0)))
        sidx = _sort_index_3d(ra, dec, dist_mean, sbound=sbound, mxh1=mxh1, mxh2=mxh2, mxh3=mxh3, pxorder=config.grid.pxorder)
        ra, dec = ra[sidx], dec[sidx]
        dist_mean = dist_mean[sidx]
        weights = weights[sidx]
        dcang = dcang[sidx]
        prob = np.asarray(prob[sidx, :].T, dtype=np.float64, order='F')
        cdf = np.asarray(np.cumsum(prob, axis=0), dtype=np.float64, order='F')
        lo_idx = np.asarray(lo_idx[sidx], dtype=np.int32)
        hi_idx = np.asarray(hi_idx[sidx], dtype=np.int32)
        if region_id is not None:
            region_id = np.asarray(region_id, dtype=np.int32)[sidx]
        sk, ll = _build_skll3d(mxh1, mxh2, mxh3, ra, dec, dist_mean, np.asarray(sbound, dtype=np.float64), np.asarray(pi_edges_search, dtype=np.float64))
        x, y, z = radec2xyz(np.deg2rad(ra), np.deg2rad(dec))
        pdf_idx = (np.arange(len(ra), dtype=np.int32) + 1)
        return PreparedProjectedSample(
            table=None,
            ra=np.asarray(ra, dtype=np.float64),
            dec=np.asarray(dec, dtype=np.float64),
            dist=np.asarray(dist_mean, dtype=np.float64),
            dcang=np.asarray(dcang, dtype=np.float64),
            pdf_repr='grid_chi_exact',
            pdf_k=0,
            pdf_prob_lib=prob,
            pdf_cdf_lib=cdf,
            pdf_grid=np.asarray(chi_grid, dtype=np.float64),
            pdf_lo_idx=lo_idx,
            pdf_hi_idx=hi_idx,
            pdf_idx=pdf_idx,
            weights=np.asarray(weights, dtype=np.float32),
            x=np.asarray(x, dtype=np.float64),
            y=np.asarray(y, dtype=np.float64),
            z=np.asarray(z, dtype=np.float64),
            sk=np.asarray(sk),
            ll=np.asarray(ll),
            wunit=bool(np.allclose(weights, 1.0)),
            sbound=tuple(float(v) for v in sbound),
            mxh1=int(mxh1),
            mxh2=int(mxh2),
            mxh3=int(mxh3),
            region_id=None if region_id is None else np.asarray(region_id, dtype=np.int32),
            grid_meta={} if grid_meta is None else dict(grid_meta),
            nrows=int(len(ra)),
        )

    raise ValueError(f"Unsupported pdf.kind={pdf.kind!r}; expected 'gmm_chi', 'grid_chi_exact', or 'quantile_chi'.")

def _prepare_sample_pdf_random_inherit(table, columns: ProjectedCatalogColumns, config, *, sbound, mxh1: int, mxh2: int, mxh3: int,
                                       pi_edges_search, data_lib: PreparedProjectedSample, pdf_idx_unsorted: np.ndarray,
                                       region_id=None, grid_meta=None):
    """Prepare a random sample by inheriting PDFs from an associated data library.

    Parameters
    ----------
    pdf_idx_unsorted : np.ndarray
        Integer array (0-based) indexing the *unsorted* data rows used during the
        initial sampling step. This function remaps those indices into the sorted
        data PDF library.
    """
    ra = _col(table, columns.ra)
    dec = _col(table, columns.dec)

    # Map unsorted data indices -> sorted library indices.
    # data_lib.pdf_* are in sorted order, with library index = position + 1.
    nlib = int(data_lib.nrows)
    # Reconstruct inverse permutation is not possible here; instead the caller
    # provides pdf_idx already mapped to sorted indices (1-based). To keep v1
    # simple, treat pdf_idx_unsorted as already-mapped 1-based indices if it
    # is int32 and min>=1.
    pdf_idx = np.asarray(pdf_idx_unsorted, dtype=np.int32)
    if pdf_idx.size:
        if int(np.min(pdf_idx)) >= 1:
            mapped = pdf_idx
        else:
            raise ValueError('Internal error: pdf_idx for randoms must be mapped to 1-based library indices.')
    else:
        mapped = pdf_idx

    dist = data_lib.dist[mapped - 1] if mapped.size else np.empty(0, dtype=np.float64)
    dcang = data_lib.dcang[mapped - 1] if mapped.size and data_lib.dcang is not None else None

    sidx = _sort_index_3d(ra, dec, dist, sbound=sbound, mxh1=mxh1, mxh2=mxh2, mxh3=mxh3, pxorder=config.grid.pxorder)
    ra, dec, dist = ra[sidx], dec[sidx], dist[sidx]
    mapped = np.asarray(mapped[sidx], dtype=np.int32)
    if dcang is not None:
        dcang = np.asarray(dcang[sidx], dtype=np.float64)
    if region_id is not None:
        region_id = np.asarray(region_id, dtype=np.int32)[sidx]

    sk, ll = _build_skll3d(mxh1, mxh2, mxh3, ra, dec, dist, np.asarray(sbound, dtype=np.float64), np.asarray(pi_edges_search, dtype=np.float64))
    x, y, z = radec2xyz(np.deg2rad(ra), np.deg2rad(dec))

    weights = np.ones(len(ra), dtype=np.float32)
    _inherit_grid_meta = {} if grid_meta is None else dict(grid_meta)
    if str(getattr(data_lib, 'pdf_repr', 'none')).strip().lower() == 'quantile_chi':
        _inherit_grid_meta.update({
            "qchi_random_inherits_library": True,
            "qchi_nquant": int(getattr(data_lib, 'pdf_k', 0) or 0),
            "qchi_nlib": int(getattr(data_lib, 'nrows', 0) or 0),
            "qchi_storage_dtype": str(np.asarray(getattr(data_lib, 'pdf_qchi_lib', np.empty(0))).dtype),
        })

    return PreparedProjectedSample(
        table=None,
        ra=np.asarray(ra, dtype=np.float64),
        dec=np.asarray(dec, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64),
        dcang=None if dcang is None else np.asarray(dcang, dtype=np.float64),
        pdf_repr=str(getattr(data_lib, 'pdf_repr', 'none')),
        pdf_k=int(data_lib.pdf_k),
        pdf_alpha_lib=data_lib.pdf_alpha_lib,
        pdf_mu_lib=data_lib.pdf_mu_lib,
        pdf_sig_lib=data_lib.pdf_sig_lib,
        pdf_prob_lib=getattr(data_lib, 'pdf_prob_lib', None),
        pdf_cdf_lib=getattr(data_lib, 'pdf_cdf_lib', None),
        pdf_grid=getattr(data_lib, 'pdf_grid', None),
        pdf_lo_idx=getattr(data_lib, 'pdf_lo_idx', None),
        pdf_hi_idx=getattr(data_lib, 'pdf_hi_idx', None),
        pdf_qchi_lib=getattr(data_lib, 'pdf_qchi_lib', None),
        pdf_qlo_lib=getattr(data_lib, 'pdf_qlo_lib', None),
        pdf_qhi_lib=getattr(data_lib, 'pdf_qhi_lib', None),
        pdf_idx=np.asarray(mapped, dtype=np.int32),
        weights=weights,
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        z=np.asarray(z, dtype=np.float64),
        sk=np.asarray(sk),
        ll=np.asarray(ll),
        wunit=True,
        sbound=tuple(float(v) for v in sbound),
        mxh1=int(mxh1),
        mxh2=int(mxh2),
        mxh3=int(mxh3),
        region_id=None if region_id is None else np.asarray(region_id, dtype=np.int32),
        grid_meta=_inherit_grid_meta,
        nrows=int(len(ra)),
    )

def prepare_projected_auto(data, random, config: ProjectedAutoConfig):
    """
    Prepare projected auto.
    
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
    rp_edges, (rp_edges2, rp_centers, rp_delta) = makebins(config.binning.nsepp, config.binning.seppmin, config.binning.dsepp, config.binning.logsepp)
    pi_edges, (pi_edges2, pi_centers, pi_delta) = makebins(config.binning.nsepv, 0.0, config.binning.dsepv, False)

    if bool(getattr(config, 'pdf', None) and config.pdf.enabled):
        _validate_pdf_source_config(config, cross=False)
        pdf_kind = str(getattr(config.pdf, 'kind', 'gmm_chi')).strip().lower()
        if str(config.pdf.random_pdf_policy).lower() != 'inherit':
            raise ValueError("Only pdf.random_pdf_policy='inherit' is supported in v1.")
        jk_meta = {"jk_enabled": bool(config.jackknife.enabled), "jk_region_source": None, "jk_nregions": None, "jk_geometry_from": None}
        data_region = random_region = None
        if config.jackknife.enabled:
            dreg = config.columns_data.region
            rreg = config.columns_random.region
            if dreg is not None and rreg is not None:
                data_region, random_region = _shared_user_region_ids([(data, dreg), (random, rreg)])
                jk_meta.update({"jk_region_source": "user", "jk_nregions": int(np.max(np.concatenate([data_region, random_region])) + 1)})
            elif dreg is not None or rreg is not None:
                raise ValueError("For projected jackknife auto-correlations, either both data/random region columns must be supplied or neither.")
            else:
                nregions = config.jackknife.nregions
                if nregions is None:
                    nregions = choose_default_nregions(config.binning.nsepp)
                assignments, _centers = _auto_region_ids([
                    (data, config.columns_data.ra, config.columns_data.dec, "data"),
                    (random, config.columns_random.ra, config.columns_random.dec, "random"),
                ], nregions=min(int(nregions), max(1, catalog_nrows(data), catalog_nrows(random))), seed=config.jackknife.seed, geometry_from=config.jackknife.geometry_from)
                data_region, random_region = assignments
                jk_meta.update({"jk_region_source": "auto", "jk_nregions": int(np.max(np.concatenate([data_region, random_region])) + 1), "jk_geometry_from": config.jackknife.geometry_from})

        data_source = config.pdf_source.pdf_data if bool(getattr(config.pdf_source, 'enabled', False)) else None
        data_summary = _pdf_summary_from_source(data, config, source=data_source, label="projected auto data pdf")
        dist_d = np.asarray(data_summary["dist"], dtype=np.float64)

        ra_r = _col(random, config.columns_random.ra)
        dec_r = _col(random, config.columns_random.dec)
        dec_d = _col(data, config.columns_data.dec)

        rng = np.random.default_rng(int(config.pdf.seed))
        pick = rng.integers(0, max(1, len(dist_d)), size=len(ra_r), dtype=np.int64) if len(ra_r) else np.empty(0, dtype=np.int64)
        dist_r = dist_d[pick] if pick.size else np.empty(0, dtype=np.float64)

        sbound = bound3d([dec_d, dec_r], [dist_d, dist_r])

        pi_max = float(pi_edges[-1])
        max_halfspan = float(np.max(data_summary["halfspan"])) if np.size(data_summary["halfspan"]) else 0.0
        pi_search = float(pi_max + float(config.pdf.rv_search_nsigma) * (2.0 * max_halfspan))
        pi_edges_search = np.asarray([0.0, pi_search], dtype=np.float64)

        d_mxh1, d_mxh2, d_mxh3, dens = best_skgrid_3d_legacy(len(dist_d), _col(data, config.columns_data.ra), sbound=sbound, nsepv=1, dsepv=pi_search, dens=config.grid.dens)
        r_mxh1, r_mxh2, r_mxh3, _dens2 = best_skgrid_3d_legacy(len(dist_r), ra_r, sbound=sbound, nsepv=1, dsepv=pi_search, dens=config.grid.dens)

        sidx_d = _sort_index_3d(_col(data, config.columns_data.ra), dec_d, dist_d, sbound=sbound, mxh1=d_mxh1, mxh2=d_mxh2, mxh3=d_mxh3, pxorder=config.grid.pxorder)
        inv = np.empty(len(dist_d), dtype=np.int64)
        inv[sidx_d] = np.arange(len(dist_d), dtype=np.int64)
        pdf_idx_random = (inv[pick] + 1).astype(np.int32) if pick.size else np.empty(0, dtype=np.int32)

        data_p = _prepare_sample_pdf_data(
            data,
            config.columns_data,
            config,
            sbound=sbound,
            mxh1=d_mxh1,
            mxh2=d_mxh2,
            mxh3=d_mxh3,
            pi_edges_search=pi_edges_search,
            use_weights=config.weights.weight_mode != 'unweighted',
            region_id=data_region,
            grid_meta={"autogrid": bool(config.grid.autogrid), "dens": dens, "pxorder": config.grid.pxorder,
                       "nsepv": int(config.binning.nsepv), "dsepv": float(config.binning.dsepv),
                       "pi_search": pi_search, "pi_edges_search": pi_edges_search, "prob_floor": float(config.pdf.prob_floor),
                       "pdf_repr": str(config.pdf.kind).strip().lower()},
            pdf_source=data_source,
            pdf_label="projected auto data pdf",
        )

        rand_p = _prepare_sample_pdf_random_inherit(
            random,
            config.columns_random,
            config,
            sbound=sbound,
            mxh1=r_mxh1,
            mxh2=r_mxh2,
            mxh3=r_mxh3,
            pi_edges_search=pi_edges_search,
            data_lib=data_p,
            pdf_idx_unsorted=pdf_idx_random,
            region_id=random_region,
            grid_meta={"autogrid": bool(config.grid.autogrid), "dens": dens, "pxorder": config.grid.pxorder,
                       "nsepv": int(config.binning.nsepv), "dsepv": float(config.binning.dsepv),
                       "pi_search": pi_search, "pi_edges_search": pi_edges_search, "prob_floor": float(config.pdf.prob_floor),
                       "pdf_repr": str(config.pdf.kind).strip().lower()},
        )

        meta = {
            "rp_edges": np.asarray(rp_edges, dtype=np.float64),
            "rp_centers": np.asarray(rp_centers, dtype=np.float64),
            "rp_delta": np.asarray(rp_delta, dtype=np.float64),
            "pi_edges": np.asarray(pi_edges, dtype=np.float64),
            "pi_centers": np.asarray(pi_centers, dtype=np.float64),
            "pi_delta": np.asarray(pi_delta, dtype=np.float64),
            "pi_search": pi_search,
            "sbound": tuple(float(v) for v in sbound),
            "pdf_enabled": True,
            "pdf_kind": str(config.pdf.kind),
            "pdf_k": int(config.pdf.k),
            "pdf_nquant": int(getattr(config.pdf, "nquant", 0)),
            "pdf_input_mode": "empirical_grid" if bool(getattr(config.pdf_source, "enabled", False)) else "precomputed_gmm",
            "pdf_compressor": (str(config.pdf_source.compressor) if bool(getattr(config.pdf_source, "enabled", False)) else None),
            "pdf_gmm_edge_moments": (bool(getattr(config.pdf_source, "edge_moments", True)) and str(getattr(config.pdf_source, "grid_kind", "centers")).strip().lower() == "edges") if bool(getattr(config.pdf_source, "enabled", False)) and str(getattr(config.pdf, "kind", "gmm_chi")).strip().lower() == "gmm_chi" else None,
            "pdf_exact_edge_refine": (int(getattr(config.pdf_source, "edge_refine", 1)) if bool(getattr(config.pdf_source, "enabled", False)) and str(getattr(config.pdf, "kind", "gmm_chi")).strip().lower() == "grid_chi_exact" and str(getattr(config.pdf_source, "grid_kind", "centers")).strip().lower() == "edges" else None),
            "pdf_multi_pi": bool(str(getattr(config.pdf, "kind", "gmm_chi")).strip().lower() in {"gmm_chi", "grid_chi_exact", "quantile_chi"} and int(config.binning.nsepv) > 1),
        }
        meta.update(jk_meta)
        return data_p, rand_p, meta

    data_dist = _distance_array(data, config.columns_data, config)
    rand_dist = _distance_array(random, config.columns_random, config)
    sbound = bound3d([_col(data, config.columns_data.dec), _col(random, config.columns_random.dec)], [data_dist, rand_dist])
    d_mxh1, d_mxh2, d_mxh3 = _grid_for_sample(data, config.columns_data, config, sbound=sbound)
    r_mxh1, r_mxh2, r_mxh3 = _grid_for_sample(random, config.columns_random, config, sbound=sbound)
    data_region = random_region = None
    jk_meta = {"jk_enabled": bool(config.jackknife.enabled), "jk_region_source": None, "jk_nregions": None, "jk_geometry_from": None}
    if config.jackknife.enabled:
        dreg = config.columns_data.region
        rreg = config.columns_random.region
        if dreg is not None and rreg is not None:
            data_region, random_region = _shared_user_region_ids([(data, dreg), (random, rreg)])
            jk_meta.update({"jk_region_source": "user", "jk_nregions": int(np.max(np.concatenate([data_region, random_region])) + 1)})
        elif dreg is not None or rreg is not None:
            raise ValueError("For projected jackknife auto-correlations, either both data/random region columns must be supplied or neither.")
        else:
            nregions = config.jackknife.nregions
            if nregions is None:
                nregions = choose_default_nregions(config.binning.nsepp)
            assignments, _centers = _auto_region_ids([
                (data, config.columns_data.ra, config.columns_data.dec, "data"),
                (random, config.columns_random.ra, config.columns_random.dec, "random"),
            ], nregions=min(int(nregions), max(1, catalog_nrows(data), catalog_nrows(random))), seed=config.jackknife.seed, geometry_from=config.jackknife.geometry_from)
            data_region, random_region = assignments
            jk_meta.update({"jk_region_source": "auto", "jk_nregions": int(np.max(np.concatenate([data_region, random_region])) + 1), "jk_geometry_from": config.jackknife.geometry_from})
    data_p = _prepare_sample(data, config.columns_data, config, sbound=sbound, mxh1=d_mxh1, mxh2=d_mxh2, mxh3=d_mxh3, pi_edges=pi_edges, use_weights=config.weights.weight_mode != "unweighted", region_id=data_region, grid_meta={"autogrid": bool(config.grid.autogrid), "dens": config.grid.dens, "pxorder": config.grid.pxorder, "nsepv": config.binning.nsepv, "dsepv": config.binning.dsepv})
    rand_p = _prepare_sample(random, config.columns_random, config, sbound=sbound, mxh1=r_mxh1, mxh2=r_mxh2, mxh3=r_mxh3, pi_edges=pi_edges, use_weights=False, region_id=random_region, grid_meta={"autogrid": bool(config.grid.autogrid), "dens": config.grid.dens, "pxorder": config.grid.pxorder, "nsepv": config.binning.nsepv, "dsepv": config.binning.dsepv})
    meta = {
        "rp_edges": np.asarray(rp_edges, dtype=np.float64),
        "rp_centers": np.asarray(rp_centers, dtype=np.float64),
        "rp_delta": np.asarray(rp_delta, dtype=np.float64),
        "pi_edges": np.asarray(pi_edges, dtype=np.float64),
        "pi_centers": np.asarray(pi_centers, dtype=np.float64),
        "pi_delta": np.asarray(pi_delta, dtype=np.float64),
        "sbound": tuple(float(v) for v in sbound),
    }
    meta.update(jk_meta)
    return data_p, rand_p, meta


def prepare_projected_cross(data1, random1, data2, random2, config: ProjectedCrossConfig):
    """Prepare projected cross-correlation inputs for the compiled counters."""
    rp_edges, (rp_edges2, rp_centers, rp_delta) = makebins(config.binning.nsepp, config.binning.seppmin, config.binning.dsepp, config.binning.logsepp)
    pi_edges, (pi_edges2, pi_centers, pi_delta) = makebins(config.binning.nsepv, 0.0, config.binning.dsepv, False)

    if bool(getattr(config, 'pdf', None) and config.pdf.enabled):
        _validate_pdf_source_config(config, cross=True)
        pdf_kind = str(getattr(config.pdf, 'kind', 'gmm_chi')).strip().lower()
        if str(config.pdf.random_pdf_policy).lower() != 'inherit':
            raise ValueError("Only pdf.random_pdf_policy='inherit' is supported in v1.")
        jk_meta = {"jk_enabled": bool(config.jackknife.enabled), "jk_region_source": None, "jk_nregions": None, "jk_geometry_from": None}
        d1_region = r1_region = d2_region = r2_region = None
        if config.jackknife.enabled:
            provided = [
                config.columns_data1.region is not None,
                config.columns_data2.region is not None,
                (random1 is not None and config.columns_random1.region is not None) or (random1 is None),
                (random2 is not None and config.columns_random2.region is not None) or (random2 is None),
            ]
            user_pairs = []
            if config.columns_data1.region is not None:
                user_pairs.append((data1, config.columns_data1.region))
            if random1 is not None and config.columns_random1.region is not None:
                user_pairs.append((random1, config.columns_random1.region))
            if config.columns_data2.region is not None:
                user_pairs.append((data2, config.columns_data2.region))
            if random2 is not None and config.columns_random2.region is not None:
                user_pairs.append((random2, config.columns_random2.region))
            if user_pairs and len(user_pairs) != (2 + (1 if random1 is not None else 0) + (1 if random2 is not None else 0)):
                raise ValueError("For projected jackknife cross-correlations, region columns must be supplied for all participating catalogs or for none of them.")
            if user_pairs:
                regs = _shared_user_region_ids(user_pairs)
                pos = 0
                d1_region = regs[pos]; pos += 1
                if random1 is not None:
                    r1_region = regs[pos]; pos += 1
                d2_region = regs[pos]; pos += 1
                if random2 is not None:
                    r2_region = regs[pos]
                all_regs = np.concatenate([arr for arr in [d1_region, r1_region, d2_region, r2_region] if arr is not None])
                jk_meta.update({"jk_region_source": "user", "jk_nregions": int(np.max(all_regs) + 1) if all_regs.size else 0})
            else:
                nregions = config.jackknife.nregions
                if nregions is None:
                    nregions = choose_default_nregions(config.binning.nsepp)
                catalogs = [
                    (data1, config.columns_data1.ra, config.columns_data1.dec, "data"),
                    (data2, config.columns_data2.ra, config.columns_data2.dec, "data"),
                ]
                if random1 is not None:
                    catalogs.append((random1, config.columns_random1.ra, config.columns_random1.dec, "random"))
                if random2 is not None:
                    catalogs.append((random2, config.columns_random2.ra, config.columns_random2.dec, "random"))
                assignments, _centers = _auto_region_ids(catalogs, nregions=min(int(nregions), max(1, sum(catalog_nrows(cat[0]) for cat in catalogs))), seed=config.jackknife.seed, geometry_from=config.jackknife.geometry_from)
                pos = 0
                d1_region = assignments[pos]; pos += 1
                d2_region = assignments[pos]; pos += 1
                if random1 is not None:
                    r1_region = assignments[pos]; pos += 1
                if random2 is not None:
                    r2_region = assignments[pos]
                all_regs = np.concatenate([arr for arr in [d1_region, r1_region, d2_region, r2_region] if arr is not None])
                jk_meta.update({"jk_region_source": "auto", "jk_nregions": int(np.max(all_regs) + 1) if all_regs.size else 0, "jk_geometry_from": config.jackknife.geometry_from})

        data1_source = config.pdf_source.pdf_data1 if bool(getattr(config.pdf_source, 'enabled', False)) else None
        data2_source = config.pdf_source.pdf_data2 if bool(getattr(config.pdf_source, 'enabled', False)) else None
        summary1 = _pdf_summary_from_source(data1, config, source=data1_source, label="projected cross data1 pdf")
        summary2 = _pdf_summary_from_source(data2, config, source=data2_source, label="projected cross data2 pdf")
        dist1 = np.asarray(summary1["dist"], dtype=np.float64)
        dist2 = np.asarray(summary2["dist"], dtype=np.float64)

        rng = np.random.default_rng(int(config.pdf.seed))
        ra_r1 = dec_r1 = None
        pick1 = dist_r1 = None
        if random1 is not None:
            ra_r1 = _col(random1, config.columns_random1.ra)
            dec_r1 = _col(random1, config.columns_random1.dec)
            pick1 = rng.integers(0, max(1, len(dist1)), size=len(ra_r1), dtype=np.int64) if len(ra_r1) else np.empty(0, dtype=np.int64)
            dist_r1 = dist1[pick1] if pick1.size else np.empty(0, dtype=np.float64)
        ra_r2 = dec_r2 = None
        pick2 = dist_r2 = None
        if random2 is not None:
            ra_r2 = _col(random2, config.columns_random2.ra)
            dec_r2 = _col(random2, config.columns_random2.dec)
            pick2 = rng.integers(0, max(1, len(dist2)), size=len(ra_r2), dtype=np.int64) if len(ra_r2) else np.empty(0, dtype=np.int64)
            dist_r2 = dist2[pick2] if pick2.size else np.empty(0, dtype=np.float64)

        dec_arrays = [_col(data1, config.columns_data1.dec), _col(data2, config.columns_data2.dec)]
        dist_arrays = [dist1, dist2]
        if random1 is not None:
            dec_arrays.append(dec_r1); dist_arrays.append(dist_r1)
        if random2 is not None:
            dec_arrays.append(dec_r2); dist_arrays.append(dist_r2)
        sbound = bound3d(dec_arrays, dist_arrays)

        pi_max = float(pi_edges[-1])
        max_halfspan1 = float(np.max(summary1["halfspan"])) if np.size(summary1["halfspan"]) else 0.0
        max_halfspan2 = float(np.max(summary2["halfspan"])) if np.size(summary2["halfspan"]) else 0.0
        pi_search = float(pi_max + float(config.pdf.rv_search_nsigma) * (max_halfspan1 + max_halfspan2))
        pi_edges_search = np.asarray([0.0, pi_search], dtype=np.float64)

        d1_grid = best_skgrid_3d_legacy(len(dist1), _col(data1, config.columns_data1.ra), sbound=sbound, nsepv=1, dsepv=pi_search, dens=config.grid.dens)[:4]
        d2_grid = best_skgrid_3d_legacy(len(dist2), _col(data2, config.columns_data2.ra), sbound=sbound, nsepv=1, dsepv=pi_search, dens=config.grid.dens)[:4]
        r1_grid = best_skgrid_3d_legacy(len(dist_r1) if dist_r1 is not None else 0, ra_r1 if ra_r1 is not None else np.empty(0), sbound=sbound, nsepv=1, dsepv=pi_search, dens=config.grid.dens)[:4] if random1 is not None else None
        r2_grid = best_skgrid_3d_legacy(len(dist_r2) if dist_r2 is not None else 0, ra_r2 if ra_r2 is not None else np.empty(0), sbound=sbound, nsepv=1, dsepv=pi_search, dens=config.grid.dens)[:4] if random2 is not None else None

        sidx1 = _sort_index_3d(_col(data1, config.columns_data1.ra), _col(data1, config.columns_data1.dec), dist1, sbound=sbound, mxh1=d1_grid[0], mxh2=d1_grid[1], mxh3=d1_grid[2], pxorder=config.grid.pxorder)
        inv1 = np.empty(len(dist1), dtype=np.int64); inv1[sidx1] = np.arange(len(dist1), dtype=np.int64)
        pdf_idx_r1 = (inv1[pick1] + 1).astype(np.int32) if (random1 is not None and pick1 is not None and pick1.size) else np.empty(0, dtype=np.int32)
        sidx2 = _sort_index_3d(_col(data2, config.columns_data2.ra), _col(data2, config.columns_data2.dec), dist2, sbound=sbound, mxh1=d2_grid[0], mxh2=d2_grid[1], mxh3=d2_grid[2], pxorder=config.grid.pxorder)
        inv2 = np.empty(len(dist2), dtype=np.int64); inv2[sidx2] = np.arange(len(dist2), dtype=np.int64)
        pdf_idx_r2 = (inv2[pick2] + 1).astype(np.int32) if (random2 is not None and pick2 is not None and pick2.size) else np.empty(0, dtype=np.int32)

        prep1 = _prepare_sample_pdf_data(
            data1, config.columns_data1, config,
            sbound=sbound, mxh1=d1_grid[0], mxh2=d1_grid[1], mxh3=d1_grid[2],
            pi_edges_search=pi_edges_search,
            use_weights=config.weights.weight_mode != 'unweighted',
            region_id=d1_region,
            grid_meta={"autogrid": bool(config.grid.autogrid), "dens": config.grid.dens, "pxorder": config.grid.pxorder,
                       "nsepv": int(config.binning.nsepv), "dsepv": float(config.binning.dsepv),
                       "pi_search": pi_search, "pi_edges_search": pi_edges_search, "prob_floor": float(config.pdf.prob_floor),
                       "pdf_repr": str(config.pdf.kind).strip().lower()},
            pdf_source=data1_source,
            pdf_label="projected cross data1 pdf",
        )
        prep2 = _prepare_sample_pdf_data(
            data2, config.columns_data2, config,
            sbound=sbound, mxh1=d2_grid[0], mxh2=d2_grid[1], mxh3=d2_grid[2],
            pi_edges_search=pi_edges_search,
            use_weights=config.weights.weight_mode != 'unweighted',
            region_id=d2_region,
            grid_meta={"autogrid": bool(config.grid.autogrid), "dens": config.grid.dens, "pxorder": config.grid.pxorder,
                       "nsepv": int(config.binning.nsepv), "dsepv": float(config.binning.dsepv),
                       "pi_search": pi_search, "pi_edges_search": pi_edges_search, "prob_floor": float(config.pdf.prob_floor),
                       "pdf_repr": str(config.pdf.kind).strip().lower()},
            pdf_source=data2_source,
            pdf_label="projected cross data2 pdf",
        )

        prep_r1 = None
        if random1 is not None:
            prep_r1 = _prepare_sample_pdf_random_inherit(
                random1, config.columns_random1, config,
                sbound=sbound, mxh1=r1_grid[0], mxh2=r1_grid[1], mxh3=r1_grid[2],
                pi_edges_search=pi_edges_search,
                data_lib=prep1,
                pdf_idx_unsorted=pdf_idx_r1,
                region_id=r1_region,
                grid_meta={"autogrid": bool(config.grid.autogrid), "dens": config.grid.dens, "pxorder": config.grid.pxorder,
                           "nsepv": int(config.binning.nsepv), "dsepv": float(config.binning.dsepv),
                           "pi_search": pi_search, "pi_edges_search": pi_edges_search, "prob_floor": float(config.pdf.prob_floor),
                           "pdf_repr": str(config.pdf.kind).strip().lower()},
            )
        prep_r2 = None
        if random2 is not None:
            prep_r2 = _prepare_sample_pdf_random_inherit(
                random2, config.columns_random2, config,
                sbound=sbound, mxh1=r2_grid[0], mxh2=r2_grid[1], mxh3=r2_grid[2],
                pi_edges_search=pi_edges_search,
                data_lib=prep2,
                pdf_idx_unsorted=pdf_idx_r2,
                region_id=r2_region,
                grid_meta={"autogrid": bool(config.grid.autogrid), "dens": config.grid.dens, "pxorder": config.grid.pxorder,
                           "nsepv": int(config.binning.nsepv), "dsepv": float(config.binning.dsepv),
                           "pi_search": pi_search, "pi_edges_search": pi_edges_search, "prob_floor": float(config.pdf.prob_floor),
                           "pdf_repr": str(config.pdf.kind).strip().lower()},
            )

        meta = {
            "rp_edges": np.asarray(rp_edges, dtype=np.float64),
            "rp_centers": np.asarray(rp_centers, dtype=np.float64),
            "rp_delta": np.asarray(rp_delta, dtype=np.float64),
            "pi_edges": np.asarray(pi_edges, dtype=np.float64),
            "pi_centers": np.asarray(pi_centers, dtype=np.float64),
            "pi_delta": np.asarray(pi_delta, dtype=np.float64),
            "pi_search": pi_search,
            "sbound": tuple(float(v) for v in sbound),
            "pdf_enabled": True,
            "pdf_kind": str(config.pdf.kind),
            "pdf_k": int(config.pdf.k),
            "pdf_nquant": int(getattr(config.pdf, "nquant", 0)),
            "pdf_input_mode": "empirical_grid" if bool(getattr(config.pdf_source, "enabled", False)) else "precomputed_gmm",
            "pdf_compressor": (str(config.pdf_source.compressor) if bool(getattr(config.pdf_source, "enabled", False)) else None),
            "pdf_gmm_edge_moments": (bool(getattr(config.pdf_source, "edge_moments", True)) and str(getattr(config.pdf_source, "grid_kind", "centers")).strip().lower() == "edges") if bool(getattr(config.pdf_source, "enabled", False)) and str(getattr(config.pdf, "kind", "gmm_chi")).strip().lower() == "gmm_chi" else None,
            "pdf_exact_edge_refine": (int(getattr(config.pdf_source, "edge_refine", 1)) if bool(getattr(config.pdf_source, "enabled", False)) and str(getattr(config.pdf, "kind", "gmm_chi")).strip().lower() == "grid_chi_exact" and str(getattr(config.pdf_source, "grid_kind", "centers")).strip().lower() == "edges" else None),
            "pdf_multi_pi": bool(str(getattr(config.pdf, "kind", "gmm_chi")).strip().lower() in {"gmm_chi", "grid_chi_exact", "quantile_chi"} and int(config.binning.nsepv) > 1),
        }
        meta.update(jk_meta)
        return prep1, prep_r1, prep2, prep_r2, meta

    d1_dist = _distance_array(data1, config.columns_data1, config)
    r1_dist = _distance_array(random1, config.columns_random1, config) if random1 is not None else np.empty(0, dtype=np.float64)
    d2_dist = _distance_array(data2, config.columns_data2, config)
    r2_dist = _distance_array(random2, config.columns_random2, config) if random2 is not None else np.empty(0, dtype=np.float64)
    dec_arrays = [_col(data1, config.columns_data1.dec), _col(data2, config.columns_data2.dec)]
    dist_arrays = [d1_dist, d2_dist]
    if random1 is not None:
        dec_arrays.append(_col(random1, config.columns_random1.dec)); dist_arrays.append(r1_dist)
    if random2 is not None:
        dec_arrays.append(_col(random2, config.columns_random2.dec)); dist_arrays.append(r2_dist)
    sbound = bound3d(dec_arrays, dist_arrays)
    d1_grid = _grid_for_sample(data1, config.columns_data1, config, sbound=sbound)
    r1_grid = _grid_for_sample(random1, config.columns_random1, config, sbound=sbound) if random1 is not None else None
    d2_grid = _grid_for_sample(data2, config.columns_data2, config, sbound=sbound)
    r2_grid = _grid_for_sample(random2, config.columns_random2, config, sbound=sbound) if random2 is not None else None
    reg1 = reg_r1 = reg2 = reg_r2 = None
    jk_meta = {"jk_enabled": bool(config.jackknife.enabled), "jk_region_source": None, "jk_nregions": None, "jk_geometry_from": None}
    if config.jackknife.enabled:
        provided = [
            config.columns_data1.region is not None,
            config.columns_data2.region is not None,
            (random1 is not None and config.columns_random1.region is not None) or (random1 is None),
            (random2 is not None and config.columns_random2.region is not None) or (random2 is None),
        ]
        if all(provided):
            pairs = [(data1, config.columns_data1.region)]
            if random1 is not None:
                pairs.append((random1, config.columns_random1.region))
            pairs.append((data2, config.columns_data2.region))
            if random2 is not None:
                pairs.append((random2, config.columns_random2.region))
            regs = _shared_user_region_ids(pairs)
            reg1 = regs[0]
            idx = 1
            if random1 is not None:
                reg_r1 = regs[idx]; idx += 1
            reg2 = regs[idx]; idx += 1
            if random2 is not None:
                reg_r2 = regs[idx]
            regions = [arr for arr in [reg1, reg_r1, reg2, reg_r2] if arr is not None]
            all_regs = np.concatenate(regions) if regions else np.empty(0, dtype=np.int32)
            jk_meta.update({"jk_region_source": "user", "jk_nregions": int(np.max(all_regs) + 1) if all_regs.size else 0})
        elif any(provided):
            raise ValueError("For projected jackknife cross-correlations, region columns must be supplied for all participating catalogs or for none of them.")
        else:
            nregions = config.jackknife.nregions
            if nregions is None:
                nregions = choose_default_nregions(config.binning.nsepp)
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
    prep1 = _prepare_sample(data1, config.columns_data1, config, sbound=sbound, mxh1=d1_grid[0], mxh2=d1_grid[1], mxh3=d1_grid[2], pi_edges=pi_edges, use_weights=config.weights.weight_mode != "unweighted", region_id=reg1)
    prep_r1 = None if random1 is None else _prepare_sample(random1, config.columns_random1, config, sbound=sbound, mxh1=r1_grid[0], mxh2=r1_grid[1], mxh3=r1_grid[2], pi_edges=pi_edges, use_weights=False, region_id=reg_r1)
    prep2 = _prepare_sample(data2, config.columns_data2, config, sbound=sbound, mxh1=d2_grid[0], mxh2=d2_grid[1], mxh3=d2_grid[2], pi_edges=pi_edges, use_weights=config.weights.weight_mode != "unweighted", region_id=reg2)
    prep_r2 = None if random2 is None else _prepare_sample(random2, config.columns_random2, config, sbound=sbound, mxh1=r2_grid[0], mxh2=r2_grid[1], mxh3=r2_grid[2], pi_edges=pi_edges, use_weights=False, region_id=reg_r2)
    meta = {
        "rp_edges": np.asarray(rp_edges, dtype=np.float64),
        "rp_centers": np.asarray(rp_centers, dtype=np.float64),
        "rp_delta": np.asarray(rp_delta, dtype=np.float64),
        "pi_edges": np.asarray(pi_edges, dtype=np.float64),
        "pi_centers": np.asarray(pi_centers, dtype=np.float64),
        "pi_delta": np.asarray(pi_delta, dtype=np.float64),
        "sbound": tuple(float(v) for v in sbound),
    }
    meta.update(jk_meta)
    return prep1, prep_r1, prep2, prep_r2, meta

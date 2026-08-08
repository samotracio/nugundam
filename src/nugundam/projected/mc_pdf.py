"""Monte-Carlo projected PDF resampling helpers."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..core.catalogs import catalog_column_names, catalog_get_column, catalog_has_column, catalog_nrows
from ..core.progress import create_status_emitter, in_notebook, run_with_progress
from ..core.jackknife import choose_default_nregions, jackknife_cov
from ..result_meta import attach_roundtrip_context, provenance_dict
from .estimators import (
    apply_bootstrap_storage_policy,
    compute_auto_cumulative_wp,
    compute_cross_cumulative_wp,
    estimate_auto,
    estimate_cross,
    compute_auto_xi2d,
    compute_cross_xi2d,
)
from .fortran_bridge import (
    _integrate_pi,
    build_auto_count_result,
    build_auto_counts,
    build_cross_count_result,
    build_cross_counts,
)
from .models import ProjectedAutoConfig, ProjectedAutoCounts, ProjectedCrossConfig, ProjectedCrossCounts
from .pdf_common import load_pdf_matrix as _shared_load_pdf_matrix, resolve_common_chi_grid
from .prepare import (
    _auto_region_ids,
    _col,
    _distance_array,
    _prepared_projected_from_arrays,
    _shared_user_region_ids,
    best_skgrid_3d_legacy,
    bound3d,
)

try:  # pragma: no cover - optional dependency
    from numba import njit
except Exception:  # pragma: no cover - fallback when numba is unavailable
    njit = None


if njit is not None:
    @njit(cache=True)
    def _sample_rows_from_cdf_impl(cdf: np.ndarray, u: np.ndarray) -> np.ndarray:  # pragma: no cover - exercised in runtime tests
        out = np.empty(u.shape[0], dtype=np.int64)
        ncol = cdf.shape[1]
        for i in range(u.shape[0]):
            ui = u[i]
            j = 0
            while j < ncol - 1 and ui > cdf[i, j]:
                j += 1
            out[i] = j
        return out
else:
    def _sample_rows_from_cdf_impl(cdf: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = np.empty(u.shape[0], dtype=np.int64)
        ncol = cdf.shape[1]
        for i, ui in enumerate(u):
            j = 0
            while j < ncol - 1 and ui > cdf[i, j]:
                j += 1
            out[i] = j
        return out


def _stage(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _run_mc_progress(config, target, *, status_prefix: str | None = None, status_emitter=None):
    return run_with_progress(
        bool(getattr(config.progress, "enabled", False)),
        getattr(config.progress, "progress_file", None),
        float(getattr(config.progress, "poll_interval", 0.15)),
        target,
        status_prefix=status_prefix,
        status_emitter=status_emitter,
    )


def _load_array_spec(value) -> np.ndarray:
    if value is None:
        raise ValueError("A common z_grid or chi_grid must be provided for mc_pdf mode.")
    if isinstance(value, (str, Path)):
        path = Path(value)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.asarray(np.load(path), dtype=np.float64)
        if suffix == ".npz":
            obj = np.load(path)
            key = next(iter(obj.files))
            return np.asarray(obj[key], dtype=np.float64)
        return np.asarray(np.loadtxt(path), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _zgrid_to_chi(z_grid: np.ndarray, config) -> np.ndarray:
    try:
        from astropy.cosmology import LambdaCDM
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "mc_pdf projected correlations require astropy to convert the shared z-grid to comoving distance."
        ) from exc
    cosmo = LambdaCDM(H0=config.distance.h0, Om0=config.distance.omegam, Ode0=config.distance.omegal)
    return np.asarray(cosmo.comoving_distance(np.asarray(z_grid, dtype=np.float64)).value, dtype=np.float64)


def _infer_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Infer bin edges from a strictly increasing center grid."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1:
        raise ValueError("Common PDF grids must be one-dimensional.")
    if centers.size < 2:
        raise ValueError("sample_within_bin=True with grid_kind='centers' requires at least two grid centers.")
    if np.any(~np.isfinite(centers)):
        raise ValueError("Common PDF grids must contain finite values.")
    d = np.diff(centers)
    if np.any(d <= 0.0):
        raise ValueError("Common PDF grid centers must be strictly increasing when sample_within_bin=True.")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * d[0]
    edges[-1] = centers[-1] + 0.5 * d[-1]
    return edges


def _resolve_sample_edges_chi(spec, config) -> np.ndarray | None:
    """Return chi-bin edges for continuous MC sampling, or None for center sampling."""
    if not bool(getattr(spec, "sample_within_bin", False)):
        return None
    kind = str(getattr(spec, "grid_kind", "centers")).strip().lower()
    if kind not in {"centers", "edges"}:
        raise ValueError("mc_pdf.grid_kind must be either 'centers' or 'edges'.")

    if spec.chi_grid is not None:
        grid = _load_array_spec(spec.chi_grid)
        edges = np.asarray(grid, dtype=np.float64) if kind == "edges" else _infer_edges_from_centers(grid)
    elif spec.z_grid is not None:
        grid = _load_array_spec(spec.z_grid)
        z_edges = np.asarray(grid, dtype=np.float64) if kind == "edges" else _infer_edges_from_centers(grid)
        edges = _zgrid_to_chi(z_edges, config)
    else:
        raise ValueError("A common z_grid or chi_grid must be provided for mc_pdf mode.")

    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("sample_within_bin=True requires a one-dimensional grid with at least two edges.")
    if np.any(~np.isfinite(edges)):
        raise ValueError("sample_within_bin=True requires finite grid edges.")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("sample_within_bin=True requires strictly increasing grid edges.")
    return edges


def _get_raw_column(table, name: str):
    key = str(name)
    if hasattr(table, "columns") and hasattr(table, "iloc"):
        if key not in table.columns:
            raise KeyError(key)
        return table[key]
    if hasattr(table, "colnames"):
        return table[key]
    if hasattr(table, "column_names") and hasattr(table, "num_rows"):
        return table[key]
    if isinstance(table, Mapping):
        return table[key]
    if isinstance(table, np.ndarray) and table.dtype.names is not None:
        return table[key]
    return table[key]


def _vector_to_matrix(values) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 2 and arr.dtype != object:
        return np.asarray(arr, dtype=np.float64)
    rows = []
    for item in arr:
        rows.append(np.asarray(item, dtype=np.float64))
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(rows).astype(np.float64, copy=False)


def _read_parquet_dataframe(path: str, columns: list[str] | None = None):
    try:  # pragma: no cover - depends on optional parquet stack
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise ImportError("Reading mc_pdf parquet sources requires pandas with parquet support.") from exc
    return pd.read_parquet(path, columns=columns)


def _columns_from_prefix(names: tuple[str, ...], prefix: str) -> list[str]:
    cols = [str(name) for name in names if str(name).startswith(prefix)]
    if not cols:
        raise ValueError(f"No PDF columns found with prefix {prefix!r}.")
    try:
        cols.sort(key=lambda s: int(s[len(prefix):]))
    except Exception:
        cols.sort()
    return cols


def _align_matrix_rows(matrix: np.ndarray, source_ids, catalog_ids) -> np.ndarray:
    if source_ids is None or catalog_ids is None:
        return matrix
    source_ids = np.asarray(source_ids)
    catalog_ids = np.asarray(catalog_ids)
    lookup = {val: i for i, val in enumerate(source_ids.tolist())}
    try:
        order = np.asarray([lookup[val] for val in catalog_ids.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise KeyError(f"mc_pdf row alignment failed because catalog id {exc.args[0]!r} is missing from the PDF source.") from exc
    return np.asarray(matrix[order], dtype=np.float64)


def _load_pdf_matrix(source, catalog, *, nrows: int) -> np.ndarray:
    return _shared_load_pdf_matrix(source, catalog, nrows=nrows, label="mc_pdf")


def _build_cdf(p: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(np.asarray(p, dtype=np.float64), axis=1)
    if cdf.size:
        cdf[:, -1] = 1.0
    return cdf


def _sample_indices(cdf: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if cdf.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    u = rng.random(cdf.shape[0], dtype=np.float64)
    return _sample_rows_from_cdf_impl(cdf, u)


def _sample_from_global(pbar: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    if size == 0:
        return np.empty(0, dtype=np.float64)
    cdf = np.cumsum(np.asarray(pbar, dtype=np.float64))
    cdf[-1] = 1.0
    u = rng.random(size, dtype=np.float64)
    idx = np.searchsorted(cdf, u, side="left")
    return idx.astype(np.int64)


def _support_bounds(p: np.ndarray | None, chi_grid: np.ndarray, *, floor: float) -> tuple[float, float] | None:
    if p is None:
        return None
    prof = np.asarray(np.sum(p, axis=0), dtype=np.float64)
    active = np.flatnonzero(prof > float(floor))
    if active.size == 0:
        active = np.arange(len(chi_grid), dtype=np.int64)
    return float(chi_grid[int(active[0])]), float(chi_grid[int(active[-1])])


def _catalog_template(table, columns, config, *, use_weights: bool) -> dict[str, Any]:
    ra = _col(table, columns.ra)
    dec = _col(table, columns.dec)
    if use_weights:
        if catalog_has_column(table, columns.weight):
            weights = _col(table, columns.weight, dtype=np.float64)
        else:
            if config.weights.weight_mode == "weighted":
                raise KeyError(columns.weight)
            weights = np.ones(len(ra), dtype=np.float64)
    else:
        weights = np.ones(len(ra), dtype=np.float64)
    return {
        "ra": np.asarray(ra, dtype=np.float64),
        "dec": np.asarray(dec, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
        "nrows": int(len(ra)),
        "wunit": bool(np.allclose(weights, 1.0)),
    }


def _set_template_region(template: dict[str, Any], region_id) -> dict[str, Any]:
    out = dict(template)
    out["region_id"] = None if region_id is None else np.asarray(region_id, dtype=np.int32)
    return out


def _take_template(template: dict[str, Any], index) -> dict[str, Any]:
    if index is None:
        return template
    idx = np.asarray(index, dtype=np.int64)
    out = {
        "ra": np.asarray(template["ra"], dtype=np.float64)[idx],
        "dec": np.asarray(template["dec"], dtype=np.float64)[idx],
        "weights": np.asarray(template["weights"], dtype=np.float64)[idx],
    }
    out["nrows"] = int(out["ra"].size)
    out["wunit"] = bool(np.allclose(out["weights"], 1.0))
    if "region_id" in template and template["region_id"] is not None:
        out["region_id"] = np.asarray(template["region_id"], dtype=np.int32)[idx]
    return out


def _stable_token(value) -> int:
    txt = str(value)
    acc = 2166136261
    for ch in txt:
        acc ^= ord(ch)
        acc = (acc * 16777619) & 0xFFFFFFFF
    return int(acc)


def _child_rng(base_seed: int, *parts) -> np.random.Generator:
    entropy = [int(base_seed) & 0xFFFFFFFF]
    entropy.extend(_stable_token(part) for part in parts)
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _resampling_nreal(spec) -> int:
    value = getattr(spec, "resampling_nreal", None)
    n = int(getattr(spec, "nreal", 0) if value is None else value)
    if n <= 0:
        raise ValueError("mc_pdf.resampling_nreal must be None or a positive integer.")
    return n


def _bootstrap_cov(realizations: np.ndarray) -> np.ndarray:
    arr = np.asarray(realizations, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Bootstrap realizations must have shape (n_bootstrap, n_bins).")
    if arr.shape[0] <= 1:
        return np.zeros((arr.shape[1], arr.shape[1]), dtype=np.float64)
    delta = arr - np.mean(arr, axis=0)[None, :]
    return (delta.T @ delta) / float(arr.shape[0])


def _apply_bootstrap_result(
    result,
    realizations: np.ndarray,
    *,
    cumulative_realizations: np.ndarray | None = None,
    backend: str = "mc_pdf_rerun",
):
    arr = np.asarray(realizations, dtype=np.float64)
    cov = _bootstrap_cov(arr)
    result.wp_err = np.sqrt(np.diag(cov))
    result.cov = cov
    result.bootstrap_realizations = arr

    if cumulative_realizations is not None:
        cumulative = np.asarray(cumulative_realizations, dtype=np.float64)
        if cumulative.ndim != 3:
            raise ValueError(
                "Cumulative bootstrap realizations must have shape "
                "(n_bootstrap, n_rp, n_pi)."
            )
        if cumulative.shape[:2] != arr.shape:
            raise ValueError(
                "Cumulative and fully integrated bootstrap realizations have "
                f"incompatible shapes: {cumulative.shape} and {arr.shape}."
            )
        if cumulative.shape[2] == 0:
            raise ValueError("Cumulative bootstrap realizations require at least one pi bin.")
        if not np.allclose(cumulative[:, :, -1], arr, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "The final cumulative bootstrap column does not match the "
                "fully integrated bootstrap realizations."
            )
        result.bootstrap_cumulative_realizations = cumulative

    result.metadata.update({
        "bootstrap": True,
        "bootstrap_backend": str(backend),
        "bootstrap_nrealizations": int(arr.shape[0]),
        "bootstrap_cumulative_available": cumulative_realizations is not None,
    })
    return result


def _mc_resampling_backend(spec) -> str:
    backend = str(getattr(spec, "resampling_backend", "auto")).strip().lower()
    if backend not in {"auto", "rerun", "fast"}:
        raise ValueError("mc_pdf.resampling_backend must be 'auto', 'rerun', or 'fast'.")
    return backend


def _mc_resampling_random_policy(spec) -> str:
    policy = str(getattr(spec, "resampling_random_policy", "reinherit")).strip().lower()
    if policy not in {"fixed", "reinherit"}:
        raise ValueError("mc_pdf.resampling_random_policy must be either 'fixed' or 'reinherit'.")
    return policy


def _mc_fixed_resampling_randoms(spec) -> bool:
    mode = str(getattr(spec, "random_mode", "fixed_global")).strip().lower()
    return mode != "inherit_realization" or _mc_resampling_random_policy(spec) == "fixed"


def _mc_fast_bootstrap_enabled(config) -> bool:
    if not bool(getattr(config.bootstrap, "enabled", False)) or bool(getattr(config.jackknife, "enabled", False)):
        return False
    backend = _mc_resampling_backend(config.mc_pdf)
    if backend == "rerun":
        return False
    allowed = _mc_fixed_resampling_randoms(config.mc_pdf)
    if backend == "fast" and not allowed:
        raise NotImplementedError(
            "mc_pdf.resampling_backend='fast' requires fixed random treatment. "
            "For random_mode='inherit_realization', set mc_pdf.resampling_random_policy='fixed' "
            "or use resampling_backend='rerun'."
        )
    return bool(allowed)


def _mc_fast_jackknife_enabled(config) -> bool:
    if not bool(getattr(config.jackknife, "enabled", False)):
        return False
    backend = _mc_resampling_backend(config.mc_pdf)
    if backend == "rerun":
        return False
    if bool(getattr(config.bootstrap, "enabled", False)):
        if backend == "fast":
            raise NotImplementedError(
                "mc_pdf.resampling_backend='fast' currently supports either bootstrap or jackknife, "
                "not both at the same time. Disable bootstrap for fast MC jackknife, or use "
                "resampling_backend='rerun'."
            )
        return False
    allowed = _mc_fixed_resampling_randoms(config.mc_pdf)
    if backend == "fast" and not allowed:
        raise NotImplementedError(
            "mc_pdf.resampling_backend='fast' jackknife requires fixed random treatment. "
            "For random_mode='inherit_realization', set mc_pdf.resampling_random_policy='fixed' "
            "or use resampling_backend='rerun'."
        )
    return bool(allowed)


def _bootstrap_products_from_auto_counts(
    counts: ProjectedAutoCounts,
    *,
    estimator: str,
    data_weights: np.ndarray | None = None,
    store_cumulative: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return integrated and optionally cumulative auto-bootstrap realizations."""
    tmp = estimate_auto(
        counts,
        estimator=estimator,
        data_weights=data_weights,
        store_bootstrap_cumulative=store_cumulative,
    )
    arr = getattr(tmp, "bootstrap_realizations", None)
    cumulative = getattr(tmp, "bootstrap_cumulative_realizations", None)
    if arr is not None:
        return (
            np.asarray(arr, dtype=np.float64),
            None if cumulative is None else np.asarray(cumulative, dtype=np.float64),
        )

    if counts.dd_boot is None or counts.dd_boot.size == 0:
        empty_wp = np.zeros((0, len(counts.rp_centers)), dtype=np.float64)
        empty_cumulative = None
        if store_cumulative:
            empty_cumulative = np.zeros(
                (0, len(counts.rp_centers), len(counts.pi_centers)),
                dtype=np.float64,
            )
        return empty_wp, empty_cumulative

    curves = []
    cumulative_curves = []
    for ib in range(counts.dd_boot.shape[2]):
        c = ProjectedAutoCounts(
            rp_edges=counts.rp_edges,
            rp_centers=counts.rp_centers,
            pi_edges=counts.pi_edges,
            pi_centers=counts.pi_centers,
            dd=counts.dd_boot[:, :, ib],
            rr=counts.rr,
            dr=counts.dr,
            intpi_dd=_integrate_pi(
                counts.dd_boot[:, :, ib],
                counts.pi_edges[1:] - counts.pi_edges[:-1],
            ),
            intpi_rr=counts.intpi_rr,
            intpi_dr=counts.intpi_dr,
            metadata=dict(counts.metadata),
        )
        one = estimate_auto(c, estimator=estimator, data_weights=data_weights)
        curves.append(np.asarray(one.wp, dtype=np.float64))
        if store_cumulative:
            cumulative_curves.append(
                compute_auto_cumulative_wp(
                    c,
                    estimator=estimator,
                    data_weights=data_weights,
                )
            )
    return (
        np.asarray(curves, dtype=np.float64),
        np.asarray(cumulative_curves, dtype=np.float64) if store_cumulative else None,
    )


def _bootstrap_wp_from_auto_counts(
    counts: ProjectedAutoCounts,
    *,
    estimator: str,
    data_weights: np.ndarray | None = None,
) -> np.ndarray:
    return _bootstrap_products_from_auto_counts(
        counts,
        estimator=estimator,
        data_weights=data_weights,
        store_cumulative=False,
    )[0]


def _bootstrap_products_from_cross_counts(
    counts: ProjectedCrossCounts,
    *,
    estimator: str,
    sum_w1: float | None = None,
    sum_w2: float | None = None,
    store_cumulative: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return integrated and optionally cumulative cross-bootstrap realizations."""
    tmp = estimate_cross(
        counts,
        estimator=estimator,
        sum_w1=sum_w1,
        sum_w2=sum_w2,
        store_bootstrap_cumulative=store_cumulative,
    )
    arr = getattr(tmp, "bootstrap_realizations", None)
    cumulative = getattr(tmp, "bootstrap_cumulative_realizations", None)
    if arr is not None:
        return (
            np.asarray(arr, dtype=np.float64),
            None if cumulative is None else np.asarray(cumulative, dtype=np.float64),
        )

    if counts.d1d2_boot is None or counts.d1d2_boot.size == 0:
        empty_wp = np.zeros((0, len(counts.rp_centers)), dtype=np.float64)
        empty_cumulative = None
        if store_cumulative:
            empty_cumulative = np.zeros(
                (0, len(counts.rp_centers), len(counts.pi_centers)),
                dtype=np.float64,
            )
        return empty_wp, empty_cumulative

    curves = []
    cumulative_curves = []
    for ib in range(counts.d1d2_boot.shape[2]):
        c = ProjectedCrossCounts(
            rp_edges=counts.rp_edges,
            rp_centers=counts.rp_centers,
            pi_edges=counts.pi_edges,
            pi_centers=counts.pi_centers,
            d1d2=counts.d1d2_boot[:, :, ib],
            d1r2=None if counts.d1r2_boot is None else counts.d1r2_boot[:, :, ib],
            r1d2=counts.r1d2,
            r1r2=counts.r1r2,
            intpi_d1d2=_integrate_pi(
                counts.d1d2_boot[:, :, ib],
                counts.pi_edges[1:] - counts.pi_edges[:-1],
            ),
            intpi_d1r2=None if counts.d1r2_boot is None else _integrate_pi(
                counts.d1r2_boot[:, :, ib],
                counts.pi_edges[1:] - counts.pi_edges[:-1],
            ),
            intpi_r1d2=counts.intpi_r1d2,
            intpi_r1r2=counts.intpi_r1r2,
            metadata=dict(counts.metadata),
        )
        one = estimate_cross(
            c,
            estimator=estimator,
            sum_w1=sum_w1,
            sum_w2=sum_w2,
        )
        curves.append(np.asarray(one.wp, dtype=np.float64))
        if store_cumulative:
            cumulative_curves.append(
                compute_cross_cumulative_wp(
                    c,
                    estimator=estimator,
                    sum_w1=sum_w1,
                    sum_w2=sum_w2,
                )
            )
    return (
        np.asarray(curves, dtype=np.float64),
        np.asarray(cumulative_curves, dtype=np.float64) if store_cumulative else None,
    )


def _bootstrap_wp_from_cross_counts(
    counts: ProjectedCrossCounts,
    *,
    estimator: str,
    sum_w1: float | None = None,
    sum_w2: float | None = None,
) -> np.ndarray:
    return _bootstrap_products_from_cross_counts(
        counts,
        estimator=estimator,
        sum_w1=sum_w1,
        sum_w2=sum_w2,
        store_cumulative=False,
    )[0]


def _grid_tuple(template: dict[str, Any], config, *, sbound) -> tuple[int, int, int]:
    if bool(config.grid.autogrid):
        h1, h2, h3, _ = best_skgrid_3d_legacy(
            int(template["nrows"]),
            np.asarray(template["ra"], dtype=np.float64),
            sbound=sbound,
            nsepv=int(config.binning.nsepv),
            dsepv=float(config.binning.dsepv),
            dens=config.grid.dens,
        )
        return int(h1), int(h2), int(h3)
    return int(config.grid.mxh1), int(config.grid.mxh2), int(config.grid.mxh3)


def _grid_meta(config) -> dict[str, Any]:
    return {
        "autogrid": bool(config.grid.autogrid),
        "dens": config.grid.dens,
        "pxorder": config.grid.pxorder,
        "nsepv": int(config.binning.nsepv),
        "dsepv": float(config.binning.dsepv),
    }


def _build_prepared(template: dict[str, Any], dist: np.ndarray, *, sbound, grid_tuple, pi_edges, grid_meta) -> Any:
    return _prepared_projected_from_arrays(
        ra=np.asarray(template["ra"], dtype=np.float64),
        dec=np.asarray(template["dec"], dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64),
        weights=np.asarray(template["weights"], dtype=np.float64),
        sbound=sbound,
        mxh1=int(grid_tuple[0]),
        mxh2=int(grid_tuple[1]),
        mxh3=int(grid_tuple[2]),
        pi_edges=np.asarray(pi_edges, dtype=np.float64),
        region_id=None if template.get("region_id", None) is None else np.asarray(template["region_id"], dtype=np.int32),
        grid_meta=grid_meta,
        sort_rows=True,
    )


def _fixed_distance_or_none(table, columns, config, pdf_source) -> np.ndarray | None:
    if pdf_source is not None:
        return None
    return np.asarray(_distance_array(table, columns, config), dtype=np.float64)


def _auto_meta(config, data, random, p_data, p_random, chi_grid) -> dict[str, Any]:
    rp_edges = np.asarray(config.binning.rp_edges, dtype=np.float64)
    rp_centers = np.asarray(config.binning.rp_centers, dtype=np.float64)
    pi_edges = np.asarray(config.binning.pi_edges, dtype=np.float64)
    pi_centers = np.asarray(config.binning.pi_centers, dtype=np.float64)
    pi_delta = np.asarray(pi_edges[1:] - pi_edges[:-1], dtype=np.float64)
    data_dist = _fixed_distance_or_none(data, config.columns_data, config, p_data)
    rand_dist = None if (p_random is None and p_data is not None) else _fixed_distance_or_none(random, config.columns_random, config, p_random)
    dec_arrays = [_col(data, config.columns_data.dec), _col(random, config.columns_random.dec)]
    dist_arrays = []
    if data_dist is None:
        dist_arrays.append(np.asarray([float(chi_grid[0]), float(chi_grid[-1])], dtype=np.float64))
    else:
        dist_arrays.append(np.asarray(data_dist, dtype=np.float64))
    if rand_dist is None:
        dist_arrays.append(np.asarray([float(chi_grid[0]), float(chi_grid[-1])], dtype=np.float64))
    else:
        dist_arrays.append(np.asarray(rand_dist, dtype=np.float64))
    sbound = bound3d(dec_arrays, dist_arrays)
    return {
        "rp_edges": rp_edges,
        "rp_centers": rp_centers,
        "pi_edges": pi_edges,
        "pi_centers": pi_centers,
        "pi_delta": pi_delta,
        "sbound": tuple(float(v) for v in sbound),
    }


def _cross_meta(config, data1, random1, data2, random2, p1, pr1, p2, pr2, chi_grid) -> dict[str, Any]:
    rp_edges = np.asarray(config.binning.rp_edges, dtype=np.float64)
    rp_centers = np.asarray(config.binning.rp_centers, dtype=np.float64)
    pi_edges = np.asarray(config.binning.pi_edges, dtype=np.float64)
    pi_centers = np.asarray(config.binning.pi_centers, dtype=np.float64)
    pi_delta = np.asarray(pi_edges[1:] - pi_edges[:-1], dtype=np.float64)
    dec_arrays = [_col(data1, config.columns_data1.dec), _col(data2, config.columns_data2.dec)]
    dist_arrays = []
    d1 = _fixed_distance_or_none(data1, config.columns_data1, config, p1)
    d2 = _fixed_distance_or_none(data2, config.columns_data2, config, p2)
    dist_arrays.append(np.asarray([float(chi_grid[0]), float(chi_grid[-1])], dtype=np.float64) if d1 is None else np.asarray(d1, dtype=np.float64))
    dist_arrays.append(np.asarray([float(chi_grid[0]), float(chi_grid[-1])], dtype=np.float64) if d2 is None else np.asarray(d2, dtype=np.float64))
    if random1 is not None:
        dec_arrays.append(_col(random1, config.columns_random1.dec))
        r1 = None if (pr1 is None and p1 is not None) else _fixed_distance_or_none(random1, config.columns_random1, config, pr1)
        dist_arrays.append(np.asarray([float(chi_grid[0]), float(chi_grid[-1])], dtype=np.float64) if r1 is None else np.asarray(r1, dtype=np.float64))
    if random2 is not None:
        dec_arrays.append(_col(random2, config.columns_random2.dec))
        r2 = None if (pr2 is None and p2 is not None) else _fixed_distance_or_none(random2, config.columns_random2, config, pr2)
        dist_arrays.append(np.asarray([float(chi_grid[0]), float(chi_grid[-1])], dtype=np.float64) if r2 is None else np.asarray(r2, dtype=np.float64))
    sbound = bound3d(dec_arrays, dist_arrays)
    return {
        "rp_edges": rp_edges,
        "rp_centers": rp_centers,
        "pi_edges": pi_edges,
        "pi_centers": pi_centers,
        "pi_delta": pi_delta,
        "sbound": tuple(float(v) for v in sbound),
    }


def _auto_jackknife_regions(data, random, config) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    meta = {"jk_enabled": bool(config.jackknife.enabled), "jk_region_source": None, "jk_nregions": None, "jk_geometry_from": None}
    if not bool(config.jackknife.enabled):
        return None, None, meta
    dreg = config.columns_data.region
    rreg = config.columns_random.region
    if dreg is not None and rreg is not None:
        data_region, random_region = _shared_user_region_ids([(data, dreg), (random, rreg)])
        all_regs = np.concatenate([data_region, random_region])
        meta.update({"jk_region_source": "user", "jk_nregions": int(np.max(all_regs) + 1) if all_regs.size else 0})
        return data_region, random_region, meta
    if dreg is not None or rreg is not None:
        raise ValueError("For projected MC jackknife auto-correlations, either both data/random region columns must be supplied or neither.")
    nregions = config.jackknife.nregions
    if nregions is None:
        nregions = choose_default_nregions(config.binning.nsepp)
    assignments, _centers = _auto_region_ids([
        (data, config.columns_data.ra, config.columns_data.dec, "data"),
        (random, config.columns_random.ra, config.columns_random.dec, "random"),
    ], nregions=min(int(nregions), max(1, catalog_nrows(data), catalog_nrows(random))), seed=config.jackknife.seed, geometry_from=config.jackknife.geometry_from)
    data_region, random_region = assignments
    all_regs = np.concatenate([data_region, random_region])
    meta.update({"jk_region_source": "auto", "jk_nregions": int(np.max(all_regs) + 1) if all_regs.size else 0, "jk_geometry_from": config.jackknife.geometry_from})
    return data_region, random_region, meta


def _cross_jackknife_regions(data1, random1, data2, random2, config) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    meta = {"jk_enabled": bool(config.jackknife.enabled), "jk_region_source": None, "jk_nregions": None, "jk_geometry_from": None}
    if not bool(config.jackknife.enabled):
        return None, None, None, None, meta
    user_pairs = []
    if config.columns_data1.region is not None:
        user_pairs.append((data1, config.columns_data1.region))
    if random1 is not None and config.columns_random1.region is not None:
        user_pairs.append((random1, config.columns_random1.region))
    if config.columns_data2.region is not None:
        user_pairs.append((data2, config.columns_data2.region))
    if random2 is not None and config.columns_random2.region is not None:
        user_pairs.append((random2, config.columns_random2.region))
    expected = 2 + (1 if random1 is not None else 0) + (1 if random2 is not None else 0)
    if user_pairs and len(user_pairs) != expected:
        raise ValueError("For projected MC jackknife cross-correlations, region columns must be supplied for all participating catalogs or for none of them.")
    if user_pairs:
        regs = _shared_user_region_ids(user_pairs)
        pos = 0
        d1_region = regs[pos]; pos += 1
        r1_region = None
        if random1 is not None:
            r1_region = regs[pos]; pos += 1
        d2_region = regs[pos]; pos += 1
        r2_region = regs[pos] if random2 is not None else None
        all_regs = np.concatenate([arr for arr in (d1_region, r1_region, d2_region, r2_region) if arr is not None])
        meta.update({"jk_region_source": "user", "jk_nregions": int(np.max(all_regs) + 1) if all_regs.size else 0})
        return d1_region, r1_region, d2_region, r2_region, meta
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
    r1_region = assignments[pos] if random1 is not None else None
    if random1 is not None:
        pos += 1
    r2_region = assignments[pos] if random2 is not None else None
    all_regs = np.concatenate([arr for arr in (d1_region, r1_region, d2_region, r2_region) if arr is not None])
    meta.update({"jk_region_source": "auto", "jk_nregions": int(np.max(all_regs) + 1) if all_regs.size else 0, "jk_geometry_from": config.jackknife.geometry_from})
    return d1_region, r1_region, d2_region, r2_region, meta


def _mean_pdf(p: np.ndarray) -> np.ndarray:
    prof = np.asarray(np.mean(p, axis=0), dtype=np.float64)
    s = float(np.sum(prof))
    if s <= 0.0:
        raise ValueError("mc_pdf mean PDF has zero total probability.")
    return prof / s


def _indices_to_dist(
    idx: np.ndarray,
    chi_grid: np.ndarray,
    rng: np.random.Generator,
    *,
    sample_edges_chi: np.ndarray | None = None,
) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return np.empty(0, dtype=np.float64)
    if sample_edges_chi is None:
        return np.asarray(chi_grid[idx], dtype=np.float64)
    edges = np.asarray(sample_edges_chi, dtype=np.float64)
    if edges.size != np.asarray(chi_grid).size + 1:
        raise ValueError(
            "sample_within_bin=True requires one more grid edge than PDF columns / chi-grid centers."
        )
    lo = edges[idx]
    hi = edges[idx + 1]
    u = rng.random(idx.size, dtype=np.float64)
    return np.asarray(lo + u * (hi - lo), dtype=np.float64)


def _sample_dist_from_source(
    p_source: np.ndarray,
    chi_grid: np.ndarray,
    rng: np.random.Generator,
    *,
    sample_edges_chi: np.ndarray | None = None,
) -> np.ndarray:
    idx = _sample_indices(_build_cdf(p_source), rng)
    return _indices_to_dist(idx, chi_grid, rng, sample_edges_chi=sample_edges_chi)


def _sample_dist_from_cdf(
    cdf: np.ndarray,
    chi_grid: np.ndarray,
    rng: np.random.Generator,
    *,
    sample_edges_chi: np.ndarray | None = None,
) -> np.ndarray:
    idx = _sample_indices(cdf, rng)
    return _indices_to_dist(idx, chi_grid, rng, sample_edges_chi=sample_edges_chi)


def _sample_random_dist(
    nrand: int,
    *,
    random_source_cdf,
    data_source_mean,
    data_draw,
    chi_grid,
    random_mode: str,
    rng: np.random.Generator,
    sample_edges_chi: np.ndarray | None = None,
) -> np.ndarray:
    if random_source_cdf is not None:
        return _sample_dist_from_cdf(random_source_cdf, chi_grid, rng, sample_edges_chi=sample_edges_chi)
    if nrand == 0:
        return np.empty(0, dtype=np.float64)
    mode = str(random_mode).strip().lower()
    if mode in {"fixed_global", "rerun_global"}:
        if data_source_mean is None:
            raise ValueError("mc_pdf random_mode requires a data PDF source when explicit random PDFs are not supplied.")
        idx = _sample_from_global(data_source_mean, nrand, rng)
        return _indices_to_dist(idx, chi_grid, rng, sample_edges_chi=sample_edges_chi)
    if mode == "inherit_realization":
        if data_draw is None or len(data_draw) == 0:
            raise ValueError("mc_pdf inherit_realization requires a non-empty data realization.")
        pick = rng.integers(0, len(data_draw), size=nrand, dtype=np.int64)
        return np.asarray(np.asarray(data_draw, dtype=np.float64)[pick], dtype=np.float64)
    raise ValueError("mc_pdf.random_mode must be 'fixed_global', 'rerun_global', or 'inherit_realization'.")


def _assemble_auto_counts_from_terms(dd_res, rr_arr, dr_arr, *, meta, rr_norm_pairs, extra_metadata=None) -> ProjectedAutoCounts:
    metadata = {
        "n_data": int(dd_res.metadata["n_data"]),
        "n_random": int(extra_metadata.get("n_random", 0) if extra_metadata else 0),
        "data_weighted": bool(dd_res.metadata.get("data_weighted", False)),
        "jk_nregions": 0,
        "jk_touch_available": False,
        "rr_norm_pairs": float(rr_norm_pairs),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
        metadata["rr_norm_pairs"] = float(rr_norm_pairs)
    return ProjectedAutoCounts(
        rp_edges=np.asarray(meta["rp_edges"], dtype=np.float64),
        rp_centers=np.asarray(meta["rp_centers"], dtype=np.float64),
        pi_edges=np.asarray(meta["pi_edges"], dtype=np.float64),
        pi_centers=np.asarray(meta["pi_centers"], dtype=np.float64),
        dd=np.asarray(dd_res.dd, dtype=np.float64),
        rr=None if rr_arr is None else np.asarray(rr_arr, dtype=np.float64),
        dr=None if dr_arr is None else np.asarray(dr_arr, dtype=np.float64),
        intpi_dd=_integrate_pi(dd_res.dd, meta["pi_delta"]),
        intpi_rr=_integrate_pi(rr_arr, meta["pi_delta"]),
        intpi_dr=_integrate_pi(dr_arr, meta["pi_delta"]),
        metadata=metadata,
    )


def _accumulate_auto(target: dict[str, Any] | None, counts: ProjectedAutoCounts) -> dict[str, Any]:
    if target is None:
        return {
            "dd": np.asarray(counts.dd, dtype=np.float64).copy(),
            "rr": None if counts.rr is None else np.asarray(counts.rr, dtype=np.float64).copy(),
            "dr": None if counts.dr is None else np.asarray(counts.dr, dtype=np.float64).copy(),
            "metadata": dict(counts.metadata),
        }
    target["dd"] += np.asarray(counts.dd, dtype=np.float64)
    if target["rr"] is not None and counts.rr is not None:
        target["rr"] += np.asarray(counts.rr, dtype=np.float64)
    if target["dr"] is not None and counts.dr is not None:
        target["dr"] += np.asarray(counts.dr, dtype=np.float64)
    return target


def _finalize_auto_counts(acc: dict[str, Any], *, meta, nreal: int) -> ProjectedAutoCounts:
    dd = np.asarray(acc["dd"], dtype=np.float64) / float(nreal)
    rr = None if acc["rr"] is None else np.asarray(acc["rr"], dtype=np.float64) / float(nreal)
    dr = None if acc["dr"] is None else np.asarray(acc["dr"], dtype=np.float64) / float(nreal)
    md = dict(acc["metadata"])
    return ProjectedAutoCounts(
        rp_edges=np.asarray(meta["rp_edges"], dtype=np.float64),
        rp_centers=np.asarray(meta["rp_centers"], dtype=np.float64),
        pi_edges=np.asarray(meta["pi_edges"], dtype=np.float64),
        pi_centers=np.asarray(meta["pi_centers"], dtype=np.float64),
        dd=dd,
        rr=rr,
        dr=dr,
        intpi_dd=_integrate_pi(dd, meta["pi_delta"]),
        intpi_rr=_integrate_pi(rr, meta["pi_delta"]),
        intpi_dr=_integrate_pi(dr, meta["pi_delta"]),
        metadata=md,
    )


def _accumulate_auto_bootstrap(target: dict[str, Any] | None, counts: ProjectedAutoCounts) -> dict[str, Any]:
    target = _accumulate_auto(target, counts)
    bdd = None if counts.dd_boot is None else np.asarray(counts.dd_boot, dtype=np.float64)
    if bdd is not None:
        if "dd_boot" not in target or target.get("dd_boot") is None:
            target["dd_boot"] = bdd.copy()
        else:
            target["dd_boot"] += bdd
    normb = None if counts.norm_dd_boot is None else np.asarray(counts.norm_dd_boot, dtype=np.float64)
    if normb is not None:
        if "norm_dd_boot" not in target or target.get("norm_dd_boot") is None:
            target["norm_dd_boot"] = normb.copy()
        else:
            target["norm_dd_boot"] += normb
    sumwb = None if counts.sum_w_data_boot is None else np.asarray(counts.sum_w_data_boot, dtype=np.float64)
    if sumwb is not None:
        if "sum_w_data_boot" not in target or target.get("sum_w_data_boot") is None:
            target["sum_w_data_boot"] = sumwb.copy()
        else:
            target["sum_w_data_boot"] += sumwb
    return target


def _finalize_auto_bootstrap_counts(acc: dict[str, Any], *, meta, nreal: int) -> ProjectedAutoCounts:
    counts = _finalize_auto_counts(acc, meta=meta, nreal=nreal)
    if acc.get("dd_boot") is not None:
        counts.dd_boot = np.asarray(acc["dd_boot"], dtype=np.float64) / float(nreal)
    if acc.get("norm_dd_boot") is not None:
        counts.norm_dd_boot = np.asarray(acc["norm_dd_boot"], dtype=np.float64) / float(nreal)
    if acc.get("sum_w_data_boot") is not None:
        counts.sum_w_data_boot = np.asarray(acc["sum_w_data_boot"], dtype=np.float64) / float(nreal)
    counts.metadata["mc_bootstrap_count_average"] = True
    return counts


def _accumulate_auto_jackknife(target: dict[str, Any] | None, counts: ProjectedAutoCounts) -> dict[str, Any]:
    target = _accumulate_auto(target, counts)
    for name in ("dd_jk_touch", "rr_jk_touch", "dr_jk_touch"):
        arr = getattr(counts, name, None)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if name not in target or target.get(name) is None:
            target[name] = arr.copy()
        else:
            target[name] += arr
    return target


def _finalize_auto_jackknife_counts(acc: dict[str, Any], *, meta, nreal: int) -> ProjectedAutoCounts:
    counts = _finalize_auto_counts(acc, meta=meta, nreal=nreal)
    for name in ("dd_jk_touch", "rr_jk_touch", "dr_jk_touch"):
        if acc.get(name) is not None:
            setattr(counts, name, np.asarray(acc[name], dtype=np.float64) / float(nreal))
    counts.metadata["mc_jackknife_count_average"] = True
    counts.metadata["jk_touch_available"] = counts.dd_jk_touch is not None
    counts.metadata["jk_nregions"] = int(meta.get("jk_nregions") or counts.metadata.get("jk_nregions", 0) or 0)
    return counts


def _auto_touch_ready_mc(counts: ProjectedAutoCounts, estimator: str) -> bool:
    est = str(estimator).upper()
    if counts.dd_jk_touch is None:
        return False
    if est == "NAT":
        return counts.rr_jk_touch is not None
    if est == "DP":
        return counts.dr_jk_touch is not None
    if est == "LS":
        return counts.rr_jk_touch is not None and counts.dr_jk_touch is not None
    return False


def _jackknife_wp_from_auto_touch(counts: ProjectedAutoCounts, data_template: dict[str, Any], rand_template: dict[str, Any], config, meta) -> np.ndarray:
    nregions = int(meta.get("jk_nregions") or counts.metadata.get("jk_nregions", 0) or 0)
    if nregions <= 1:
        return np.zeros((0, len(meta["rp_centers"])), dtype=np.float64)
    if not _auto_touch_ready_mc(counts, config.estimator):
        raise RuntimeError("MC fast jackknife requested but auto touch counts are unavailable for this estimator.")
    data_reg = np.asarray(data_template.get("region_id"), dtype=np.int64)
    rand_reg = np.asarray(rand_template.get("region_id"), dtype=np.int64)
    ndata_reg = np.bincount(data_reg, minlength=nregions)
    nrand_reg = np.bincount(rand_reg, minlength=nregions)
    ndata = int(len(data_reg))
    nrand = int(len(rand_reg))
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and not data_template["wunit"])
    if weighted:
        w = np.asarray(data_template["weights"], dtype=np.float64)
        sumw = float(np.sum(w))
        sumw2 = float(np.sum(np.square(w)))
        sumw_reg = np.bincount(data_reg, weights=w, minlength=nregions).astype(np.float64)
        sumw2_reg = np.bincount(data_reg, weights=np.square(w), minlength=nregions).astype(np.float64)
    else:
        sumw = sumw2 = None
        sumw_reg = sumw2_reg = None
    pi_delta = np.asarray(meta["pi_delta"], dtype=np.float64)
    out = np.zeros((nregions, len(meta["rp_centers"])), dtype=np.float64)
    for k in range(nregions):
        counts_k = ProjectedAutoCounts(
            rp_edges=counts.rp_edges, rp_centers=counts.rp_centers,
            pi_edges=counts.pi_edges, pi_centers=counts.pi_centers,
            dd=counts.dd - counts.dd_jk_touch[:, :, k],
            rr=None if counts.rr is None else counts.rr - counts.rr_jk_touch[:, :, k],
            dr=None if counts.dr is None else counts.dr - counts.dr_jk_touch[:, :, k],
            metadata={
                "n_data": ndata - int(ndata_reg[k]),
                "n_random": nrand - int(nrand_reg[k]),
                "data_weighted": weighted,
            },
        )
        xi2d = compute_auto_xi2d(
            counts_k,
            estimator=config.estimator,
            sum_w_data=None if not weighted else (sumw - float(sumw_reg[k])),
            sum_w2_data=None if not weighted else (sumw2 - float(sumw2_reg[k])),
        )
        out[k] = 2.0 * np.sum(xi2d * pi_delta[None, :], axis=1)
    return out


def _accumulate_cross(target: dict[str, Any] | None, counts: ProjectedCrossCounts) -> dict[str, Any]:
    if target is None:
        return {
            "d1d2": np.asarray(counts.d1d2, dtype=np.float64).copy(),
            "d1r2": None if counts.d1r2 is None else np.asarray(counts.d1r2, dtype=np.float64).copy(),
            "r1d2": None if counts.r1d2 is None else np.asarray(counts.r1d2, dtype=np.float64).copy(),
            "r1r2": None if counts.r1r2 is None else np.asarray(counts.r1r2, dtype=np.float64).copy(),
            "metadata": dict(counts.metadata),
        }
    target["d1d2"] += np.asarray(counts.d1d2, dtype=np.float64)
    if target["d1r2"] is not None and counts.d1r2 is not None:
        target["d1r2"] += np.asarray(counts.d1r2, dtype=np.float64)
    if target["r1d2"] is not None and counts.r1d2 is not None:
        target["r1d2"] += np.asarray(counts.r1d2, dtype=np.float64)
    if target["r1r2"] is not None and counts.r1r2 is not None:
        target["r1r2"] += np.asarray(counts.r1r2, dtype=np.float64)
    return target


def _finalize_cross_counts(acc: dict[str, Any], *, meta, nreal: int) -> ProjectedCrossCounts:
    d1d2 = np.asarray(acc["d1d2"], dtype=np.float64) / float(nreal)
    d1r2 = None if acc["d1r2"] is None else np.asarray(acc["d1r2"], dtype=np.float64) / float(nreal)
    r1d2 = None if acc["r1d2"] is None else np.asarray(acc["r1d2"], dtype=np.float64) / float(nreal)
    r1r2 = None if acc["r1r2"] is None else np.asarray(acc["r1r2"], dtype=np.float64) / float(nreal)
    return ProjectedCrossCounts(
        rp_edges=np.asarray(meta["rp_edges"], dtype=np.float64),
        rp_centers=np.asarray(meta["rp_centers"], dtype=np.float64),
        pi_edges=np.asarray(meta["pi_edges"], dtype=np.float64),
        pi_centers=np.asarray(meta["pi_centers"], dtype=np.float64),
        d1d2=d1d2,
        d1r2=d1r2,
        r1d2=r1d2,
        r1r2=r1r2,
        intpi_d1d2=_integrate_pi(d1d2, meta["pi_delta"]),
        intpi_d1r2=_integrate_pi(d1r2, meta["pi_delta"]),
        intpi_r1d2=_integrate_pi(r1d2, meta["pi_delta"]),
        intpi_r1r2=_integrate_pi(r1r2, meta["pi_delta"]),
        metadata=dict(acc["metadata"]),
    )


def _accumulate_cross_bootstrap(target: dict[str, Any] | None, counts: ProjectedCrossCounts) -> dict[str, Any]:
    target = _accumulate_cross(target, counts)
    b12 = None if counts.d1d2_boot is None else np.asarray(counts.d1d2_boot, dtype=np.float64)
    if b12 is not None:
        if "d1d2_boot" not in target or target.get("d1d2_boot") is None:
            target["d1d2_boot"] = b12.copy()
        else:
            target["d1d2_boot"] += b12
    b1r = None if counts.d1r2_boot is None else np.asarray(counts.d1r2_boot, dtype=np.float64)
    if b1r is not None:
        if "d1r2_boot" not in target or target.get("d1r2_boot") is None:
            target["d1r2_boot"] = b1r.copy()
        else:
            target["d1r2_boot"] += b1r
    return target


def _finalize_cross_bootstrap_counts(acc: dict[str, Any], *, meta, nreal: int) -> ProjectedCrossCounts:
    counts = _finalize_cross_counts(acc, meta=meta, nreal=nreal)
    if acc.get("d1d2_boot") is not None:
        counts.d1d2_boot = np.asarray(acc["d1d2_boot"], dtype=np.float64) / float(nreal)
    if acc.get("d1r2_boot") is not None:
        counts.d1r2_boot = np.asarray(acc["d1r2_boot"], dtype=np.float64) / float(nreal)
    counts.metadata["mc_bootstrap_count_average"] = True
    return counts


def _accumulate_cross_jackknife(target: dict[str, Any] | None, counts: ProjectedCrossCounts) -> dict[str, Any]:
    target = _accumulate_cross(target, counts)
    for name in ("d1d2_jk_touch", "d1r2_jk_touch", "r1d2_jk_touch", "r1r2_jk_touch"):
        arr = getattr(counts, name, None)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if name not in target or target.get(name) is None:
            target[name] = arr.copy()
        else:
            target[name] += arr
    return target


def _finalize_cross_jackknife_counts(acc: dict[str, Any], *, meta, nreal: int) -> ProjectedCrossCounts:
    counts = _finalize_cross_counts(acc, meta=meta, nreal=nreal)
    for name in ("d1d2_jk_touch", "d1r2_jk_touch", "r1d2_jk_touch", "r1r2_jk_touch"):
        if acc.get(name) is not None:
            setattr(counts, name, np.asarray(acc[name], dtype=np.float64) / float(nreal))
    counts.metadata["mc_jackknife_count_average"] = True
    counts.metadata["jk_touch_available"] = counts.d1d2_jk_touch is not None
    counts.metadata["jk_nregions"] = int(meta.get("jk_nregions") or counts.metadata.get("jk_nregions", 0) or 0)
    return counts


def _cross_touch_ready_mc(counts: ProjectedCrossCounts, estimator: str) -> bool:
    est = str(estimator).upper()
    if counts.d1d2_jk_touch is None:
        return False
    if est == "NAT":
        return counts.r1r2_jk_touch is not None
    if est == "DP":
        return counts.d1r2_jk_touch is not None
    if est == "LS":
        return counts.d1r2_jk_touch is not None and counts.r1d2_jk_touch is not None and counts.r1r2_jk_touch is not None
    return False


def _jackknife_wp_from_cross_touch(counts: ProjectedCrossCounts, t1: dict[str, Any], t2: dict[str, Any], tr1: dict[str, Any] | None, tr2: dict[str, Any] | None, config, meta) -> np.ndarray:
    nregions = int(meta.get("jk_nregions") or counts.metadata.get("jk_nregions", 0) or 0)
    if nregions <= 1:
        return np.zeros((0, len(meta["rp_centers"])), dtype=np.float64)
    if not _cross_touch_ready_mc(counts, config.estimator):
        raise RuntimeError("MC fast jackknife requested but cross touch counts are unavailable for this estimator.")
    d1reg = np.asarray(t1.get("region_id"), dtype=np.int64)
    d2reg = np.asarray(t2.get("region_id"), dtype=np.int64)
    d1n = np.bincount(d1reg, minlength=nregions)
    d2n = np.bincount(d2reg, minlength=nregions)
    r1n = np.zeros(nregions, dtype=np.int64) if tr1 is None else np.bincount(np.asarray(tr1.get("region_id"), dtype=np.int64), minlength=nregions)
    r2n = np.zeros(nregions, dtype=np.int64) if tr2 is None else np.bincount(np.asarray(tr2.get("region_id"), dtype=np.int64), minlength=nregions)
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and (not t1["wunit"] or not t2["wunit"]))
    if weighted:
        w1 = np.asarray(t1["weights"], dtype=np.float64)
        w2 = np.asarray(t2["weights"], dtype=np.float64)
        sw1 = float(np.sum(w1)); sw2 = float(np.sum(w2))
        sw1_reg = np.bincount(d1reg, weights=w1, minlength=nregions).astype(np.float64)
        sw2_reg = np.bincount(d2reg, weights=w2, minlength=nregions).astype(np.float64)
    else:
        sw1 = sw2 = None
        sw1_reg = sw2_reg = None
    pi_delta = np.asarray(meta["pi_delta"], dtype=np.float64)
    out = np.zeros((nregions, len(meta["rp_centers"])), dtype=np.float64)
    for k in range(nregions):
        counts_k = ProjectedCrossCounts(
            rp_edges=counts.rp_edges, rp_centers=counts.rp_centers,
            pi_edges=counts.pi_edges, pi_centers=counts.pi_centers,
            d1d2=counts.d1d2 - counts.d1d2_jk_touch[:, :, k],
            d1r2=None if counts.d1r2 is None else counts.d1r2 - counts.d1r2_jk_touch[:, :, k],
            r1d2=None if counts.r1d2 is None else counts.r1d2 - counts.r1d2_jk_touch[:, :, k],
            r1r2=None if counts.r1r2 is None else counts.r1r2 - counts.r1r2_jk_touch[:, :, k],
            metadata={
                "n_data1": int(t1["nrows"]) - int(d1n[k]),
                "n_random1": (0 if tr1 is None else int(tr1["nrows"]) - int(r1n[k])),
                "n_data2": int(t2["nrows"]) - int(d2n[k]),
                "n_random2": (0 if tr2 is None else int(tr2["nrows"]) - int(r2n[k])),
                "primary": counts.metadata.get("primary", config.bootstrap.primary),
            },
        )
        xi2d = compute_cross_xi2d(
            counts_k,
            estimator=config.estimator,
            sum_w1=None if not weighted else (sw1 - float(sw1_reg[k])),
            sum_w2=None if not weighted else (sw2 - float(sw2_reg[k])),
        )
        out[k] = 2.0 * np.sum(xi2d * pi_delta[None, :], axis=1)
    return out


def _auto_mc_mean_counts_for_selection(
    *,
    data_template: dict[str, Any],
    rand_template: dict[str, Any],
    cdf_data: np.ndarray,
    pbar_data: np.ndarray,
    cdf_random: np.ndarray | None,
    chi_grid: np.ndarray,
    sample_edges_chi: np.ndarray | None,
    config: ProjectedAutoConfig,
    meta: dict[str, Any],
    sbound,
    grid_meta,
    data_grid,
    rand_grid,
    nreal: int,
    base_seed: int,
    data_index=None,
    rand_index=None,
    fixed_rand_dist: np.ndarray | None = None,
    status_label: str = "resample",
) -> tuple[ProjectedAutoCounts, dict[str, Any]]:
    data_t = _take_template(data_template, data_index)
    rand_t = _take_template(rand_template, rand_index)
    acc = None
    mode = str(config.mc_pdf.random_mode).strip().lower()
    fixed_random = mode == "fixed_global"
    fixed_rand_p = None
    fixed_rr = None
    fixed_rr_meta = None
    data_idx = None if data_index is None else np.asarray(data_index, dtype=np.int64)
    rand_idx = None if rand_index is None else np.asarray(rand_index, dtype=np.int64)
    cdf_random_sub = None if cdf_random is None else (cdf_random if rand_idx is None else np.asarray(cdf_random, dtype=np.float64)[rand_idx])
    for m in range(int(nreal)):
        status_prefix = f"[pcf:mc_pdf:{status_label}] realization {m + 1}/{int(nreal)}  "
        rng = _child_rng(base_seed, status_label, m)
        full_data_dist = _sample_dist_from_cdf(cdf_data, chi_grid, rng, sample_edges_chi=sample_edges_chi)
        data_dist = full_data_dist if data_idx is None else np.asarray(full_data_dist, dtype=np.float64)[data_idx]
        data_p = _build_prepared(data_t, data_dist, sbound=sbound, grid_tuple=data_grid, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

        if fixed_random and fixed_rand_dist is not None:
            rdist = np.asarray(fixed_rand_dist, dtype=np.float64) if rand_idx is None else np.asarray(fixed_rand_dist, dtype=np.float64)[rand_idx]
            if fixed_rand_p is None:
                fixed_rand_p = _build_prepared(rand_t, rdist, sbound=sbound, grid_tuple=rand_grid, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
            rand_p = fixed_rand_p
        else:
            if cdf_random_sub is not None:
                rdist = _sample_dist_from_cdf(cdf_random_sub, chi_grid, rng, sample_edges_chi=sample_edges_chi)
            else:
                rdist = _sample_random_dist(
                    rand_t["nrows"],
                    random_source_cdf=None,
                    data_source_mean=pbar_data,
                    data_draw=data_dist,
                    chi_grid=chi_grid,
                    random_mode=config.mc_pdf.random_mode,
                    rng=rng,
                    sample_edges_chi=sample_edges_chi,
                )
            rand_p = _build_prepared(rand_t, rdist, sbound=sbound, grid_tuple=rand_grid, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

        if fixed_random:
            if fixed_rr is None:
                counts = _run_mc_progress(config, lambda progress_path: build_auto_counts(
                    data_p,
                    rand_p,
                    rp_edges=meta["rp_edges"],
                    rp_centers=meta["rp_centers"],
                    pi_edges=meta["pi_edges"],
                    pi_centers=meta["pi_centers"],
                    pi_delta=meta["pi_delta"],
                    nthreads=config.nthreads,
                    estimator=config.estimator,
                    weight_mode=config.weights.weight_mode,
                    doboot=False,
                    dojk=False,
                    nreg=0,
                    nbts=0,
                    bseed=config.bootstrap.bseed,
                    progress_file=progress_path,
                    split_random=config.split_random,
                ), status_prefix=status_prefix)
                fixed_rr = None if counts.rr is None else np.asarray(counts.rr, dtype=np.float64).copy()
                fixed_rr_meta = dict(counts.metadata)
            else:
                dd_res = _run_mc_progress(config, lambda progress_path: build_auto_count_result(
                    data_p,
                    rp_edges=meta["rp_edges"],
                    rp_centers=meta["rp_centers"],
                    pi_edges=meta["pi_edges"],
                    pi_centers=meta["pi_centers"],
                    pi_delta=meta["pi_delta"],
                    nthreads=config.nthreads,
                    weight_mode=config.weights.weight_mode,
                    doboot=False,
                    dojk=False,
                    nreg=0,
                    nbts=0,
                    bseed=config.bootstrap.bseed,
                    progress_file=progress_path,
                ), status_prefix=status_prefix)
                dr_arr = None
                if str(config.estimator).upper() in {"DP", "LS"}:
                    dr_res = _run_mc_progress(config, lambda progress_path: build_cross_count_result(
                        data_p,
                        rand_p,
                        rp_edges=meta["rp_edges"],
                        rp_centers=meta["rp_centers"],
                        pi_edges=meta["pi_edges"],
                        pi_centers=meta["pi_centers"],
                        pi_delta=meta["pi_delta"],
                        nthreads=config.nthreads,
                        weight_mode=config.weights.weight_mode,
                        doboot=False,
                        dojk=False,
                        nreg=0,
                        nbts=0,
                        bseed=config.bootstrap.bseed,
                        primary="data1",
                        progress_file=progress_path,
                    ), status_prefix=status_prefix)
                    dr_arr = np.asarray(dr_res.d1d2, dtype=np.float64)
                counts = _assemble_auto_counts_from_terms(
                    dd_res,
                    fixed_rr if str(config.estimator).upper() in {"NAT", "LS"} else None,
                    dr_arr,
                    meta=meta,
                    rr_norm_pairs=float(fixed_rr_meta.get("rr_norm_pairs", 0.5 * rand_p.nrows * max(rand_p.nrows - 1, 0))),
                    extra_metadata={k: v for k, v in fixed_rr_meta.items() if k not in {"n_data", "data_weighted"}},
                )
        else:
            counts = _run_mc_progress(config, lambda progress_path: build_auto_counts(
                data_p,
                rand_p,
                rp_edges=meta["rp_edges"],
                rp_centers=meta["rp_centers"],
                pi_edges=meta["pi_edges"],
                pi_centers=meta["pi_centers"],
                pi_delta=meta["pi_delta"],
                nthreads=config.nthreads,
                estimator=config.estimator,
                weight_mode=config.weights.weight_mode,
                doboot=False,
                dojk=False,
                nreg=0,
                nbts=0,
                bseed=config.bootstrap.bseed,
                progress_file=progress_path,
                split_random=config.split_random,
            ), status_prefix=status_prefix)
        acc = _accumulate_auto(acc, counts)
    mean_counts = _finalize_auto_counts(acc, meta=meta, nreal=int(nreal))
    return mean_counts, {"data_template": data_t, "random_template": rand_t}


def _auto_mc_result_for_selection(**kwargs):
    counts, ctx = _auto_mc_mean_counts_for_selection(**kwargs)
    config = kwargs["config"]
    data_t = ctx["data_template"]
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and not data_t["wunit"])
    result = estimate_auto(counts, estimator=config.estimator, data_weights=(data_t["weights"] if weighted else None))
    return result


def _cross_mc_mean_counts_for_selection(
    *,
    t1: dict[str, Any],
    t2: dict[str, Any],
    tr1: dict[str, Any] | None,
    tr2: dict[str, Any] | None,
    cdf1: np.ndarray | None,
    cdf2: np.ndarray | None,
    pbar1: np.ndarray | None,
    pbar2: np.ndarray | None,
    cdf_r1: np.ndarray | None,
    cdf_r2: np.ndarray | None,
    data1,
    data2,
    random1,
    random2,
    chi_grid: np.ndarray,
    sample_edges_chi: np.ndarray | None,
    config: ProjectedCrossConfig,
    meta: dict[str, Any],
    sbound,
    grid_meta,
    g1,
    g2,
    gr1,
    gr2,
    nreal: int,
    base_seed: int,
    idx1=None,
    idx2=None,
    ridx1=None,
    ridx2=None,
    fixed_rdist1: np.ndarray | None = None,
    fixed_rdist2: np.ndarray | None = None,
    status_label: str = "resample",
) -> tuple[ProjectedCrossCounts, dict[str, Any]]:
    tt1 = _take_template(t1, idx1)
    tt2 = _take_template(t2, idx2)
    rr1t = None if tr1 is None else _take_template(tr1, ridx1)
    rr2t = None if tr2 is None else _take_template(tr2, ridx2)
    ii1 = None if idx1 is None else np.asarray(idx1, dtype=np.int64)
    ii2 = None if idx2 is None else np.asarray(idx2, dtype=np.int64)
    ri1 = None if ridx1 is None else np.asarray(ridx1, dtype=np.int64)
    ri2 = None if ridx2 is None else np.asarray(ridx2, dtype=np.int64)
    cdf_r1_sub = None if cdf_r1 is None or ri1 is None else np.asarray(cdf_r1, dtype=np.float64)[ri1]
    if cdf_r1 is not None and ri1 is None:
        cdf_r1_sub = cdf_r1
    cdf_r2_sub = None if cdf_r2 is None or ri2 is None else np.asarray(cdf_r2, dtype=np.float64)[ri2]
    if cdf_r2 is not None and ri2 is None:
        cdf_r2_sub = cdf_r2
    mode = str(config.mc_pdf.random_mode).strip().lower()
    fixed_random = mode == "fixed_global"
    fixed_prep_r1 = None
    fixed_prep_r2 = None
    acc = None
    for m in range(int(nreal)):
        status_prefix = f"[pccf:mc_pdf:{status_label}] realization {m + 1}/{int(nreal)}  "
        rng = _child_rng(base_seed, status_label, m)
        if cdf1 is None:
            dist1_full = _distance_array(data1, config.columns_data1, config)
        else:
            dist1_full = _sample_dist_from_cdf(cdf1, chi_grid, rng, sample_edges_chi=sample_edges_chi)
        if cdf2 is None:
            dist2_full = _distance_array(data2, config.columns_data2, config)
        else:
            dist2_full = _sample_dist_from_cdf(cdf2, chi_grid, rng, sample_edges_chi=sample_edges_chi)
        dist1 = dist1_full if ii1 is None else np.asarray(dist1_full, dtype=np.float64)[ii1]
        dist2 = dist2_full if ii2 is None else np.asarray(dist2_full, dtype=np.float64)[ii2]
        prep1 = _build_prepared(tt1, dist1, sbound=sbound, grid_tuple=g1, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
        prep2 = _build_prepared(tt2, dist2, sbound=sbound, grid_tuple=g2, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

        if rr1t is None:
            prep_r1 = None
        elif fixed_random and fixed_rdist1 is not None:
            rdist1 = np.asarray(fixed_rdist1, dtype=np.float64) if ri1 is None else np.asarray(fixed_rdist1, dtype=np.float64)[ri1]
            if fixed_prep_r1 is None:
                fixed_prep_r1 = _build_prepared(rr1t, rdist1, sbound=sbound, grid_tuple=gr1, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
            prep_r1 = fixed_prep_r1
        else:
            if cdf_r1_sub is not None:
                rdist1 = _sample_dist_from_cdf(cdf_r1_sub, chi_grid, rng, sample_edges_chi=sample_edges_chi)
            elif pbar1 is None:
                rdist1 = _distance_array(random1, config.columns_random1, config)
                rdist1 = rdist1 if ri1 is None else np.asarray(rdist1, dtype=np.float64)[ri1]
            else:
                rdist1 = _sample_random_dist(rr1t["nrows"], random_source_cdf=None, data_source_mean=pbar1, data_draw=dist1, chi_grid=chi_grid, random_mode=config.mc_pdf.random_mode, rng=rng, sample_edges_chi=sample_edges_chi)
            prep_r1 = _build_prepared(rr1t, rdist1, sbound=sbound, grid_tuple=gr1, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

        if rr2t is None:
            prep_r2 = None
        elif fixed_random and fixed_rdist2 is not None:
            rdist2 = np.asarray(fixed_rdist2, dtype=np.float64) if ri2 is None else np.asarray(fixed_rdist2, dtype=np.float64)[ri2]
            if fixed_prep_r2 is None:
                fixed_prep_r2 = _build_prepared(rr2t, rdist2, sbound=sbound, grid_tuple=gr2, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
            prep_r2 = fixed_prep_r2
        else:
            if cdf_r2_sub is not None:
                rdist2 = _sample_dist_from_cdf(cdf_r2_sub, chi_grid, rng, sample_edges_chi=sample_edges_chi)
            elif pbar2 is None:
                rdist2 = _distance_array(random2, config.columns_random2, config)
                rdist2 = rdist2 if ri2 is None else np.asarray(rdist2, dtype=np.float64)[ri2]
            else:
                rdist2 = _sample_random_dist(rr2t["nrows"], random_source_cdf=None, data_source_mean=pbar2, data_draw=dist2, chi_grid=chi_grid, random_mode=config.mc_pdf.random_mode, rng=rng, sample_edges_chi=sample_edges_chi)
            prep_r2 = _build_prepared(rr2t, rdist2, sbound=sbound, grid_tuple=gr2, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

        counts = _run_mc_progress(config, lambda progress_path: build_cross_counts(
            prep1,
            prep_r1,
            prep2,
            prep_r2,
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            pi_delta=meta["pi_delta"],
            nthreads=config.nthreads,
            estimator=config.estimator,
            weight_mode=config.weights.weight_mode,
            doboot=False,
            dojk=False,
            nreg=0,
            nbts=0,
            bseed=config.bootstrap.bseed,
            primary=config.bootstrap.primary,
            progress_file=progress_path,
        ), status_prefix=status_prefix)
        acc = _accumulate_cross(acc, counts)
    mean_counts = _finalize_cross_counts(acc, meta=meta, nreal=int(nreal))
    return mean_counts, {"t1": tt1, "t2": tt2}


def _cross_mc_result_for_selection(**kwargs):
    counts, ctx = _cross_mc_mean_counts_for_selection(**kwargs)
    config = kwargs["config"]
    t1 = ctx["t1"]
    t2 = ctx["t2"]
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and (not t1["wunit"] or not t2["wunit"]))
    result = estimate_cross(counts, estimator=config.estimator, sum_w1=(float(np.sum(t1["weights"])) if weighted else None), sum_w2=(float(np.sum(t2["weights"])) if weighted else None))
    return result


def _auto_mc_fast_bootstrap_counts(
    *,
    data_template: dict[str, Any],
    rand_template: dict[str, Any],
    cdf_data: np.ndarray,
    pbar_data: np.ndarray,
    cdf_random: np.ndarray | None,
    chi_grid: np.ndarray,
    sample_edges_chi: np.ndarray | None,
    config: ProjectedAutoConfig,
    meta: dict[str, Any],
    sbound,
    grid_meta,
    data_grid,
    rand_grid,
    nreal: int,
    base_seed: int,
    status_label: str = "bootstrap-fast-auto",
) -> ProjectedAutoCounts:
    mode = str(config.mc_pdf.random_mode).strip().lower()
    fixed_global_random = mode == "fixed_global"
    acc = None
    fixed_rand_p = None
    shared_status_emitter = None
    if bool(getattr(config.progress, "enabled", False)) and in_notebook():
        shared_status_emitter = create_status_emitter(
            notebook=True,
            min_update_interval=float(getattr(config.progress, "poll_interval", 0.15)),
        )

    for m in range(int(nreal)):
        status_prefix = f"[pcf:mc_pdf:bootstrap] realization {m + 1}/{int(nreal)}  "
        rng = _child_rng(base_seed, status_label, m)
        data_dist = _sample_dist_from_cdf(cdf_data, chi_grid, rng, sample_edges_chi=sample_edges_chi)
        data_p = _build_prepared(data_template, data_dist, sbound=sbound, grid_tuple=data_grid, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

        if fixed_global_random and fixed_rand_p is not None:
            rand_p = fixed_rand_p
        else:
            rand_dist = _sample_random_dist(
                rand_template["nrows"],
                random_source_cdf=cdf_random,
                data_source_mean=pbar_data,
                data_draw=(data_dist if mode == "inherit_realization" else None),
                chi_grid=chi_grid,
                random_mode=config.mc_pdf.random_mode,
                rng=rng,
                sample_edges_chi=sample_edges_chi,
            )
            rand_p = _build_prepared(rand_template, rand_dist, sbound=sbound, grid_tuple=rand_grid, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
            if fixed_global_random:
                fixed_rand_p = rand_p

        counts = _run_mc_progress(config, lambda progress_path: build_auto_counts(
            data_p,
            rand_p,
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            pi_delta=meta["pi_delta"],
            nthreads=config.nthreads,
            estimator=config.estimator,
            weight_mode=config.weights.weight_mode,
            doboot=True,
            dojk=False,
            nreg=0,
            nbts=int(config.bootstrap.nbts),
            bseed=int(config.bootstrap.bseed),
            progress_file=progress_path,
            split_random=config.split_random,
        ), status_prefix=status_prefix, status_emitter=shared_status_emitter)
        acc = _accumulate_auto_bootstrap(acc, counts)

    if shared_status_emitter is not None:
        shared_status_emitter.close()
    return _finalize_auto_bootstrap_counts(acc, meta=meta, nreal=int(nreal))


def _cross_mc_fast_bootstrap_counts(
    *,
    t1: dict[str, Any],
    t2: dict[str, Any],
    tr1: dict[str, Any] | None,
    tr2: dict[str, Any] | None,
    cdf1: np.ndarray | None,
    cdf2: np.ndarray | None,
    pbar1: np.ndarray | None,
    pbar2: np.ndarray | None,
    cdf_r1: np.ndarray | None,
    cdf_r2: np.ndarray | None,
    data1,
    data2,
    random1,
    random2,
    chi_grid: np.ndarray,
    sample_edges_chi: np.ndarray | None,
    config: ProjectedCrossConfig,
    meta: dict[str, Any],
    sbound,
    grid_meta,
    g1,
    g2,
    gr1,
    gr2,
    nreal: int,
    base_seed: int,
    status_label: str = "bootstrap-fast-cross",
) -> ProjectedCrossCounts:
    mode = str(config.mc_pdf.random_mode).strip().lower()
    fixed_global_random = mode == "fixed_global"
    fixed_prep_r1 = None
    fixed_prep_r2 = None
    acc = None
    shared_status_emitter = None
    if bool(getattr(config.progress, "enabled", False)) and in_notebook():
        shared_status_emitter = create_status_emitter(
            notebook=True,
            min_update_interval=float(getattr(config.progress, "poll_interval", 0.15)),
        )

    for m in range(int(nreal)):
        status_prefix = f"[pccf:mc_pdf:bootstrap] realization {m + 1}/{int(nreal)}  "
        rng = _child_rng(base_seed, status_label, m)
        if cdf1 is None:
            dist1 = _distance_array(data1, config.columns_data1, config)
        else:
            dist1 = _sample_dist_from_cdf(cdf1, chi_grid, rng, sample_edges_chi=sample_edges_chi)
        if cdf2 is None:
            dist2 = _distance_array(data2, config.columns_data2, config)
        else:
            dist2 = _sample_dist_from_cdf(cdf2, chi_grid, rng, sample_edges_chi=sample_edges_chi)

        prep1 = _build_prepared(t1, dist1, sbound=sbound, grid_tuple=g1, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
        prep2 = _build_prepared(t2, dist2, sbound=sbound, grid_tuple=g2, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

        if tr1 is None:
            prep_r1 = None
        elif fixed_global_random and fixed_prep_r1 is not None:
            prep_r1 = fixed_prep_r1
        else:
            if cdf_r1 is not None:
                rdist1 = _sample_dist_from_cdf(cdf_r1, chi_grid, rng, sample_edges_chi=sample_edges_chi)
            elif pbar1 is None:
                rdist1 = _distance_array(random1, config.columns_random1, config)
            else:
                rdist1 = _sample_random_dist(
                    tr1["nrows"], random_source_cdf=None, data_source_mean=pbar1,
                    data_draw=(dist1 if mode == "inherit_realization" else None),
                    chi_grid=chi_grid, random_mode=config.mc_pdf.random_mode, rng=rng,
                    sample_edges_chi=sample_edges_chi,
                )
            prep_r1 = _build_prepared(tr1, rdist1, sbound=sbound, grid_tuple=gr1, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
            if fixed_global_random:
                fixed_prep_r1 = prep_r1

        if tr2 is None:
            prep_r2 = None
        elif fixed_global_random and fixed_prep_r2 is not None:
            prep_r2 = fixed_prep_r2
        else:
            if cdf_r2 is not None:
                rdist2 = _sample_dist_from_cdf(cdf_r2, chi_grid, rng, sample_edges_chi=sample_edges_chi)
            elif pbar2 is None:
                rdist2 = _distance_array(random2, config.columns_random2, config)
            else:
                rdist2 = _sample_random_dist(
                    tr2["nrows"], random_source_cdf=None, data_source_mean=pbar2,
                    data_draw=(dist2 if mode == "inherit_realization" else None),
                    chi_grid=chi_grid, random_mode=config.mc_pdf.random_mode, rng=rng,
                    sample_edges_chi=sample_edges_chi,
                )
            prep_r2 = _build_prepared(tr2, rdist2, sbound=sbound, grid_tuple=gr2, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
            if fixed_global_random:
                fixed_prep_r2 = prep_r2

        counts = _run_mc_progress(config, lambda progress_path: build_cross_counts(
            prep1, prep_r1, prep2, prep_r2,
            rp_edges=meta["rp_edges"], rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"], pi_centers=meta["pi_centers"], pi_delta=meta["pi_delta"],
            nthreads=config.nthreads, estimator=config.estimator, weight_mode=config.weights.weight_mode,
            doboot=True, dojk=False, nreg=0, nbts=int(config.bootstrap.nbts),
            bseed=int(config.bootstrap.bseed), primary=config.bootstrap.primary, progress_file=progress_path,
        ), status_prefix=status_prefix, status_emitter=shared_status_emitter)
        acc = _accumulate_cross_bootstrap(acc, counts)

    if shared_status_emitter is not None:
        shared_status_emitter.close()
    return _finalize_cross_bootstrap_counts(acc, meta=meta, nreal=int(nreal))


def _auto_mc_fast_jackknife_counts(
    *,
    data_template: dict[str, Any],
    rand_template: dict[str, Any],
    cdf_data: np.ndarray,
    pbar_data: np.ndarray,
    cdf_random: np.ndarray | None,
    chi_grid: np.ndarray,
    sample_edges_chi: np.ndarray | None,
    config: ProjectedAutoConfig,
    meta: dict[str, Any],
    sbound,
    grid_meta,
    data_grid,
    rand_grid,
    nreal: int,
    base_seed: int,
    status_label: str = "jackknife-fast-auto",
) -> ProjectedAutoCounts:
    mode = str(config.mc_pdf.random_mode).strip().lower()
    fixed_global_random = mode == "fixed_global"
    acc = None
    fixed_rand_p = None
    nregions = int(meta.get("jk_nregions") or 0)
    shared_status_emitter = None
    if bool(getattr(config.progress, "enabled", False)) and in_notebook():
        shared_status_emitter = create_status_emitter(
            notebook=True,
            min_update_interval=float(getattr(config.progress, "poll_interval", 0.15)),
        )

    try:
        for m in range(int(nreal)):
            status_prefix = f"[pcf:mc_pdf:jackknife] realization {m + 1}/{int(nreal)}  "
            rng = _child_rng(base_seed, status_label, m)
            data_dist = _sample_dist_from_cdf(cdf_data, chi_grid, rng, sample_edges_chi=sample_edges_chi)
            data_p = _build_prepared(data_template, data_dist, sbound=sbound, grid_tuple=data_grid, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

            if fixed_global_random and fixed_rand_p is not None:
                rand_p = fixed_rand_p
            else:
                rand_dist = _sample_random_dist(
                    rand_template["nrows"],
                    random_source_cdf=cdf_random,
                    data_source_mean=pbar_data,
                    data_draw=(data_dist if mode == "inherit_realization" else None),
                    chi_grid=chi_grid,
                    random_mode=config.mc_pdf.random_mode,
                    rng=rng,
                    sample_edges_chi=sample_edges_chi,
                )
                rand_p = _build_prepared(rand_template, rand_dist, sbound=sbound, grid_tuple=rand_grid, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
                if fixed_global_random:
                    fixed_rand_p = rand_p

            counts = _run_mc_progress(config, lambda progress_path: build_auto_counts(
                data_p,
                rand_p,
                rp_edges=meta["rp_edges"],
                rp_centers=meta["rp_centers"],
                pi_edges=meta["pi_edges"],
                pi_centers=meta["pi_centers"],
                pi_delta=meta["pi_delta"],
                nthreads=config.nthreads,
                estimator=config.estimator,
                weight_mode=config.weights.weight_mode,
                doboot=False,
                dojk=True,
                nreg=nregions,
                nbts=0,
                bseed=int(config.bootstrap.bseed),
                progress_file=progress_path,
                split_random=config.split_random,
            ), status_prefix=status_prefix, status_emitter=shared_status_emitter)
            acc = _accumulate_auto_jackknife(acc, counts)
    finally:
        if shared_status_emitter is not None:
            shared_status_emitter.close()

    return _finalize_auto_jackknife_counts(acc, meta=meta, nreal=int(nreal))


def _cross_mc_fast_jackknife_counts(
    *,
    t1: dict[str, Any],
    t2: dict[str, Any],
    tr1: dict[str, Any] | None,
    tr2: dict[str, Any] | None,
    cdf1: np.ndarray | None,
    cdf2: np.ndarray | None,
    pbar1: np.ndarray | None,
    pbar2: np.ndarray | None,
    cdf_r1: np.ndarray | None,
    cdf_r2: np.ndarray | None,
    data1,
    data2,
    random1,
    random2,
    chi_grid: np.ndarray,
    sample_edges_chi: np.ndarray | None,
    config: ProjectedCrossConfig,
    meta: dict[str, Any],
    sbound,
    grid_meta,
    g1,
    g2,
    gr1,
    gr2,
    nreal: int,
    base_seed: int,
    status_label: str = "jackknife-fast-cross",
) -> ProjectedCrossCounts:
    mode = str(config.mc_pdf.random_mode).strip().lower()
    fixed_global_random = mode == "fixed_global"
    fixed_prep_r1 = None
    fixed_prep_r2 = None
    acc = None
    nregions = int(meta.get("jk_nregions") or 0)
    shared_status_emitter = None
    if bool(getattr(config.progress, "enabled", False)) and in_notebook():
        shared_status_emitter = create_status_emitter(
            notebook=True,
            min_update_interval=float(getattr(config.progress, "poll_interval", 0.15)),
        )

    try:
        for m in range(int(nreal)):
            status_prefix = f"[pccf:mc_pdf:jackknife] realization {m + 1}/{int(nreal)}  "
            rng = _child_rng(base_seed, status_label, m)
            if cdf1 is None:
                dist1 = _distance_array(data1, config.columns_data1, config)
            else:
                dist1 = _sample_dist_from_cdf(cdf1, chi_grid, rng, sample_edges_chi=sample_edges_chi)
            if cdf2 is None:
                dist2 = _distance_array(data2, config.columns_data2, config)
            else:
                dist2 = _sample_dist_from_cdf(cdf2, chi_grid, rng, sample_edges_chi=sample_edges_chi)

            prep1 = _build_prepared(t1, dist1, sbound=sbound, grid_tuple=g1, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
            prep2 = _build_prepared(t2, dist2, sbound=sbound, grid_tuple=g2, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

            if tr1 is None:
                prep_r1 = None
            elif fixed_global_random and fixed_prep_r1 is not None:
                prep_r1 = fixed_prep_r1
            else:
                if cdf_r1 is not None:
                    rdist1 = _sample_dist_from_cdf(cdf_r1, chi_grid, rng, sample_edges_chi=sample_edges_chi)
                elif pbar1 is None:
                    rdist1 = _distance_array(random1, config.columns_random1, config)
                else:
                    rdist1 = _sample_random_dist(
                        tr1["nrows"], random_source_cdf=None, data_source_mean=pbar1,
                        data_draw=(dist1 if mode == "inherit_realization" else None),
                        chi_grid=chi_grid, random_mode=config.mc_pdf.random_mode, rng=rng,
                        sample_edges_chi=sample_edges_chi,
                    )
                prep_r1 = _build_prepared(tr1, rdist1, sbound=sbound, grid_tuple=gr1, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
                if fixed_global_random:
                    fixed_prep_r1 = prep_r1

            if tr2 is None:
                prep_r2 = None
            elif fixed_global_random and fixed_prep_r2 is not None:
                prep_r2 = fixed_prep_r2
            else:
                if cdf_r2 is not None:
                    rdist2 = _sample_dist_from_cdf(cdf_r2, chi_grid, rng, sample_edges_chi=sample_edges_chi)
                elif pbar2 is None:
                    rdist2 = _distance_array(random2, config.columns_random2, config)
                else:
                    rdist2 = _sample_random_dist(
                        tr2["nrows"], random_source_cdf=None, data_source_mean=pbar2,
                        data_draw=(dist2 if mode == "inherit_realization" else None),
                        chi_grid=chi_grid, random_mode=config.mc_pdf.random_mode, rng=rng,
                        sample_edges_chi=sample_edges_chi,
                    )
                prep_r2 = _build_prepared(tr2, rdist2, sbound=sbound, grid_tuple=gr2, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
                if fixed_global_random:
                    fixed_prep_r2 = prep_r2

            counts = _run_mc_progress(config, lambda progress_path: build_cross_counts(
                prep1, prep_r1, prep2, prep_r2,
                rp_edges=meta["rp_edges"], rp_centers=meta["rp_centers"],
                pi_edges=meta["pi_edges"], pi_centers=meta["pi_centers"], pi_delta=meta["pi_delta"],
                nthreads=config.nthreads, estimator=config.estimator, weight_mode=config.weights.weight_mode,
                doboot=False, dojk=True, nreg=nregions, nbts=0,
                bseed=int(config.bootstrap.bseed), primary=config.bootstrap.primary, progress_file=progress_path,
            ), status_prefix=status_prefix, status_emitter=shared_status_emitter)
            acc = _accumulate_cross_jackknife(acc, counts)
    finally:
        if shared_status_emitter is not None:
            shared_status_emitter.close()

    return _finalize_cross_jackknife_counts(acc, meta=meta, nreal=int(nreal))


def _mc_pdf_enabled(config) -> bool:
    return bool(getattr(getattr(config, "mc_pdf", None), "enabled", False))


def _validate_mc_pdf_config(config, *, cross: bool) -> None:
    if bool(getattr(getattr(config, "pdf", None), "enabled", False)):
        raise ValueError("Choose either config.pdf.enabled or config.mc_pdf.enabled, not both.")
    if (bool(getattr(config.bootstrap, "enabled", False)) or bool(getattr(config.jackknife, "enabled", False))) and bool(getattr(getattr(config, "split_random", None), "enabled", False)):
        raise NotImplementedError("mc_pdf bootstrap/jackknife currently does not support split_random; disable split_random for MC resampling runs.")
    nreal = int(getattr(config.mc_pdf, "nreal", 0))
    if nreal <= 0:
        raise ValueError("mc_pdf.nreal must be a positive integer.")
    if not cross and getattr(config.mc_pdf, "pdf_data", None) is None:
        raise ValueError("mc_pdf auto mode requires mc_pdf.pdf_data.")
    if cross and getattr(config.mc_pdf, "pdf_data1", None) is None and getattr(config.mc_pdf, "pdf_data2", None) is None:
        raise ValueError("mc_pdf cross mode requires at least one of mc_pdf.pdf_data1 or mc_pdf.pdf_data2.")
    # Validate resampling backend/policy early so incompatible fast-bootstrap
    # requests fail before the expensive full-sample MC run starts.
    _mc_resampling_backend(config.mc_pdf)
    _mc_resampling_random_policy(config.mc_pdf)
    if bool(getattr(config.bootstrap, "enabled", False)) and not bool(getattr(config.jackknife, "enabled", False)):
        if str(getattr(config.mc_pdf, "resampling_backend", "auto")).strip().lower() == "fast":
            _mc_fast_bootstrap_enabled(config)
    if bool(getattr(config.jackknife, "enabled", False)):
        if str(getattr(config.mc_pdf, "resampling_backend", "auto")).strip().lower() == "fast":
            _mc_fast_jackknife_enabled(config)


def run_auto_mc_pdf(data, random, config: ProjectedAutoConfig):
    _validate_mc_pdf_config(config, cross=False)
    spec = config.mc_pdf
    chi_grid = resolve_common_chi_grid(z_grid=spec.z_grid, chi_grid=spec.chi_grid, config=config, grid_kind=str(getattr(spec, "grid_kind", "centers")), label="mc_pdf")
    sample_edges_chi = _resolve_sample_edges_chi(spec, config)
    if chi_grid.ndim != 1:
        raise ValueError("mc_pdf common grid must be one-dimensional.")
    if chi_grid.size == 0:
        raise ValueError("mc_pdf common grid cannot be empty.")
    support_grid = sample_edges_chi if sample_edges_chi is not None else chi_grid
    meta = _auto_meta(config, data, random, spec.pdf_data, spec.pdf_random, support_grid)
    sbound = meta["sbound"]
    grid_meta = _grid_meta(config)

    data_template = _catalog_template(data, config.columns_data, config, use_weights=(config.weights.weight_mode != "unweighted"))
    rand_template = _catalog_template(random, config.columns_random, config, use_weights=False)
    if bool(config.jackknife.enabled):
        data_region, random_region, jk_meta = _auto_jackknife_regions(data, random, config)
        data_template = _set_template_region(data_template, data_region)
        rand_template = _set_template_region(rand_template, random_region)
        meta.update(jk_meta)
    data_grid = _grid_tuple(data_template, config, sbound=sbound)
    rand_grid = _grid_tuple(rand_template, config, sbound=sbound)

    p_data = _load_pdf_matrix(spec.pdf_data, data, nrows=data_template["nrows"])
    cdf_data = _build_cdf(p_data)
    pbar_data = _mean_pdf(p_data)

    p_random = None if spec.pdf_random is None else _load_pdf_matrix(spec.pdf_random, random, nrows=rand_template["nrows"])
    cdf_random = None if p_random is None else _build_cdf(p_random)

    rng = np.random.default_rng(int(spec.seed))
    fixed_random = str(spec.random_mode).strip().lower() == "fixed_global"
    fixed_rand_p = None
    fixed_rr = None
    fixed_rr_meta = None
    acc = None
    wp_real = []
    shared_status_emitter = None
    if bool(getattr(config.progress, "enabled", False)) and in_notebook():
        shared_status_emitter = create_status_emitter(
            notebook=True,
            min_update_interval=float(getattr(config.progress, "poll_interval", 0.15)),
        )

    try:
        for ireal in range(int(spec.nreal)):
            status_prefix = f"[pcf:mc_pdf] realization {ireal + 1}/{int(spec.nreal)}  "
            data_dist = _sample_dist_from_cdf(cdf_data, chi_grid, rng, sample_edges_chi=sample_edges_chi)
            data_p = _build_prepared(data_template, data_dist, sbound=sbound, grid_tuple=data_grid, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

            if fixed_random:
                if fixed_rand_p is None:
                    rand_dist = _sample_random_dist(rand_template["nrows"], random_source_cdf=cdf_random, data_source_mean=pbar_data, data_draw=data_dist, chi_grid=chi_grid, random_mode=spec.random_mode, rng=rng, sample_edges_chi=sample_edges_chi)
                    fixed_rand_p = _build_prepared(rand_template, rand_dist, sbound=sbound, grid_tuple=rand_grid, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
                rand_p = fixed_rand_p
            else:
                rand_dist = _sample_random_dist(rand_template["nrows"], random_source_cdf=cdf_random, data_source_mean=pbar_data, data_draw=data_dist, chi_grid=chi_grid, random_mode=spec.random_mode, rng=rng, sample_edges_chi=sample_edges_chi)
                rand_p = _build_prepared(rand_template, rand_dist, sbound=sbound, grid_tuple=rand_grid, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

            if fixed_random:
                if fixed_rr is None:
                    counts = _run_mc_progress(config, lambda progress_path: build_auto_counts(
                        data_p,
                        rand_p,
                        rp_edges=meta["rp_edges"],
                        rp_centers=meta["rp_centers"],
                        pi_edges=meta["pi_edges"],
                        pi_centers=meta["pi_centers"],
                        pi_delta=meta["pi_delta"],
                        nthreads=config.nthreads,
                        estimator=config.estimator,
                        weight_mode=config.weights.weight_mode,
                        doboot=False,
                        dojk=False,
                        nreg=0,
                        nbts=0,
                        bseed=config.bootstrap.bseed,
                        progress_file=progress_path,
                        split_random=config.split_random,
                    ), status_prefix=status_prefix, status_emitter=shared_status_emitter)
                    fixed_rr = None if counts.rr is None else np.asarray(counts.rr, dtype=np.float64).copy()
                    fixed_rr_meta = dict(counts.metadata)
                else:
                    dd_res = _run_mc_progress(config, lambda progress_path: build_auto_count_result(
                        data_p,
                        rp_edges=meta["rp_edges"],
                        rp_centers=meta["rp_centers"],
                        pi_edges=meta["pi_edges"],
                        pi_centers=meta["pi_centers"],
                        pi_delta=meta["pi_delta"],
                        nthreads=config.nthreads,
                        weight_mode=config.weights.weight_mode,
                        doboot=False,
                        dojk=False,
                        nreg=0,
                        nbts=0,
                        bseed=config.bootstrap.bseed,
                        progress_file=progress_path,
                    ), status_prefix=status_prefix, status_emitter=shared_status_emitter)
                    dr_arr = None
                    if str(config.estimator).upper() in {"DP", "LS"}:
                        dr_res = _run_mc_progress(config, lambda progress_path: build_cross_count_result(
                            data_p,
                            rand_p,
                            rp_edges=meta["rp_edges"],
                            rp_centers=meta["rp_centers"],
                            pi_edges=meta["pi_edges"],
                            pi_centers=meta["pi_centers"],
                            pi_delta=meta["pi_delta"],
                            nthreads=config.nthreads,
                            weight_mode=config.weights.weight_mode,
                            doboot=False,
                            dojk=False,
                            nreg=0,
                            nbts=0,
                            bseed=config.bootstrap.bseed,
                            primary="data1",
                            progress_file=progress_path,
                        ), status_prefix=status_prefix, status_emitter=shared_status_emitter)
                        dr_arr = np.asarray(dr_res.d1d2, dtype=np.float64)
                    counts = _assemble_auto_counts_from_terms(
                        dd_res,
                        fixed_rr if str(config.estimator).upper() in {"NAT", "LS"} else None,
                        dr_arr,
                        meta=meta,
                        rr_norm_pairs=float(fixed_rr_meta.get("rr_norm_pairs", 0.5 * rand_p.nrows * max(rand_p.nrows - 1, 0))),
                        extra_metadata={k: v for k, v in fixed_rr_meta.items() if k not in {"n_data", "data_weighted"}},
                    )
            else:
                counts = _run_mc_progress(config, lambda progress_path: build_auto_counts(
                    data_p,
                    rand_p,
                    rp_edges=meta["rp_edges"],
                    rp_centers=meta["rp_centers"],
                    pi_edges=meta["pi_edges"],
                    pi_centers=meta["pi_centers"],
                    pi_delta=meta["pi_delta"],
                    nthreads=config.nthreads,
                    estimator=config.estimator,
                    weight_mode=config.weights.weight_mode,
                    doboot=False,
                    dojk=False,
                    nreg=0,
                    nbts=0,
                    bseed=config.bootstrap.bseed,
                    progress_file=progress_path,
                    split_random=config.split_random,
                ), status_prefix=status_prefix, status_emitter=shared_status_emitter)

            acc = _accumulate_auto(acc, counts)
            if bool(spec.store_realizations):
                weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and not data_p.wunit)
                res_i = estimate_auto(counts, estimator=config.estimator, data_weights=(data_p.weights if weighted else None))
                wp_real.append(np.asarray(res_i.wp, dtype=np.float64))
    finally:
        if shared_status_emitter is not None:
            shared_status_emitter.close()

    mean_counts = _finalize_auto_counts(acc, meta=meta, nreal=int(spec.nreal))
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and not data_template["wunit"])
    result = estimate_auto(mean_counts, estimator=config.estimator, data_weights=(data_template["weights"] if weighted else None))
    if wp_real:
        arr = np.asarray(wp_real, dtype=np.float64)
        result.mc_realizations = arr
        result.mc_wp_std = np.std(arr, axis=0)
    result.metadata.update({
        "mc_pdf": True,
        "mc_pdf_mode": "grid_sampler_within_bin" if sample_edges_chi is not None else "grid_sampler",
        "mc_sample_within_bin": bool(sample_edges_chi is not None),
        "mc_grid_kind": str(getattr(spec, "grid_kind", "centers")),
        "mc_nreal": int(spec.nreal),
        "mc_resampling_nreal": int(_resampling_nreal(spec)) if (config.bootstrap.enabled or config.jackknife.enabled) else None,
        "mc_random_mode": str(spec.random_mode),
        "mc_rr_fixed": bool(fixed_random),
    })

    fixed_rand_dist_resampling = None
    if fixed_random:
        rng_fixed = _child_rng(int(spec.seed), "auto", "fixed-random-resampling")
        fixed_rand_dist_resampling = _sample_random_dist(
            rand_template["nrows"],
            random_source_cdf=cdf_random,
            data_source_mean=pbar_data,
            data_draw=None,
            chi_grid=chi_grid,
            random_mode=spec.random_mode,
            rng=rng_fixed,
            sample_edges_chi=sample_edges_chi,
        )

    if bool(config.bootstrap.enabled):
        nres_mc = _resampling_nreal(spec)
        if _mc_fast_bootstrap_enabled(config):
            _stage(config.progress.enabled, "[pcf:mc_pdf] assembling fast bootstrap covariance")
            boot_counts = _auto_mc_fast_bootstrap_counts(
                data_template=data_template,
                rand_template=rand_template,
                cdf_data=cdf_data,
                pbar_data=pbar_data,
                cdf_random=cdf_random,
                chi_grid=chi_grid,
                sample_edges_chi=sample_edges_chi,
                config=config,
                meta=meta,
                sbound=sbound,
                grid_meta=grid_meta,
                data_grid=data_grid,
                rand_grid=rand_grid,
                nreal=nres_mc,
                base_seed=int(spec.seed),
            )
            weighted_boot = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and not data_template["wunit"])
            boot, boot_cumulative = _bootstrap_products_from_auto_counts(
                boot_counts,
                estimator=config.estimator,
                data_weights=(data_template["weights"] if weighted_boot else None),
                store_cumulative=bool(getattr(config.bootstrap, "store_cumulative", True)),
            )
            result = _apply_bootstrap_result(
                result,
                boot,
                cumulative_realizations=boot_cumulative,
                backend="mc_pdf_fast",
            )
            if bool(getattr(config.bootstrap, "store_counts", True)):
                result.bootstrap_counts = boot_counts
            result.metadata["mc_resampling_backend"] = "fast"
            result.metadata["mc_resampling_random_policy"] = "fixed" if str(spec.random_mode).strip().lower() == "inherit_realization" else str(spec.random_mode)
            result.metadata["mc_fast_bootstrap_count_average"] = True
        else:
            _stage(config.progress.enabled, "[pcf:mc_pdf] assembling bootstrap covariance")
            nbts = int(config.bootstrap.nbts)
            rng_boot = np.random.default_rng(int(config.bootstrap.bseed))
            boot = np.zeros((nbts, len(meta["rp_centers"])), dtype=np.float64)
            boot_cumulative = None
            if bool(getattr(config.bootstrap, "store_cumulative", True)):
                boot_cumulative = np.zeros(
                    (nbts, len(meta["rp_centers"]), len(meta["pi_centers"])),
                    dtype=np.float64,
                )
            for ib in range(nbts):
                draw = rng_boot.integers(0, int(data_template["nrows"]), size=int(data_template["nrows"]), dtype=np.int64)
                res_b = _auto_mc_result_for_selection(
                    data_template=data_template,
                    rand_template=rand_template,
                    cdf_data=cdf_data,
                    pbar_data=pbar_data,
                    cdf_random=cdf_random,
                    chi_grid=chi_grid,
                    sample_edges_chi=sample_edges_chi,
                    config=config,
                    meta=meta,
                    sbound=sbound,
                    grid_meta=grid_meta,
                    data_grid=data_grid,
                    rand_grid=rand_grid,
                    nreal=nres_mc,
                    base_seed=int(spec.seed),
                    data_index=draw,
                    rand_index=None,
                    fixed_rand_dist=fixed_rand_dist_resampling,
                    status_label=f"bootstrap-auto-{ib}",
                )
                boot[ib] = np.asarray(res_b.wp, dtype=np.float64)
                if boot_cumulative is not None:
                    boot_cumulative[ib] = compute_auto_cumulative_wp(
                        res_b.counts,
                        estimator=res_b.estimator,
                        sum_w_data=res_b.metadata.get("sum_w_data"),
                        sum_w2_data=res_b.metadata.get("sum_w2_data"),
                    )
            result = _apply_bootstrap_result(
                result,
                boot,
                cumulative_realizations=boot_cumulative,
                backend="mc_pdf_rerun",
            )
            result.metadata["mc_resampling_backend"] = "rerun"
            result.metadata["mc_resampling_random_policy"] = "reinherit" if str(spec.random_mode).strip().lower() == "inherit_realization" else str(spec.random_mode)

    if bool(config.jackknife.enabled):
        nregions = int(meta.get("jk_nregions") or 0)
        nres_mc = _resampling_nreal(spec)
        if _mc_fast_jackknife_enabled(config):
            _stage(config.progress.enabled, "[pcf:mc_pdf] assembling fast jackknife covariance")
            jk_counts = _auto_mc_fast_jackknife_counts(
                data_template=data_template,
                rand_template=rand_template,
                cdf_data=cdf_data,
                pbar_data=pbar_data,
                cdf_random=cdf_random,
                chi_grid=chi_grid,
                sample_edges_chi=sample_edges_chi,
                config=config,
                meta=meta,
                sbound=sbound,
                grid_meta=grid_meta,
                data_grid=data_grid,
                rand_grid=rand_grid,
                nreal=nres_mc,
                base_seed=int(spec.seed),
            )
            jk = _jackknife_wp_from_auto_touch(jk_counts, data_template, rand_template, config, meta)
            cov = jackknife_cov(jk)
            result.wp_err = np.sqrt(np.diag(cov))
            result.cov = cov if config.jackknife.return_cov else None
            result.realizations = jk if config.jackknife.return_realizations else None
            result.metadata.update({
                "jackknife": True,
                "jk_nregions": int(jk.shape[0]),
                "jk_region_source": meta.get("jk_region_source"),
                "jk_touch_fast": True,
                "mc_resampling_backend": "fast",
                "mc_resampling_random_policy": "fixed" if str(spec.random_mode).strip().lower() == "inherit_realization" else str(spec.random_mode),
                "mc_fast_jackknife_count_average": True,
            })
        else:
            _stage(config.progress.enabled, "[pcf:mc_pdf] assembling jackknife covariance")
            jk = np.zeros((nregions, len(meta["rp_centers"])), dtype=np.float64)
            data_reg = np.asarray(data_template.get("region_id"), dtype=np.int32)
            rand_reg = np.asarray(rand_template.get("region_id"), dtype=np.int32)
            for k in range(nregions):
                didx = np.flatnonzero(data_reg != k).astype(np.int64)
                ridx = np.flatnonzero(rand_reg != k).astype(np.int64)
                res_k = _auto_mc_result_for_selection(
                    data_template=data_template,
                    rand_template=rand_template,
                    cdf_data=cdf_data,
                    pbar_data=pbar_data,
                    cdf_random=cdf_random,
                    chi_grid=chi_grid,
                    sample_edges_chi=sample_edges_chi,
                    config=config,
                    meta=meta,
                    sbound=sbound,
                    grid_meta=grid_meta,
                    data_grid=data_grid,
                    rand_grid=rand_grid,
                    nreal=nres_mc,
                    base_seed=int(spec.seed),
                    data_index=didx,
                    rand_index=ridx,
                    fixed_rand_dist=fixed_rand_dist_resampling,
                    status_label=f"jackknife-auto-{k}",
                )
                jk[k] = np.asarray(res_k.wp, dtype=np.float64)
            cov = jackknife_cov(jk)
            result.wp_err = np.sqrt(np.diag(cov))
            result.cov = cov if config.jackknife.return_cov else None
            result.realizations = jk if config.jackknife.return_realizations else None
            result.metadata.update({
                "jackknife": True,
                "jk_nregions": int(jk.shape[0]),
                "jk_region_source": meta.get("jk_region_source"),
                "jk_touch_fast": False,
                "mc_resampling_backend": "rerun",
                "mc_resampling_random_policy": "reinherit" if str(spec.random_mode).strip().lower() == "inherit_realization" else str(spec.random_mode),
            })
    result = apply_bootstrap_storage_policy(result, config.bootstrap)
    _stage(config.progress.enabled, "[pcf:mc_pdf] done")
    return attach_roundtrip_context(result, config=config, provenance=provenance_dict("pcf"), extra_metadata=meta)


def run_cross_mc_pdf(data1, data2, config: ProjectedCrossConfig, *, random1=None, random2=None):
    _validate_mc_pdf_config(config, cross=True)
    spec = config.mc_pdf
    chi_grid = resolve_common_chi_grid(z_grid=spec.z_grid, chi_grid=spec.chi_grid, config=config, grid_kind=str(getattr(spec, "grid_kind", "centers")), label="mc_pdf")
    sample_edges_chi = _resolve_sample_edges_chi(spec, config)
    if chi_grid.ndim != 1:
        raise ValueError("mc_pdf common grid must be one-dimensional.")
    if chi_grid.size == 0:
        raise ValueError("mc_pdf common grid cannot be empty.")
    support_grid = sample_edges_chi if sample_edges_chi is not None else chi_grid
    meta = _cross_meta(config, data1, random1, data2, random2, spec.pdf_data1, spec.pdf_random1, spec.pdf_data2, spec.pdf_random2, support_grid)
    sbound = meta["sbound"]
    grid_meta = _grid_meta(config)

    t1 = _catalog_template(data1, config.columns_data1, config, use_weights=(config.weights.weight_mode != "unweighted"))
    t2 = _catalog_template(data2, config.columns_data2, config, use_weights=(config.weights.weight_mode != "unweighted"))
    tr1 = None if random1 is None else _catalog_template(random1, config.columns_random1, config, use_weights=False)
    tr2 = None if random2 is None else _catalog_template(random2, config.columns_random2, config, use_weights=False)
    if bool(config.jackknife.enabled):
        d1_region, r1_region, d2_region, r2_region, jk_meta = _cross_jackknife_regions(data1, random1, data2, random2, config)
        t1 = _set_template_region(t1, d1_region)
        t2 = _set_template_region(t2, d2_region)
        if tr1 is not None:
            tr1 = _set_template_region(tr1, r1_region)
        if tr2 is not None:
            tr2 = _set_template_region(tr2, r2_region)
        meta.update(jk_meta)

    g1 = _grid_tuple(t1, config, sbound=sbound)
    g2 = _grid_tuple(t2, config, sbound=sbound)
    gr1 = None if tr1 is None else _grid_tuple(tr1, config, sbound=sbound)
    gr2 = None if tr2 is None else _grid_tuple(tr2, config, sbound=sbound)

    p1 = None if spec.pdf_data1 is None else _load_pdf_matrix(spec.pdf_data1, data1, nrows=t1["nrows"])
    p2 = None if spec.pdf_data2 is None else _load_pdf_matrix(spec.pdf_data2, data2, nrows=t2["nrows"])
    cdf1 = None if p1 is None else _build_cdf(p1)
    cdf2 = None if p2 is None else _build_cdf(p2)
    pbar1 = None if p1 is None else _mean_pdf(p1)
    pbar2 = None if p2 is None else _mean_pdf(p2)

    pr1 = None if spec.pdf_random1 is None or random1 is None else _load_pdf_matrix(spec.pdf_random1, random1, nrows=tr1["nrows"])
    pr2 = None if spec.pdf_random2 is None or random2 is None else _load_pdf_matrix(spec.pdf_random2, random2, nrows=tr2["nrows"])
    cdf_r1 = None if pr1 is None else _build_cdf(pr1)
    cdf_r2 = None if pr2 is None else _build_cdf(pr2)

    fixed_random = str(spec.random_mode).strip().lower() == "fixed_global"
    rng = np.random.default_rng(int(spec.seed))
    fixed_prep_r1 = None
    fixed_prep_r2 = None
    acc = None
    wp_real = []
    shared_status_emitter = None
    if bool(getattr(config.progress, "enabled", False)) and in_notebook():
        shared_status_emitter = create_status_emitter(
            notebook=True,
            min_update_interval=float(getattr(config.progress, "poll_interval", 0.15)),
        )

    try:
        for ireal in range(int(spec.nreal)):
            status_prefix = f"[pccf:mc_pdf] realization {ireal + 1}/{int(spec.nreal)}  "
            dist1 = _distance_array(data1, config.columns_data1, config) if cdf1 is None else _sample_dist_from_cdf(cdf1, chi_grid, rng, sample_edges_chi=sample_edges_chi)
            dist2 = _distance_array(data2, config.columns_data2, config) if cdf2 is None else _sample_dist_from_cdf(cdf2, chi_grid, rng, sample_edges_chi=sample_edges_chi)
            prep1 = _build_prepared(t1, dist1, sbound=sbound, grid_tuple=g1, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
            prep2 = _build_prepared(t2, dist2, sbound=sbound, grid_tuple=g2, pi_edges=meta["pi_edges"], grid_meta=grid_meta)

            if random1 is None:
                prep_r1 = None
            elif fixed_random and fixed_prep_r1 is not None:
                prep_r1 = fixed_prep_r1
            else:
                rdist1 = _distance_array(random1, config.columns_random1, config) if (cdf_r1 is None and pbar1 is None) else _sample_random_dist(tr1["nrows"], random_source_cdf=cdf_r1, data_source_mean=pbar1, data_draw=dist1 if cdf1 is not None else None, chi_grid=chi_grid, random_mode=spec.random_mode, rng=rng, sample_edges_chi=sample_edges_chi)
                prep_r1 = _build_prepared(tr1, rdist1, sbound=sbound, grid_tuple=gr1, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
                if fixed_random:
                    fixed_prep_r1 = prep_r1

            if random2 is None:
                prep_r2 = None
            elif fixed_random and fixed_prep_r2 is not None:
                prep_r2 = fixed_prep_r2
            else:
                rdist2 = _distance_array(random2, config.columns_random2, config) if (cdf_r2 is None and pbar2 is None) else _sample_random_dist(tr2["nrows"], random_source_cdf=cdf_r2, data_source_mean=pbar2, data_draw=dist2 if cdf2 is not None else None, chi_grid=chi_grid, random_mode=spec.random_mode, rng=rng, sample_edges_chi=sample_edges_chi)
                prep_r2 = _build_prepared(tr2, rdist2, sbound=sbound, grid_tuple=gr2, pi_edges=meta["pi_edges"], grid_meta=grid_meta)
                if fixed_random:
                    fixed_prep_r2 = prep_r2

            counts = _run_mc_progress(config, lambda progress_path: build_cross_counts(
                prep1,
                prep_r1,
                prep2,
                prep_r2,
                rp_edges=meta["rp_edges"],
                rp_centers=meta["rp_centers"],
                pi_edges=meta["pi_edges"],
                pi_centers=meta["pi_centers"],
                pi_delta=meta["pi_delta"],
                nthreads=config.nthreads,
                estimator=config.estimator,
                weight_mode=config.weights.weight_mode,
                doboot=False,
                dojk=False,
                nreg=0,
                nbts=0,
                bseed=config.bootstrap.bseed,
                primary=config.bootstrap.primary,
                progress_file=progress_path,
            ), status_prefix=status_prefix, status_emitter=shared_status_emitter)
            acc = _accumulate_cross(acc, counts)
            if bool(spec.store_realizations):
                weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and (not prep1.wunit or not prep2.wunit))
                res_i = estimate_cross(counts, estimator=config.estimator, sum_w1=(float(prep1.weights.sum()) if weighted else None), sum_w2=(float(prep2.weights.sum()) if weighted else None))
                wp_real.append(np.asarray(res_i.wp, dtype=np.float64))
    finally:
        if shared_status_emitter is not None:
            shared_status_emitter.close()

    mean_counts = _finalize_cross_counts(acc, meta=meta, nreal=int(spec.nreal))
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and (not t1["wunit"] or not t2["wunit"]))
    result = estimate_cross(mean_counts, estimator=config.estimator, sum_w1=(float(np.sum(t1["weights"])) if weighted else None), sum_w2=(float(np.sum(t2["weights"])) if weighted else None))
    if wp_real:
        arr = np.asarray(wp_real, dtype=np.float64)
        result.mc_realizations = arr
        result.mc_wp_std = np.std(arr, axis=0)
    result.metadata.update({
        "mc_pdf": True,
        "mc_pdf_mode": "grid_sampler_within_bin" if sample_edges_chi is not None else "grid_sampler",
        "mc_sample_within_bin": bool(sample_edges_chi is not None),
        "mc_grid_kind": str(getattr(spec, "grid_kind", "centers")),
        "mc_nreal": int(spec.nreal),
        "mc_resampling_nreal": int(_resampling_nreal(spec)) if (config.bootstrap.enabled or config.jackknife.enabled) else None,
        "mc_random_mode": str(spec.random_mode),
        "mc_rr_fixed": bool(fixed_random),
    })

    fixed_rdist1_resampling = None
    fixed_rdist2_resampling = None
    if fixed_random:
        rng_fixed = _child_rng(int(spec.seed), "cross", "fixed-random-resampling")
        if tr1 is not None:
            if cdf_r1 is not None:
                fixed_rdist1_resampling = _sample_dist_from_cdf(cdf_r1, chi_grid, rng_fixed, sample_edges_chi=sample_edges_chi)
            elif pbar1 is not None:
                fixed_rdist1_resampling = _sample_random_dist(tr1["nrows"], random_source_cdf=None, data_source_mean=pbar1, data_draw=None, chi_grid=chi_grid, random_mode=spec.random_mode, rng=rng_fixed, sample_edges_chi=sample_edges_chi)
            else:
                fixed_rdist1_resampling = _distance_array(random1, config.columns_random1, config)
        if tr2 is not None:
            if cdf_r2 is not None:
                fixed_rdist2_resampling = _sample_dist_from_cdf(cdf_r2, chi_grid, rng_fixed, sample_edges_chi=sample_edges_chi)
            elif pbar2 is not None:
                fixed_rdist2_resampling = _sample_random_dist(tr2["nrows"], random_source_cdf=None, data_source_mean=pbar2, data_draw=None, chi_grid=chi_grid, random_mode=spec.random_mode, rng=rng_fixed, sample_edges_chi=sample_edges_chi)
            else:
                fixed_rdist2_resampling = _distance_array(random2, config.columns_random2, config)

    if bool(config.bootstrap.enabled):
        nres_mc = _resampling_nreal(spec)
        if _mc_fast_bootstrap_enabled(config):
            _stage(config.progress.enabled, "[pccf:mc_pdf] assembling fast bootstrap covariance")
            boot_counts = _cross_mc_fast_bootstrap_counts(
                t1=t1, t2=t2, tr1=tr1, tr2=tr2,
                cdf1=cdf1, cdf2=cdf2, pbar1=pbar1, pbar2=pbar2, cdf_r1=cdf_r1, cdf_r2=cdf_r2,
                data1=data1, data2=data2, random1=random1, random2=random2,
                chi_grid=chi_grid, sample_edges_chi=sample_edges_chi, config=config, meta=meta, sbound=sbound, grid_meta=grid_meta,
                g1=g1, g2=g2, gr1=gr1, gr2=gr2, nreal=nres_mc, base_seed=int(spec.seed),
            )
            weighted_boot = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and (not t1["wunit"] or not t2["wunit"]))
            boot, boot_cumulative = _bootstrap_products_from_cross_counts(
                boot_counts,
                estimator=config.estimator,
                sum_w1=(float(np.sum(t1["weights"])) if weighted_boot else None),
                sum_w2=(float(np.sum(t2["weights"])) if weighted_boot else None),
                store_cumulative=bool(getattr(config.bootstrap, "store_cumulative", True)),
            )
            result = _apply_bootstrap_result(
                result,
                boot,
                cumulative_realizations=boot_cumulative,
                backend="mc_pdf_fast",
            )
            if bool(getattr(config.bootstrap, "store_counts", True)):
                result.bootstrap_counts = boot_counts
            result.metadata["mc_resampling_backend"] = "fast"
            result.metadata["mc_resampling_random_policy"] = "fixed" if str(spec.random_mode).strip().lower() == "inherit_realization" else str(spec.random_mode)
            result.metadata["mc_fast_bootstrap_count_average"] = True
        else:
            _stage(config.progress.enabled, "[pccf:mc_pdf] assembling bootstrap covariance")
            nbts = int(config.bootstrap.nbts)
            rng_boot = np.random.default_rng(int(config.bootstrap.bseed))
            boot = np.zeros((nbts, len(meta["rp_centers"])), dtype=np.float64)
            boot_cumulative = None
            if bool(getattr(config.bootstrap, "store_cumulative", True)):
                boot_cumulative = np.zeros(
                    (nbts, len(meta["rp_centers"]), len(meta["pi_centers"])),
                    dtype=np.float64,
                )
            primary = str(config.bootstrap.primary).strip().lower()
            for ib in range(nbts):
                idx1 = idx2 = None
                if primary == "data2":
                    idx2 = rng_boot.integers(0, int(t2["nrows"]), size=int(t2["nrows"]), dtype=np.int64)
                else:
                    idx1 = rng_boot.integers(0, int(t1["nrows"]), size=int(t1["nrows"]), dtype=np.int64)
                res_b = _cross_mc_result_for_selection(
                    t1=t1, t2=t2, tr1=tr1, tr2=tr2,
                    cdf1=cdf1, cdf2=cdf2, pbar1=pbar1, pbar2=pbar2, cdf_r1=cdf_r1, cdf_r2=cdf_r2,
                    data1=data1, data2=data2, random1=random1, random2=random2,
                    chi_grid=chi_grid, sample_edges_chi=sample_edges_chi, config=config, meta=meta, sbound=sbound, grid_meta=grid_meta,
                    g1=g1, g2=g2, gr1=gr1, gr2=gr2, nreal=nres_mc, base_seed=int(spec.seed),
                    idx1=idx1, idx2=idx2, ridx1=None, ridx2=None, fixed_rdist1=fixed_rdist1_resampling, fixed_rdist2=fixed_rdist2_resampling,
                    status_label=f"bootstrap-cross-{ib}",
                )
                boot[ib] = np.asarray(res_b.wp, dtype=np.float64)
                if boot_cumulative is not None:
                    boot_cumulative[ib] = compute_cross_cumulative_wp(
                        res_b.counts,
                        estimator=res_b.estimator,
                        sum_w1=res_b.metadata.get("sum_w1"),
                        sum_w2=res_b.metadata.get("sum_w2"),
                    )
            result = _apply_bootstrap_result(
                result,
                boot,
                cumulative_realizations=boot_cumulative,
                backend="mc_pdf_rerun",
            )
            result.metadata["mc_resampling_backend"] = "rerun"
            result.metadata["mc_resampling_random_policy"] = "reinherit" if str(spec.random_mode).strip().lower() == "inherit_realization" else str(spec.random_mode)

    if bool(config.jackknife.enabled):
        nregions = int(meta.get("jk_nregions") or 0)
        nres_mc = _resampling_nreal(spec)
        if _mc_fast_jackknife_enabled(config):
            _stage(config.progress.enabled, "[pccf:mc_pdf] assembling fast jackknife covariance")
            jk_counts = _cross_mc_fast_jackknife_counts(
                t1=t1, t2=t2, tr1=tr1, tr2=tr2,
                cdf1=cdf1, cdf2=cdf2, pbar1=pbar1, pbar2=pbar2, cdf_r1=cdf_r1, cdf_r2=cdf_r2,
                data1=data1, data2=data2, random1=random1, random2=random2,
                chi_grid=chi_grid, sample_edges_chi=sample_edges_chi, config=config, meta=meta, sbound=sbound, grid_meta=grid_meta,
                g1=g1, g2=g2, gr1=gr1, gr2=gr2, nreal=nres_mc, base_seed=int(spec.seed),
            )
            jk = _jackknife_wp_from_cross_touch(jk_counts, t1, t2, tr1, tr2, config, meta)
            cov = jackknife_cov(jk)
            result.wp_err = np.sqrt(np.diag(cov))
            result.cov = cov if config.jackknife.return_cov else None
            result.realizations = jk if config.jackknife.return_realizations else None
            result.metadata.update({
                "jackknife": True,
                "jk_nregions": int(jk.shape[0]),
                "jk_region_source": meta.get("jk_region_source"),
                "jk_touch_fast": True,
                "mc_resampling_backend": "fast",
                "mc_resampling_random_policy": "fixed" if str(spec.random_mode).strip().lower() == "inherit_realization" else str(spec.random_mode),
                "mc_fast_jackknife_count_average": True,
            })
        else:
            _stage(config.progress.enabled, "[pccf:mc_pdf] assembling jackknife covariance")
            jk = np.zeros((nregions, len(meta["rp_centers"])), dtype=np.float64)
            d1reg = np.asarray(t1.get("region_id"), dtype=np.int32)
            d2reg = np.asarray(t2.get("region_id"), dtype=np.int32)
            r1reg = None if tr1 is None else np.asarray(tr1.get("region_id"), dtype=np.int32)
            r2reg = None if tr2 is None else np.asarray(tr2.get("region_id"), dtype=np.int32)
            for k in range(nregions):
                idx1 = np.flatnonzero(d1reg != k).astype(np.int64)
                idx2 = np.flatnonzero(d2reg != k).astype(np.int64)
                ridx1 = None if r1reg is None else np.flatnonzero(r1reg != k).astype(np.int64)
                ridx2 = None if r2reg is None else np.flatnonzero(r2reg != k).astype(np.int64)
                res_k = _cross_mc_result_for_selection(
                    t1=t1, t2=t2, tr1=tr1, tr2=tr2,
                    cdf1=cdf1, cdf2=cdf2, pbar1=pbar1, pbar2=pbar2, cdf_r1=cdf_r1, cdf_r2=cdf_r2,
                    data1=data1, data2=data2, random1=random1, random2=random2,
                    chi_grid=chi_grid, sample_edges_chi=sample_edges_chi, config=config, meta=meta, sbound=sbound, grid_meta=grid_meta,
                    g1=g1, g2=g2, gr1=gr1, gr2=gr2, nreal=nres_mc, base_seed=int(spec.seed),
                    idx1=idx1, idx2=idx2, ridx1=ridx1, ridx2=ridx2, fixed_rdist1=fixed_rdist1_resampling, fixed_rdist2=fixed_rdist2_resampling,
                    status_label=f"jackknife-cross-{k}",
                )
                jk[k] = np.asarray(res_k.wp, dtype=np.float64)
            cov = jackknife_cov(jk)
            result.wp_err = np.sqrt(np.diag(cov))
            result.cov = cov if config.jackknife.return_cov else None
            result.realizations = jk if config.jackknife.return_realizations else None
            result.metadata.update({
                "jackknife": True,
                "jk_nregions": int(jk.shape[0]),
                "jk_region_source": meta.get("jk_region_source"),
                "jk_touch_fast": False,
                "mc_resampling_backend": "rerun",
                "mc_resampling_random_policy": "reinherit" if str(spec.random_mode).strip().lower() == "inherit_realization" else str(spec.random_mode),
            })
    result = apply_bootstrap_storage_policy(result, config.bootstrap)
    _stage(config.progress.enabled, "[pccf:mc_pdf] done")
    return attach_roundtrip_context(result, config=config, provenance=provenance_dict("pccf"), extra_metadata=meta)


__all__ = ["_mc_pdf_enabled", "run_auto_mc_pdf", "run_cross_mc_pdf"]

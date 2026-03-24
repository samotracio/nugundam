"""Marked-correlation wrappers built on top of the existing nuGUNDAM APIs.

This module provides thin, high-level helpers that evaluate a standard
correlation function twice on matched catalogs and then combine the results into
an object-marked statistic. The first branch is always an ordinary unweighted
measurement. The second branch reuses the same estimator, binning, and
resampling options, but replaces the data weights with user-supplied mark
values.

The marked statistics currently implemented here are restricted to *object*
marks, meaning that a pair receives a weight built from per-object scalars such
as luminosity, color, stellar mass, or any other positive quantity attached to
each object. More general pair-dependent kernels are intentionally left for a
future extension.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal

import numpy as np

from .angular.api import acf, accf
from .angular.estimators import _normalize_auto_unweighted as _ang_norm_auto_unweighted
from .angular.estimators import _normalize_auto_weighted as _ang_norm_auto_weighted
from .angular.models import (
    AngularAutoConfig,
    AngularCorrelationResult,
    AngularCrossConfig,
    CatalogColumns,
    ConfigDocMixin,
    ResultIOMixin,
    WeightSpec,
)
from .core.catalogs import catalog_column_names, catalog_get_column
from .projected.api import pcf, pccf
from .projected.estimators import _integrate_bootstrap as _proj_integrate_bootstrap
from .projected.estimators import _normalize_auto_unweighted as _proj_norm_auto_unweighted
from .projected.estimators import _normalize_auto_weighted as _proj_norm_auto_weighted
from .projected.models import (
    ProjectedAutoConfig,
    ProjectedCatalogColumns,
    ProjectedCorrelationResult,
    ProjectedCrossConfig,
)
from .result_meta import attach_roundtrip_context, provenance_dict

MarkNormalize = Literal["mean", "median", "none"]
MarkTransform = Literal["identity", "rank"]
MarkTarget = Literal["data1", "data2", "both"]

_MARK_COL = "__nugundam_mark_weight__"
_UNIT_COL = "__nugundam_mark_unit__"


@dataclass(slots=True)
class AutoMarkSpec(ConfigDocMixin):
    """Specification of the object mark used by marked auto-correlations.

    Parameters
    ----------
    column : str
        Name of the catalog column containing the per-object mark.
    normalize : {"mean", "median", "none"}, default="mean"
        Normalization applied after any transform and clipping. ``"mean"``
        divides the marks by their sample mean, ``"median"`` divides by their
        sample median, and ``"none"`` leaves the transformed values unchanged.
        Using ``"mean"`` is the most common choice because it makes the mark
        average close to unity.
    transform : {"identity", "rank"}, default="identity"
        Optional transform applied before normalization. ``"rank"`` replaces
        the marks with ordinal ranks from 1 to ``N`` (ties keep the ordering
        implied by ``numpy.argsort``), which is useful when the original mark
        distribution is very skewed.
    clip : tuple[float, float] or None, default=None
        Optional lower and upper clipping bounds applied before the transform
        and normalization. When provided, values are clipped to the inclusive
        interval ``[clip[0], clip[1]]``.
    missing : {"raise", "drop"}, default="raise"
        Policy used when the mark column contains NaN or infinite values.
        ``"raise"`` aborts the run with an informative error. ``"drop"``
        removes those rows from both the plain and marked branches so that the
        statistic is built from matched object sets.

    Notes
    -----
    The marked wrappers interpret the processed mark as a positive object
    weight. If the transformed and normalized marks contain non-positive
    entries, a :class:`ValueError` is raised and the user should first remap the
    science quantity to a strictly positive mark.
    """

    column: str = field(metadata={"doc": "Catalog column containing the per-object mark."})
    normalize: MarkNormalize = field(default="mean", metadata={"doc": "Normalization applied after any transform and clipping."})
    transform: MarkTransform = field(default="identity", metadata={"doc": "Optional transform applied before normalization."})
    clip: tuple[float, float] | None = field(default=None, metadata={"doc": "Optional lower and upper clipping bounds applied to the raw marks."})
    missing: Literal["raise", "drop"] = field(default="raise", metadata={"doc": "Policy used when the mark column contains NaN or infinite values."})


@dataclass(slots=True)
class CrossMarkSpec(ConfigDocMixin):
    """Specification of the object mark used by marked cross-correlations.

    Parameters
    ----------
    column1 : str or None, default=None
        Mark column for the first data catalog. This field is required when
        ``mark_on`` is ``"data1"`` or ``"both"``.
    column2 : str or None, default=None
        Mark column for the second data catalog. This field is required when
        ``mark_on`` is ``"data2"`` or ``"both"``.
    mark_on : {"data1", "data2", "both"}, default="both"
        Which cross-correlation sample(s) receive mark weights in the weighted
        branch.
    normalize : {"mean", "median", "none"}, default="mean"
        Normalization applied independently to each marked sample after any
        transform and clipping.
    transform : {"identity", "rank"}, default="identity"
        Optional transform applied independently to each marked sample before
        normalization.
    clip : tuple[float, float] or None, default=None
        Optional lower and upper clipping bounds applied independently to the
        raw marks before the transform and normalization.
    missing : {"raise", "drop"}, default="raise"
        Policy used when one of the mark columns contains NaN or infinite
        values. ``"drop"`` removes the affected rows from the corresponding
        catalog in both the plain and weighted branches.

    Notes
    -----
    In the weighted branch, the marked sample uses its processed mark as the
    object weight, while any unmarked sample receives unit weights. Random
    catalogs always remain unweighted.
    """

    column1: str | None = field(default=None, metadata={"doc": "Mark column for the first data catalog."})
    column2: str | None = field(default=None, metadata={"doc": "Mark column for the second data catalog."})
    mark_on: MarkTarget = field(default="both", metadata={"doc": "Which cross-correlation sample(s) receive mark weights in the weighted branch."})
    normalize: MarkNormalize = field(default="mean", metadata={"doc": "Normalization applied independently to each marked sample."})
    transform: MarkTransform = field(default="identity", metadata={"doc": "Optional transform applied independently to each marked sample."})
    clip: tuple[float, float] | None = field(default=None, metadata={"doc": "Optional clipping bounds applied to each marked sample before normalization."})
    missing: Literal["raise", "drop"] = field(default="raise", metadata={"doc": "Policy used when a mark column contains NaN or infinite values."})


@dataclass(slots=True)
class MarkedAngularCorrelationResult(ResultIOMixin):
    """Marked angular correlation result returned by :func:`macf` or :func:`maccf`.

    Parameters
    ----------
    theta_edges, theta_centers : ndarray
        Angular bin edges and centers copied from the underlying nuGUNDAM runs.
    mtheta : ndarray
        Final marked statistic. By default this is
        ``(1 + w_marked) / (1 + w_plain)`` evaluated bin-by-bin.
    mtheta_err : ndarray
        Diagonal uncertainty estimate for ``mtheta``. When bootstrap or
        jackknife resampling is enabled, these uncertainties are derived from
        matched marked realizations rather than by combining the plain and
        weighted diagonal errors independently.
    estimator : {"NAT", "DP", "LS"}
        Estimator used in both underlying branches.
    plain : AngularCorrelationResult
        Ordinary unweighted angular correlation result.
    weighted : AngularCorrelationResult
        Angular correlation result obtained after replacing the data weights by
        the processed mark values.
    cov : ndarray or None, default=None
        Marked covariance matrix, currently populated only for jackknife runs.
    realizations : ndarray or None, default=None
        Optional matrix of matched marked realizations with shape
        ``(nrealizations, nbins)``.
    metadata : dict, default_factory=dict
        Provenance and configuration metadata attached by nuGUNDAM.
    """

    theta_edges: np.ndarray
    theta_centers: np.ndarray
    mtheta: np.ndarray
    mtheta_err: np.ndarray
    estimator: str
    plain: AngularCorrelationResult
    weighted: AngularCorrelationResult
    cov: np.ndarray | None = None
    realizations: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def plain_wtheta(self) -> np.ndarray:
        """Ordinary unweighted angular correlation used in the denominator."""
        return self.plain.wtheta

    @property
    def weighted_wtheta(self) -> np.ndarray:
        """Mark-weighted angular correlation used in the numerator."""
        return self.weighted.wtheta


@dataclass(slots=True)
class MarkedProjectedCorrelationResult(ResultIOMixin):
    """Marked projected correlation result returned by :func:`mpcf` or :func:`mpccf`.

    Parameters
    ----------
    rp_edges, rp_centers : ndarray
        Projected-separation bin edges and centers copied from the underlying
        nuGUNDAM runs.
    mrp : ndarray
        Final marked projected statistic. The current default is
        ``(1 + wp_marked / rp) / (1 + wp_plain / rp)`` evaluated bin-by-bin.
    mrp_err : ndarray
        Diagonal uncertainty estimate for ``mrp`` derived from matched marked
        realizations when resampling is enabled.
    estimator : {"NAT", "DP", "LS"}
        Estimator used in both underlying branches.
    plain : ProjectedCorrelationResult
        Ordinary unweighted projected correlation result.
    weighted : ProjectedCorrelationResult
        Projected correlation result obtained after replacing the data weights
        by the processed mark values.
    cov : ndarray or None, default=None
        Marked covariance matrix, currently populated only for jackknife runs.
    realizations : ndarray or None, default=None
        Optional matrix of matched marked realizations with shape
        ``(nrealizations, nbins)``.
    metadata : dict, default_factory=dict
        Provenance and configuration metadata attached by nuGUNDAM.
    """

    rp_edges: np.ndarray
    rp_centers: np.ndarray
    mrp: np.ndarray
    mrp_err: np.ndarray
    estimator: str
    plain: ProjectedCorrelationResult
    weighted: ProjectedCorrelationResult
    cov: np.ndarray | None = None
    realizations: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def plain_wp(self) -> np.ndarray:
        """Ordinary unweighted projected correlation used in the denominator."""
        return self.plain.wp

    @property
    def weighted_wp(self) -> np.ndarray:
        """Mark-weighted projected correlation used in the numerator."""
        return self.weighted.wp


def _table_to_mapping(table, *, row_mask: np.ndarray | None = None, extra: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Materialize a supported catalog input as a plain mapping of NumPy arrays."""
    names = catalog_column_names(table)
    out: dict[str, np.ndarray] = {}
    for name in names:
        arr = np.asarray(catalog_get_column(table, name))
        if row_mask is not None:
            arr = np.asarray(arr[row_mask])
        out[str(name)] = arr
    if extra:
        for key, value in extra.items():
            arr = np.asarray(value)
            if row_mask is not None and arr.shape[0] != int(np.sum(row_mask)):
                # ``extra`` arrays are expected to already be aligned to the
                # filtered sample. Leave them untouched in that case.
                out[key] = arr
            else:
                out[key] = arr
    return out


def _rank_transform(values: np.ndarray) -> np.ndarray:
    """Return 1-based ordinal ranks for a 1-D array."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def _process_marks(table, spec: AutoMarkSpec, *, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Load, validate, transform, and normalize a mark column.

    Returns
    -------
    keep_mask : ndarray of bool
        Row-selection mask to apply to the catalog before entering the plain and
        weighted branches. The mask is all True unless ``spec.missing`` is
        ``"drop"`` and the mark column contains invalid values.
    marks : ndarray
        Processed strictly positive mark values aligned with the filtered rows.
    """
    raw = np.asarray(catalog_get_column(table, column, dtype=np.float64), dtype=np.float64)
    if raw.ndim != 1:
        raw = np.ravel(raw)
    finite = np.isfinite(raw)
    if not np.all(finite):
        if spec.missing == "raise":
            bad = int(np.count_nonzero(~finite))
            raise ValueError(f"Mark column {column!r} contains {bad} NaN or infinite values.")
        keep = finite
        raw = raw[keep]
    else:
        keep = np.ones(raw.shape[0], dtype=bool)

    if raw.size == 0:
        raise ValueError(f"Mark column {column!r} does not contain any valid rows after filtering.")

    if spec.clip is not None:
        lo, hi = float(spec.clip[0]), float(spec.clip[1])
        if not lo <= hi:
            raise ValueError(f"Invalid clip interval {spec.clip!r}; expected clip[0] <= clip[1].")
        raw = np.clip(raw, lo, hi)

    if spec.transform == "identity":
        marks = raw.astype(np.float64, copy=False)
    elif spec.transform == "rank":
        marks = _rank_transform(raw)
    else:
        raise ValueError(f"Unsupported mark transform {spec.transform!r}.")

    if spec.normalize == "mean":
        denom = float(np.mean(marks))
    elif spec.normalize == "median":
        denom = float(np.median(marks))
    elif spec.normalize == "none":
        denom = 1.0
    else:
        raise ValueError(f"Unsupported mark normalization {spec.normalize!r}.")

    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(
            f"Cannot normalize mark column {column!r}: the chosen normalization produced a non-positive denominator ({denom})."
        )

    marks = marks / denom
    if not np.all(np.isfinite(marks)):
        raise ValueError(f"Processed marks for column {column!r} contain non-finite values.")
    if np.any(marks <= 0.0):
        raise ValueError(
            f"Processed marks for column {column!r} are not strictly positive. "
            "Please shift or transform the science quantity before using it as a mark."
        )
    return keep, marks


def _validate_cross_mark_spec(spec: CrossMarkSpec) -> None:
    """Validate that the requested marked cross-correlation is well defined."""
    if spec.mark_on in {"data1", "both"} and not spec.column1:
        raise ValueError("CrossMarkSpec.column1 is required when mark_on='data1' or mark_on='both'.")
    if spec.mark_on in {"data2", "both"} and not spec.column2:
        raise ValueError("CrossMarkSpec.column2 is required when mark_on='data2' or mark_on='both'.")


def _mark_to_auto_spec(spec: CrossMarkSpec, column: str) -> AutoMarkSpec:
    """Project a cross mark specification onto a single-catalog auto spec."""
    return AutoMarkSpec(
        column=column,
        normalize=spec.normalize,
        transform=spec.transform,
        clip=spec.clip,
        missing=spec.missing,
    )


def _bootstrap_realizations_angular_auto(counts, *, estimator: str, data_weights: np.ndarray | None = None) -> np.ndarray:
    """Return bootstrap angular auto realizations with shape ``(nboot, nbins)``."""
    if counts.dd_boot is None or counts.dd_boot.size == 0:
        return np.zeros((0, len(counts.theta_centers)), dtype=np.float64)
    if data_weights is None:
        dd_n, rr_n, dr_n, bdd_n = _ang_norm_auto_unweighted(counts)
    else:
        dd_n, rr_n, dr_n, bdd_n = _ang_norm_auto_weighted(counts, data_weights)
    assert bdd_n is not None
    nb = bdd_n.shape[1]
    out = np.zeros((nb, len(counts.theta_centers)), dtype=np.float64)
    for i in range(len(counts.theta_centers)):
        if estimator == "NAT":
            if rr_n is not None and rr_n[i] > 0:
                out[:, i] = bdd_n[i] / rr_n[i] - 1.0
        elif estimator == "DP":
            if dr_n is not None and dr_n[i] > 0:
                out[:, i] = bdd_n[i] / dr_n[i] - 1.0
        elif estimator == "LS":
            if rr_n is not None and rr_n[i] > 0 and dr_n is not None:
                out[:, i] = (bdd_n[i] - 2.0 * dr_n[i] + rr_n[i]) / rr_n[i]
        else:
            raise ValueError(f"Unsupported angular auto estimator {estimator!r}.")
    return out


def _bootstrap_realizations_angular_cross(counts, *, estimator: str, sum_w1: float | None = None, sum_w2: float | None = None) -> np.ndarray:
    """Return bootstrap angular cross realizations with shape ``(nboot, nbins)``."""
    if counts.d1d2_boot is None or counts.d1d2_boot.size == 0:
        return np.zeros((0, len(counts.theta_centers)), dtype=np.float64)
    n1 = float(counts.metadata["n_data1"])
    nr1 = float(counts.metadata["n_random1"])
    n2 = float(counts.metadata["n_data2"])
    nr2 = float(counts.metadata["n_random2"])
    sw1 = n1 if sum_w1 is None else float(sum_w1)
    sw2 = n2 if sum_w2 is None else float(sum_w2)
    d1r2_n = counts.d1r2 / (sw1 * nr2) if counts.d1r2 is not None else None
    r1d2_n = counts.r1d2 / (nr1 * sw2) if counts.r1d2 is not None else None
    r1r2_n = counts.r1r2 / (nr1 * nr2) if counts.r1r2 is not None else None
    b_d1d2_n = counts.d1d2_boot / (sw1 * sw2)
    b_d1r2_n = counts.d1r2_boot / (sw1 * nr2) if counts.d1r2_boot is not None and counts.d1r2_boot.size > 0 and d1r2_n is not None else None
    nb = b_d1d2_n.shape[1]
    out = np.zeros((nb, len(counts.theta_centers)), dtype=np.float64)
    for i in range(len(counts.theta_centers)):
        if estimator == "NAT":
            if r1r2_n is not None and r1r2_n[i] > 0:
                out[:, i] = b_d1d2_n[i] / r1r2_n[i] - 1.0
        elif estimator == "DP":
            if d1r2_n is not None and d1r2_n[i] > 0 and b_d1r2_n is not None:
                out[:, i] = b_d1d2_n[i] / b_d1r2_n[i] - 1.0
        elif estimator == "LS":
            if r1r2_n is not None and r1r2_n[i] > 0 and d1r2_n is not None and r1d2_n is not None and b_d1r2_n is not None:
                out[:, i] = (b_d1d2_n[i] - b_d1r2_n[i] - r1d2_n[i] + r1r2_n[i]) / r1r2_n[i]
        else:
            raise ValueError(f"Unsupported angular cross estimator {estimator!r}.")
    return out


def _bootstrap_realizations_projected_auto(counts, *, estimator: str, data_weights: np.ndarray | None = None) -> np.ndarray:
    """Return bootstrap projected auto realizations with shape ``(nboot, nbins)``."""
    if counts.dd_boot is None or counts.dd_boot.size == 0:
        return np.zeros((0, len(counts.rp_centers)), dtype=np.float64)
    if data_weights is None:
        dd_n, rr_n, dr_n, bdd_n = _proj_norm_auto_unweighted(counts)
    else:
        dd_n, rr_n, dr_n, bdd_n = _proj_norm_auto_weighted(counts, data_weights)
    assert bdd_n is not None
    bxi = np.zeros_like(bdd_n, dtype=np.float64)
    for i in range(counts.dd.shape[0]):
        if estimator == "NAT":
            if rr_n is not None:
                mask = rr_n[i] > 0
                if np.any(mask):
                    bxi[i, mask, :] = bdd_n[i, mask, :] / rr_n[i, mask][:, None] - 1.0
        elif estimator == "DP":
            if dr_n is not None:
                mask = dr_n[i] > 0
                if np.any(mask):
                    bxi[i, mask, :] = bdd_n[i, mask, :] / dr_n[i, mask][:, None] - 1.0
        elif estimator == "LS":
            if rr_n is not None and dr_n is not None:
                mask = rr_n[i] > 0
                if np.any(mask):
                    bxi[i, mask, :] = (bdd_n[i, mask, :] - 2.0 * dr_n[i, mask][:, None] + rr_n[i, mask][:, None]) / rr_n[i, mask][:, None]
        else:
            raise ValueError(f"Unsupported projected auto estimator {estimator!r}.")
    return _proj_integrate_bootstrap(bxi, counts.pi_edges[1:] - counts.pi_edges[:-1]).T


def _bootstrap_realizations_projected_cross(counts, *, estimator: str, sum_w1: float | None = None, sum_w2: float | None = None) -> np.ndarray:
    """Return bootstrap projected cross realizations with shape ``(nboot, nbins)``."""
    if counts.d1d2_boot is None or counts.d1d2_boot.size == 0:
        return np.zeros((0, len(counts.rp_centers)), dtype=np.float64)
    n1 = float(counts.metadata["n_data1"])
    nr1 = float(counts.metadata["n_random1"])
    n2 = float(counts.metadata["n_data2"])
    nr2 = float(counts.metadata["n_random2"])
    sw1 = n1 if sum_w1 is None else float(sum_w1)
    sw2 = n2 if sum_w2 is None else float(sum_w2)
    d1r2_n = counts.d1r2 / (sw1 * nr2) if counts.d1r2 is not None else None
    r1d2_n = counts.r1d2 / (nr1 * sw2) if counts.r1d2 is not None else None
    r1r2_n = counts.r1r2 / (nr1 * nr2) if counts.r1r2 is not None else None
    b_d1d2_n = counts.d1d2_boot / (sw1 * sw2)
    b_d1r2_n = counts.d1r2_boot / (sw1 * nr2) if counts.d1r2_boot is not None and counts.d1r2_boot.size > 0 and d1r2_n is not None else None
    bxi = np.zeros_like(b_d1d2_n, dtype=np.float64)
    for i in range(counts.d1d2.shape[0]):
        if estimator == "NAT":
            if r1r2_n is not None:
                mask = r1r2_n[i] > 0
                if np.any(mask):
                    bxi[i, mask, :] = b_d1d2_n[i, mask, :] / r1r2_n[i, mask][:, None] - 1.0
        elif estimator == "DP":
            if d1r2_n is not None and b_d1r2_n is not None:
                mask = d1r2_n[i] > 0
                if np.any(mask):
                    bxi[i, mask, :] = b_d1d2_n[i, mask, :] / b_d1r2_n[i, mask, :] - 1.0
        elif estimator == "LS":
            if r1r2_n is not None and d1r2_n is not None and r1d2_n is not None and b_d1r2_n is not None:
                mask = r1r2_n[i] > 0
                if np.any(mask):
                    bxi[i, mask, :] = (b_d1d2_n[i, mask, :] - b_d1r2_n[i, mask, :] - r1d2_n[i, mask][:, None] + r1r2_n[i, mask][:, None]) / r1r2_n[i, mask][:, None]
        else:
            raise ValueError(f"Unsupported projected cross estimator {estimator!r}.")
    return _proj_integrate_bootstrap(bxi, counts.pi_edges[1:] - counts.pi_edges[:-1]).T


def _combine_marked_angular(plain: np.ndarray, weighted: np.ndarray) -> np.ndarray:
    """Combine plain and weighted angular branches into ``M(theta)``."""
    denom = 1.0 + np.asarray(plain, dtype=np.float64)
    num = 1.0 + np.asarray(weighted, dtype=np.float64)
    return num / denom


def _combine_marked_projected(rp_centers: np.ndarray, plain: np.ndarray, weighted: np.ndarray) -> np.ndarray:
    """Combine plain and weighted projected branches into ``M(r_p)``."""
    rp = np.asarray(rp_centers, dtype=np.float64)
    if np.any(rp <= 0.0):
        raise ValueError("Projected marked correlations require strictly positive rp centers.")
    denom = 1.0 + np.asarray(plain, dtype=np.float64) / rp
    num = 1.0 + np.asarray(weighted, dtype=np.float64) / rp
    return num / denom


def _marked_metadata(base_config, mark_spec, *, definition: str, plain_metadata: dict, weighted_metadata: dict) -> dict:
    """Construct shared metadata for marked results."""
    return {
        "definition": definition,
        "mark": asdict(mark_spec),
        "plain_metadata": plain_metadata,
        "weighted_metadata": weighted_metadata,
    }


def macf(data, random, config: AngularAutoConfig, *, mark: AutoMarkSpec) -> MarkedAngularCorrelationResult:
    r"""Measure the marked angular auto-correlation function.

    The helper evaluates two matched nuGUNDAM angular auto-correlation runs on
    the same binning and estimator: an ordinary unweighted branch and a
    mark-weighted branch where the data weights are replaced by the processed
    mark values. The final statistic is then computed as

    .. math::

       M(\theta) = \frac{1 + w_{\mathrm{marked}}(\theta)}{1 + w(\theta)}.

    Parameters
    ----------
    data : table-like
        Data catalog understood by :func:`nugundam.acf`. The mark column is
        read from this catalog.
    random : table-like
        Random catalog sampling the same angular footprint.
    config : AngularAutoConfig
        Standard angular auto-correlation configuration. The marked wrapper
        reuses the same estimator, binning, resampling, and split-random
        options, but it always forces the plain branch to be unweighted and the
        weighted branch to use the processed mark values as object weights.
    mark : AutoMarkSpec
        Specification of the mark column and the preprocessing applied to it.

    Returns
    -------
    MarkedAngularCorrelationResult
        Result object containing the final marked statistic, its uncertainties,
        and the nested plain and weighted nuGUNDAM results.

    Notes
    -----
    When bootstrap or jackknife resampling is enabled, the marked error bars
    are derived from matched marked realizations built from the two branches,
    rather than by combining the already-compressed diagonal uncertainties from
    the plain and weighted runs.
    """
    keep, marks = _process_marks(data, mark, column=mark.column)
    data_base = _table_to_mapping(data, row_mask=keep)
    data_weighted = dict(data_base)
    data_weighted[_MARK_COL] = marks

    plain_cfg = replace(config, weights=WeightSpec(weight_mode="unweighted"))
    weighted_cfg = replace(
        config,
        columns_data=replace(config.columns_data, weight=_MARK_COL),
        weights=WeightSpec(weight_mode="weighted", data_col=_MARK_COL),
    )
    if config.jackknife.enabled and not config.jackknife.return_realizations:
        jk = replace(config.jackknife, return_realizations=True)
        plain_cfg = replace(plain_cfg, jackknife=jk)
        weighted_cfg = replace(weighted_cfg, jackknife=jk)

    plain = acf(data_base, random, plain_cfg)
    weighted = acf(data_weighted, random, weighted_cfg)

    mtheta = _combine_marked_angular(plain.wtheta, weighted.wtheta)
    cov = None
    realizations = None
    if config.jackknife.enabled:
        plain_real = np.asarray(plain.realizations, dtype=np.float64)
        weighted_real = np.asarray(weighted.realizations, dtype=np.float64)
        realizations_full = _combine_marked_angular(plain_real, weighted_real)
        cov = np.cov(realizations_full, rowvar=False, ddof=1) * (realizations_full.shape[0] - 1)
        err = np.sqrt(np.diag(cov))
        realizations = realizations_full if config.jackknife.return_realizations else None
    elif config.bootstrap.enabled:
        plain_boot = _bootstrap_realizations_angular_auto(plain.counts, estimator=config.estimator, data_weights=None)
        weighted_boot = _bootstrap_realizations_angular_auto(weighted.counts, estimator=config.estimator, data_weights=marks)
        boot = _combine_marked_angular(plain_boot, weighted_boot)
        err = np.std(boot, axis=0) if boot.size else np.zeros_like(mtheta)
    else:
        err = np.zeros_like(mtheta)

    result = MarkedAngularCorrelationResult(
        theta_edges=np.asarray(plain.theta_edges, dtype=np.float64),
        theta_centers=np.asarray(plain.theta_centers, dtype=np.float64),
        mtheta=np.asarray(mtheta, dtype=np.float64),
        mtheta_err=np.asarray(err, dtype=np.float64),
        estimator=str(config.estimator),
        plain=plain,
        weighted=weighted,
        cov=None if cov is None or not config.jackknife.return_cov else cov,
        realizations=realizations,
        metadata=_marked_metadata(
            config,
            mark,
            definition="(1 + w_marked(theta)) / (1 + w_plain(theta))",
            plain_metadata=plain.metadata,
            weighted_metadata=weighted.metadata,
        ),
    )
    return attach_roundtrip_context(
        result,
        config={"base_config": asdict(config), "mark": asdict(mark)},
        provenance=provenance_dict("macf"),
    )


def maccf(data1, data2, config: AngularCrossConfig, *, mark: CrossMarkSpec, random1=None, random2=None) -> MarkedAngularCorrelationResult:
    r"""Measure the marked angular cross-correlation function.

    The helper evaluates two matched angular cross-correlation runs and combines
    them as

    .. math::

       M_{12}(\theta) = \frac{1 + w^{\mathrm{marked}}_{12}(\theta)}{1 + w_{12}(\theta)}.

    Depending on ``mark.mark_on``, the weighted branch applies the processed
    mark to the first data catalog, the second data catalog, or both.

    Parameters
    ----------
    data1, data2 : table-like
        Data catalogs understood by :func:`nugundam.accf`.
    config : AngularCrossConfig
        Standard angular cross-correlation configuration reused by both
        branches. The wrapper forces the plain branch to be unweighted and uses
        unit weights for any unmarked data sample in the weighted branch.
    mark : CrossMarkSpec
        Specification of which cross-correlation sample(s) carry marks and how
        those marks are preprocessed.
    random1, random2 : table-like, optional
        Random catalogs required by the chosen estimator and bootstrap primary.

    Returns
    -------
    MarkedAngularCorrelationResult
        Result object containing the final marked statistic, its uncertainties,
        and the nested plain and weighted nuGUNDAM cross-correlation results.
    """
    _validate_cross_mark_spec(mark)

    keep1, marks1 = (np.ones(len(catalog_get_column(data1, config.columns_data1.ra)), dtype=bool), None)
    keep2, marks2 = (np.ones(len(catalog_get_column(data2, config.columns_data2.ra)), dtype=bool), None)
    if mark.mark_on in {"data1", "both"}:
        keep1, marks1 = _process_marks(data1, _mark_to_auto_spec(mark, mark.column1), column=str(mark.column1))
    if mark.mark_on in {"data2", "both"}:
        keep2, marks2 = _process_marks(data2, _mark_to_auto_spec(mark, mark.column2), column=str(mark.column2))

    data1_base = _table_to_mapping(data1, row_mask=keep1)
    data2_base = _table_to_mapping(data2, row_mask=keep2)
    data1_weighted = dict(data1_base)
    data2_weighted = dict(data2_base)
    ones1 = np.ones(len(next(iter(data1_base.values()))) if data1_base else 0, dtype=np.float64)
    ones2 = np.ones(len(next(iter(data2_base.values()))) if data2_base else 0, dtype=np.float64)
    data1_weighted[_UNIT_COL] = ones1
    data2_weighted[_UNIT_COL] = ones2
    if marks1 is not None:
        data1_weighted[_MARK_COL] = marks1
    if marks2 is not None:
        data2_weighted[_MARK_COL] = marks2

    plain_cfg = replace(config, weights=WeightSpec(weight_mode="unweighted"))
    weight_col1 = _MARK_COL if marks1 is not None else _UNIT_COL
    weight_col2 = _MARK_COL if marks2 is not None else _UNIT_COL
    weighted_cfg = replace(
        config,
        columns_data1=replace(config.columns_data1, weight=weight_col1),
        columns_data2=replace(config.columns_data2, weight=weight_col2),
        weights=WeightSpec(weight_mode="weighted", data1_col=weight_col1, data2_col=weight_col2),
    )
    if config.jackknife.enabled and not config.jackknife.return_realizations:
        jk = replace(config.jackknife, return_realizations=True)
        plain_cfg = replace(plain_cfg, jackknife=jk)
        weighted_cfg = replace(weighted_cfg, jackknife=jk)

    plain = accf(data1_base, data2_base, plain_cfg, random1=random1, random2=random2)
    weighted = accf(data1_weighted, data2_weighted, weighted_cfg, random1=random1, random2=random2)

    mtheta = _combine_marked_angular(plain.wtheta, weighted.wtheta)
    cov = None
    realizations = None
    sum_w1 = None if marks1 is None else float(np.sum(marks1))
    sum_w2 = None if marks2 is None else float(np.sum(marks2))
    if config.jackknife.enabled:
        plain_real = np.asarray(plain.realizations, dtype=np.float64)
        weighted_real = np.asarray(weighted.realizations, dtype=np.float64)
        realizations_full = _combine_marked_angular(plain_real, weighted_real)
        cov = np.cov(realizations_full, rowvar=False, ddof=1) * (realizations_full.shape[0] - 1)
        err = np.sqrt(np.diag(cov))
        realizations = realizations_full if config.jackknife.return_realizations else None
    elif config.bootstrap.enabled:
        plain_boot = _bootstrap_realizations_angular_cross(plain.counts, estimator=config.estimator)
        weighted_boot = _bootstrap_realizations_angular_cross(weighted.counts, estimator=config.estimator, sum_w1=sum_w1, sum_w2=sum_w2)
        boot = _combine_marked_angular(plain_boot, weighted_boot)
        err = np.std(boot, axis=0) if boot.size else np.zeros_like(mtheta)
    else:
        err = np.zeros_like(mtheta)

    result = MarkedAngularCorrelationResult(
        theta_edges=np.asarray(plain.theta_edges, dtype=np.float64),
        theta_centers=np.asarray(plain.theta_centers, dtype=np.float64),
        mtheta=np.asarray(mtheta, dtype=np.float64),
        mtheta_err=np.asarray(err, dtype=np.float64),
        estimator=str(config.estimator),
        plain=plain,
        weighted=weighted,
        cov=None if cov is None or not config.jackknife.return_cov else cov,
        realizations=realizations,
        metadata=_marked_metadata(
            config,
            mark,
            definition="(1 + w_marked_12(theta)) / (1 + w_plain_12(theta))",
            plain_metadata=plain.metadata,
            weighted_metadata=weighted.metadata,
        ),
    )
    return attach_roundtrip_context(
        result,
        config={"base_config": asdict(config), "mark": asdict(mark)},
        provenance=provenance_dict("maccf"),
    )


def mpcf(data, random, config: ProjectedAutoConfig, *, mark: AutoMarkSpec) -> MarkedProjectedCorrelationResult:
    r"""Measure the marked projected auto-correlation function.

    The helper evaluates two matched projected auto-correlation runs and then
    combines them as

    .. math::

       M(r_p) = \frac{1 + w_{p,\mathrm{marked}}(r_p) / r_p}{1 + w_p(r_p) / r_p}.

    Parameters
    ----------
    data : table-like
        Data catalog understood by :func:`nugundam.pcf`. The mark column is
        read from this catalog.
    random : table-like
        Random catalog sampling the same survey selection function.
    config : ProjectedAutoConfig
        Standard projected auto-correlation configuration reused by both
        branches.
    mark : AutoMarkSpec
        Specification of the mark column and the preprocessing applied to it.

    Returns
    -------
    MarkedProjectedCorrelationResult
        Result object containing the final marked statistic, its uncertainties,
        and the nested plain and weighted nuGUNDAM results.
    """
    keep, marks = _process_marks(data, mark, column=mark.column)
    data_base = _table_to_mapping(data, row_mask=keep)
    data_weighted = dict(data_base)
    data_weighted[_MARK_COL] = marks

    plain_cfg = replace(config, weights=WeightSpec(weight_mode="unweighted"))
    weighted_cfg = replace(
        config,
        columns_data=replace(config.columns_data, weight=_MARK_COL),
        weights=WeightSpec(weight_mode="weighted", data_col=_MARK_COL),
    )
    if config.jackknife.enabled and not config.jackknife.return_realizations:
        jk = replace(config.jackknife, return_realizations=True)
        plain_cfg = replace(plain_cfg, jackknife=jk)
        weighted_cfg = replace(weighted_cfg, jackknife=jk)

    plain = pcf(data_base, random, plain_cfg)
    weighted = pcf(data_weighted, random, weighted_cfg)

    mrp = _combine_marked_projected(plain.rp_centers, plain.wp, weighted.wp)
    cov = None
    realizations = None
    if config.jackknife.enabled:
        plain_real = np.asarray(plain.realizations, dtype=np.float64)
        weighted_real = np.asarray(weighted.realizations, dtype=np.float64)
        rp = np.asarray(plain.rp_centers, dtype=np.float64)[None, :]
        realizations_full = (1.0 + weighted_real / rp) / (1.0 + plain_real / rp)
        cov = np.cov(realizations_full, rowvar=False, ddof=1) * (realizations_full.shape[0] - 1)
        err = np.sqrt(np.diag(cov))
        realizations = realizations_full if config.jackknife.return_realizations else None
    elif config.bootstrap.enabled:
        plain_boot = _bootstrap_realizations_projected_auto(plain.counts, estimator=config.estimator, data_weights=None)
        weighted_boot = _bootstrap_realizations_projected_auto(weighted.counts, estimator=config.estimator, data_weights=marks)
        boot = _combine_marked_projected(plain.rp_centers, plain_boot, weighted_boot)
        err = np.std(boot, axis=0) if boot.size else np.zeros_like(mrp)
    else:
        err = np.zeros_like(mrp)

    result = MarkedProjectedCorrelationResult(
        rp_edges=np.asarray(plain.rp_edges, dtype=np.float64),
        rp_centers=np.asarray(plain.rp_centers, dtype=np.float64),
        mrp=np.asarray(mrp, dtype=np.float64),
        mrp_err=np.asarray(err, dtype=np.float64),
        estimator=str(config.estimator),
        plain=plain,
        weighted=weighted,
        cov=None if cov is None or not config.jackknife.return_cov else cov,
        realizations=realizations,
        metadata=_marked_metadata(
            config,
            mark,
            definition="(1 + wp_marked(rp) / rp) / (1 + wp_plain(rp) / rp)",
            plain_metadata=plain.metadata,
            weighted_metadata=weighted.metadata,
        ),
    )
    return attach_roundtrip_context(
        result,
        config={"base_config": asdict(config), "mark": asdict(mark)},
        provenance=provenance_dict("mpcf"),
    )


def mpccf(data1, data2, config: ProjectedCrossConfig, *, mark: CrossMarkSpec, random1=None, random2=None) -> MarkedProjectedCorrelationResult:
    r"""Measure the marked projected cross-correlation function.

    The weighted branch applies the processed mark to ``data1``, ``data2``, or
    both according to ``mark.mark_on`` and combines the branches as

    .. math::

       M_{12}(r_p) = \frac{1 + w^{\mathrm{marked}}_{p,12}(r_p) / r_p}{1 + w_{p,12}(r_p) / r_p}.

    Parameters
    ----------
    data1, data2 : table-like
        Data catalogs understood by :func:`nugundam.pccf`.
    config : ProjectedCrossConfig
        Standard projected cross-correlation configuration reused by both
        branches.
    mark : CrossMarkSpec
        Specification of which cross-correlation sample(s) carry marks and how
        those marks are preprocessed.
    random1, random2 : table-like, optional
        Random catalogs required by the chosen estimator and bootstrap primary.

    Returns
    -------
    MarkedProjectedCorrelationResult
        Result object containing the final marked statistic, its uncertainties,
        and the nested plain and weighted nuGUNDAM cross-correlation results.
    """
    _validate_cross_mark_spec(mark)

    keep1, marks1 = (np.ones(len(catalog_get_column(data1, config.columns_data1.ra)), dtype=bool), None)
    keep2, marks2 = (np.ones(len(catalog_get_column(data2, config.columns_data2.ra)), dtype=bool), None)
    if mark.mark_on in {"data1", "both"}:
        keep1, marks1 = _process_marks(data1, _mark_to_auto_spec(mark, mark.column1), column=str(mark.column1))
    if mark.mark_on in {"data2", "both"}:
        keep2, marks2 = _process_marks(data2, _mark_to_auto_spec(mark, mark.column2), column=str(mark.column2))

    data1_base = _table_to_mapping(data1, row_mask=keep1)
    data2_base = _table_to_mapping(data2, row_mask=keep2)
    data1_weighted = dict(data1_base)
    data2_weighted = dict(data2_base)
    ones1 = np.ones(len(next(iter(data1_base.values()))) if data1_base else 0, dtype=np.float64)
    ones2 = np.ones(len(next(iter(data2_base.values()))) if data2_base else 0, dtype=np.float64)
    data1_weighted[_UNIT_COL] = ones1
    data2_weighted[_UNIT_COL] = ones2
    if marks1 is not None:
        data1_weighted[_MARK_COL] = marks1
    if marks2 is not None:
        data2_weighted[_MARK_COL] = marks2

    plain_cfg = replace(config, weights=WeightSpec(weight_mode="unweighted"))
    weight_col1 = _MARK_COL if marks1 is not None else _UNIT_COL
    weight_col2 = _MARK_COL if marks2 is not None else _UNIT_COL
    weighted_cfg = replace(
        config,
        columns_data1=replace(config.columns_data1, weight=weight_col1),
        columns_data2=replace(config.columns_data2, weight=weight_col2),
        weights=WeightSpec(weight_mode="weighted", data1_col=weight_col1, data2_col=weight_col2),
    )
    if config.jackknife.enabled and not config.jackknife.return_realizations:
        jk = replace(config.jackknife, return_realizations=True)
        plain_cfg = replace(plain_cfg, jackknife=jk)
        weighted_cfg = replace(weighted_cfg, jackknife=jk)

    plain = pccf(data1_base, data2_base, plain_cfg, random1=random1, random2=random2)
    weighted = pccf(data1_weighted, data2_weighted, weighted_cfg, random1=random1, random2=random2)

    mrp = _combine_marked_projected(plain.rp_centers, plain.wp, weighted.wp)
    cov = None
    realizations = None
    sum_w1 = None if marks1 is None else float(np.sum(marks1))
    sum_w2 = None if marks2 is None else float(np.sum(marks2))
    if config.jackknife.enabled:
        plain_real = np.asarray(plain.realizations, dtype=np.float64)
        weighted_real = np.asarray(weighted.realizations, dtype=np.float64)
        rp = np.asarray(plain.rp_centers, dtype=np.float64)[None, :]
        realizations_full = (1.0 + weighted_real / rp) / (1.0 + plain_real / rp)
        cov = np.cov(realizations_full, rowvar=False, ddof=1) * (realizations_full.shape[0] - 1)
        err = np.sqrt(np.diag(cov))
        realizations = realizations_full if config.jackknife.return_realizations else None
    elif config.bootstrap.enabled:
        plain_boot = _bootstrap_realizations_projected_cross(plain.counts, estimator=config.estimator)
        weighted_boot = _bootstrap_realizations_projected_cross(weighted.counts, estimator=config.estimator, sum_w1=sum_w1, sum_w2=sum_w2)
        boot = _combine_marked_projected(plain.rp_centers, plain_boot, weighted_boot)
        err = np.std(boot, axis=0) if boot.size else np.zeros_like(mrp)
    else:
        err = np.zeros_like(mrp)

    result = MarkedProjectedCorrelationResult(
        rp_edges=np.asarray(plain.rp_edges, dtype=np.float64),
        rp_centers=np.asarray(plain.rp_centers, dtype=np.float64),
        mrp=np.asarray(mrp, dtype=np.float64),
        mrp_err=np.asarray(err, dtype=np.float64),
        estimator=str(config.estimator),
        plain=plain,
        weighted=weighted,
        cov=None if cov is None or not config.jackknife.return_cov else cov,
        realizations=realizations,
        metadata=_marked_metadata(
            config,
            mark,
            definition="(1 + wp_marked_12(rp) / rp) / (1 + wp_plain_12(rp) / rp)",
            plain_metadata=plain.metadata,
            weighted_metadata=weighted.metadata,
        ),
    )
    return attach_roundtrip_context(
        result,
        config={"base_config": asdict(config), "mark": asdict(mark)},
        provenance=provenance_dict("mpccf"),
    )


__all__ = [
    "AutoMarkSpec",
    "CrossMarkSpec",
    "MarkedAngularCorrelationResult",
    "MarkedProjectedCorrelationResult",
    "macf",
    "maccf",
    "mpcf",
    "mpccf",
]

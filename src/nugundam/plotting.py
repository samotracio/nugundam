"""Shared plotting helpers for 1D and 2D nuGUNDAM results."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

from .angular.models import (
    AngularAutoCounts,
    AngularAutoCountsResult,
    AngularCorrelationResult,
    AngularCrossCounts,
    AngularCrossCountsResult,
)
from .projected.estimators import compute_auto_xi2d, compute_cross_xi2d
from .projected.models import (
    ProjectedAutoCounts,
    ProjectedAutoCountsResult,
    ProjectedCorrelationResult,
    ProjectedCrossCounts,
    ProjectedCrossCountsResult,
)


_ANGULAR_UNIT_SCALE = {
    "deg": 1.0,
    "arcmin": 60.0,
    "arcsec": 3600.0,
}


def _merge_kwargs(base: dict[str, Any] | None, extra: dict[str, Any] | None) -> dict[str, Any]:
    """
    Merge kwargs.
    
    Parameters
    ----------
    base : object
        Value for ``base``.
    extra : object
        Value for ``extra``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    merged: dict[str, Any] = {}
    if base:
        merged.update(base)
    if extra:
        merged.update(extra)
    return merged


def _as_1d_array(values: Any, *, name: str) -> np.ndarray:
    """
    Convert the supplied value to 1d array.
    
    Parameters
    ----------
    values : object
        Value for ``values``.
    name : object
        Value for ``name``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = np.asarray([arr.item()])
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}")
    return arr.astype(float, copy=False)


def _broadcast_yerr(yerr: Any, n: int) -> np.ndarray:
    """
    Broadcast yerr.
    
    Parameters
    ----------
    yerr : object
        Value for ``yerr``.
    n : object
        Value for ``n``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    arr = np.asarray(yerr)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"yerr must be scalar or 1-D; got shape {arr.shape}")
    if len(arr) != n:
        raise ValueError(f"yerr has length {len(arr)} but expected {n}")
    return arr.astype(float, copy=False)


def _shift_log_x(x: np.ndarray, shift: float) -> np.ndarray:
    """
    Shift log x.
    
    Parameters
    ----------
    x : object
        Value for ``x``.
    shift : object
        Value for ``shift``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if shift == 0.0:
        return x
    if np.any(x <= 0.0):
        raise ValueError("shift requires strictly positive x values")
    if x.size < 2:
        return x
    lx = np.log10(x)
    dlx = np.median(np.diff(lx))
    return 10.0 ** (lx + shift * dlx)


def _mask_plot_values(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None,
    *,
    xscale: str,
    yscale: str,
    errors: str,
    mask_nonpositive: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Mask plot values.
    
    Parameters
    ----------
    x : object
        Value for ``x``.
    y : object
        Value for ``y``.
    yerr : object
        Value for ``yerr``.
    xscale : object
        Value for ``xscale``. This argument is keyword-only.
    yscale : object
        Value for ``yscale``. This argument is keyword-only.
    errors : object
        Value for ``errors``. This argument is keyword-only.
    mask_nonpositive : object
        Value for ``mask_nonpositive``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        mask &= np.isfinite(yerr)
    if mask_nonpositive:
        if xscale == "log":
            mask &= x > 0.0
        if yscale == "log":
            mask &= y > 0.0
            if yerr is not None and errors in {"bar", "band"}:
                mask &= (y - yerr) > 0.0
    return x[mask], y[mask], None if yerr is None else yerr[mask]


def _angular_xlabel(unit: str) -> str:
    """
    Angular xlabel.
    
    Parameters
    ----------
    unit : object
        Value for ``unit``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    mapping = {
        "deg": r"$\theta\,[^\circ]$",
        "arcmin": r"$\theta\,[\prime]$",
        "arcsec": r"$\theta\,[\prime\prime]$",
    }
    try:
        return mapping[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported angular unit: {unit!r}") from exc


def _count_ylabel(field: str) -> str:
    """
    Count ylabel.
    
    Parameters
    ----------
    field : object
        Value for ``field``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    return rf"$\mathrm{{{field.upper()}}}$"


def _projected_xlabel() -> str:
    """
    Projected xlabel.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    return r"$r_p\,[h^{-1}\,\mathrm{Mpc}]$"



def _projected_ylabel() -> str:
    """
    Projected ylabel.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    return r"$\pi\,[h^{-1}\,\mathrm{Mpc}]$"


def _sky_xlabel() -> str:
    r"""Return the default x-axis label for sky-region plots.

    Returns
    -------
    str
        Label for right ascension in degrees.
    """
    return r"$\alpha\,[^\circ]$"


def _sky_ylabel() -> str:
    r"""Return the default y-axis label for sky-region plots.

    Returns
    -------
    str
        Label for declination in degrees.
    """
    return r"$\delta\,[^\circ]$"


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix into a correlation matrix.

    Parameters
    ----------
    cov : ndarray
        Square covariance matrix.

    Returns
    -------
    ndarray
        Correlation matrix. Entries with non-finite normalization are set to
        zero.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be a square 2-D array; got shape {cov.shape}")
    diag = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(diag, diag)
    corr[~np.isfinite(corr)] = 0.0
    return corr


def _matrix_tick_spec(x: np.ndarray | None, n: int, *, max_ticks: int = 7) -> tuple[np.ndarray, list[str]]:
    """Choose tick positions and labels for matrix visualizations.

    Parameters
    ----------
    x : ndarray or None
        Optional physical coordinate associated with each matrix bin.
    n : int
        Number of matrix bins.
    max_ticks : int, default=7
        Maximum number of labeled ticks to display.

    Returns
    -------
    positions : ndarray
        Tick positions in image coordinates.
    labels : list of str
        Tick labels corresponding to the chosen positions.
    """
    if n <= 0:
        return np.empty(0, dtype=float), []
    step = max(1, int(np.ceil(n / max_ticks)))
    pos = np.arange(0, n, step, dtype=int)
    if pos[-1] != n - 1:
        pos = np.unique(np.append(pos, n - 1))
    if x is None:
        labels = [str(int(i)) for i in pos]
    else:
        labels = [f"{float(x[i]):.2g}" for i in pos]
    return pos, labels


def _extract_matrix_plot_spec(
    obj: Any,
    *,
    x: Any = None,
    angunit: str = "deg",
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Extract matrix data, coordinates, and labels for covariance plots.

    Parameters
    ----------
    obj : object
        nuGUNDAM result object exposing ``cov`` or a raw square covariance
        matrix.
    x : array-like, optional
        Bin coordinates associated with a raw covariance matrix.
    angunit : {"deg", "arcmin", "arcsec"}, default="deg"
        Angular unit used when ``obj`` is an angular result.

    Returns
    -------
    matrix : ndarray
        Square matrix to plot.
    xvals : ndarray or None
        Bin coordinates associated with the matrix axes.
    axis_label : str
        Default axis label for both matrix dimensions.
    """
    if isinstance(obj, AngularCorrelationResult):
        if obj.cov is None:
            raise ValueError("AngularCorrelationResult does not contain a covariance matrix.")
        matrix = np.asarray(obj.cov, dtype=float)
        xvals = _as_1d_array(obj.theta_centers, name="theta_centers") * _ANGULAR_UNIT_SCALE[angunit]
        return matrix, xvals, _angular_xlabel(angunit)
    if isinstance(obj, ProjectedCorrelationResult) or (hasattr(obj, "cov") and hasattr(obj, "rp_centers")):
        cov = getattr(obj, "cov", None)
        if cov is None:
            raise ValueError(f"{type(obj).__name__} does not contain a covariance matrix.")
        matrix = np.asarray(cov, dtype=float)
        xvals = _as_1d_array(getattr(obj, "rp_centers"), name="rp_centers")
        return matrix, xvals, _projected_xlabel()

    matrix = np.asarray(obj, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix input must be square; got shape {matrix.shape}")
    xvals = None if x is None else _as_1d_array(x, name="x")
    if xvals is not None and len(xvals) != matrix.shape[0]:
        raise ValueError(f"x has length {len(xvals)} but matrix has shape {matrix.shape}")
    return matrix, xvals, "Separation"


def _plot_matrix(
    matrix: np.ndarray,
    *,
    ax=None,
    x: np.ndarray | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    max_ticks: int = 7,
    imshow_kwargs: dict[str, Any] | None = None,
):
    """Render a square matrix with nuGUNDAM's default axis formatting.

    Parameters
    ----------
    matrix : ndarray
        Square matrix to display.
    ax : matplotlib.axes.Axes, optional
        Axes that receive the image. When omitted, the current axes are used.
    x : ndarray, optional
        Physical coordinates associated with each matrix bin. These are used to
        label the image axes.
    xlabel, ylabel, title : str, optional
        Plot labels.
    cmap : str, default="viridis"
        Matplotlib colormap name.
    colorbar : bool, default=True
        Draw a colorbar alongside the matrix.
    colorbar_label : str, optional
        Label for the colorbar.
    vmin, vmax : float, optional
        Explicit color limits.
    max_ticks : int, default=7
        Maximum number of labeled ticks per axis.
    imshow_kwargs : dict, optional
        Additional keyword arguments forwarded to :meth:`Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that received the plot.
    """
    if ax is None:
        ax = plt.gca()
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"matrix must be square; got shape {arr.shape}")
    opts = {"origin": "lower", "aspect": "auto", "cmap": cmap}
    if imshow_kwargs:
        opts.update(imshow_kwargs)
    if vmin is not None:
        opts["vmin"] = vmin
    if vmax is not None:
        opts["vmax"] = vmax
    im = ax.imshow(arr, **opts)
    positions, labels = _matrix_tick_spec(x, arr.shape[0], max_ticks=max_ticks)
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)
    return ax


def plot_cov_matrix(
    obj: Any,
    *,
    ax=None,
    x: Any = None,
    angunit: str = "deg",
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    max_ticks: int = 7,
    imshow_kwargs: dict[str, Any] | None = None,
):
    r"""Plot a covariance matrix for a nuGUNDAM result or raw covariance array.

    Parameters
    ----------
    obj : AngularCorrelationResult, ProjectedCorrelationResult, or array-like
        Result object exposing ``cov`` or a raw square covariance matrix.
    ax : matplotlib.axes.Axes, optional
        Axes that receive the plot. When omitted, the current axes are used.
    x : array-like, optional
        Bin-coordinate array used when ``obj`` is a raw covariance matrix.
    angunit : {"deg", "arcmin", "arcsec"}, default="deg"
        Angular unit for angular results.
    xlabel, ylabel : str, optional
        Axis-label overrides. By default, both axes are labelled with the
        physical separation associated with the covariance bins.
    title : str, optional
        Plot title. When omitted, a default is used for nuGUNDAM result objects.
    cmap : str, default="viridis"
        Matplotlib colormap name.
    colorbar : bool, default=True
        Draw a colorbar alongside the image.
    colorbar_label : str, optional
        Label for the colorbar. Defaults to ``"Covariance"``.
    vmin, vmax : float, optional
        Explicit color limits.
    max_ticks : int, default=7
        Maximum number of labeled ticks per axis.
    imshow_kwargs : dict, optional
        Additional keyword arguments passed to :meth:`Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that received the plot.
    """
    matrix, xvals, axis_label = _extract_matrix_plot_spec(obj, x=x, angunit=angunit)
    if title is None and isinstance(obj, (AngularCorrelationResult, ProjectedCorrelationResult)):
        title = "Jackknife covariance matrix"
    return _plot_matrix(
        matrix,
        ax=ax,
        x=xvals,
        xlabel=axis_label if xlabel is None else xlabel,
        ylabel=axis_label if ylabel is None else ylabel,
        title=title,
        cmap=cmap,
        colorbar=colorbar,
        colorbar_label="Covariance" if colorbar_label is None else colorbar_label,
        vmin=vmin,
        vmax=vmax,
        max_ticks=max_ticks,
        imshow_kwargs=imshow_kwargs,
    )


def plot_corr_matrix(
    obj: Any,
    *,
    ax=None,
    x: Any = None,
    angunit: str = "deg",
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    max_ticks: int = 7,
    imshow_kwargs: dict[str, Any] | None = None,
):
    r"""Plot a correlation matrix derived from a covariance matrix.

    Parameters
    ----------
    obj : AngularCorrelationResult, ProjectedCorrelationResult, or array-like
        Result object exposing ``cov`` or a raw square covariance matrix.
    ax : matplotlib.axes.Axes, optional
        Axes that receive the plot. When omitted, the current axes are used.
    x : array-like, optional
        Bin-coordinate array used when ``obj`` is a raw covariance matrix.
    angunit : {"deg", "arcmin", "arcsec"}, default="deg"
        Angular unit for angular results.
    xlabel, ylabel : str, optional
        Axis-label overrides. By default, both axes are labelled with the
        physical separation associated with the covariance bins.
    title : str, optional
        Plot title. When omitted, a default is used for nuGUNDAM result objects.
    cmap : str, default="viridis"
        Matplotlib colormap name.
    colorbar : bool, default=True
        Draw a colorbar alongside the image.
    colorbar_label : str, optional
        Label for the colorbar. Defaults to ``"Correlation"``.
    vmin, vmax : float, default=-1, 1
        Color limits for the displayed correlation coefficients.
    max_ticks : int, default=7
        Maximum number of labeled ticks per axis.
    imshow_kwargs : dict, optional
        Additional keyword arguments passed to :meth:`Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that received the plot.
    """
    matrix, xvals, axis_label = _extract_matrix_plot_spec(obj, x=x, angunit=angunit)
    corr = _cov_to_corr(matrix)
    if title is None and isinstance(obj, (AngularCorrelationResult, ProjectedCorrelationResult)):
        title = "Jackknife correlation matrix"
    return _plot_matrix(
        corr,
        ax=ax,
        x=xvals,
        xlabel=axis_label if xlabel is None else xlabel,
        ylabel=axis_label if ylabel is None else ylabel,
        title=title,
        cmap=cmap,
        colorbar=colorbar,
        colorbar_label="Correlation" if colorbar_label is None else colorbar_label,
        vmin=vmin,
        vmax=vmax,
        max_ticks=max_ticks,
        imshow_kwargs=imshow_kwargs,
    )


def plot_jk_regions(
    data: Any = None,
    random: Any = None,
    config: Any = None,
    *,
    sample: Any = None,
    catalog: str = "data",
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "tab20",
    colorbar: bool = True,
    colorbar_label: str = "Jackknife region",
    size: float | None = None,
    alpha: float = 1.0,
    scatter_kwargs: dict[str, Any] | None = None,
):
    r"""Plot jackknife regions on the sky using data or random coordinates.

    Parameters
    ----------
    data, random : table-like, optional
        Input catalogs used to prepare jackknife regions. These are required
        unless ``sample`` is provided directly.
    config : AngularAutoConfig or ProjectedAutoConfig, optional
        Auto-correlation configuration describing how jackknife regions should
        be prepared. The helper currently supports the auto-correlation
        configuration classes because those map naturally onto a single data and
        random catalog pair.
    sample : PreparedAngularSample or PreparedProjectedSample, optional
        Already prepared sample exposing ``ra``, ``dec``, and ``region_id``.
        When supplied, ``data``, ``random``, and ``config`` are ignored.
    catalog : {"data", "random"}, default="data"
        Which catalog to display when preparation is performed internally.
    ax : matplotlib.axes.Axes, optional
        Axes that receive the scatter plot. When omitted, the current axes are
        used.
    title : str, optional
        Plot title. A descriptive default is chosen when omitted.
    xlabel, ylabel : str, optional
        Axis-label overrides. By default, the axes are labelled as right
        ascension and declination in degrees.
    cmap : str, default="tab20"
        Matplotlib colormap name used for the discrete region coloring.
    colorbar : bool, default=True
        Draw a colorbar showing the jackknife region identifiers.
    colorbar_label : str, default="Jackknife region"
        Colorbar label.
    size : float, optional
        Marker size passed to :meth:`Axes.scatter`. When omitted, nuGUNDAM uses a
        small default suited to the chosen catalog.
    alpha : float, default=1.0
        Marker transparency.
    scatter_kwargs : dict, optional
        Additional keyword arguments forwarded to :meth:`Axes.scatter`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that received the scatter plot.

    Raises
    ------
    ValueError
        If jackknife is not enabled, the selected catalog is unavailable, or
        the prepared sample does not contain region identifiers.
    TypeError
        If ``config`` is not one of nuGUNDAM's auto-correlation config classes.
    """
    if ax is None:
        ax = plt.gca()

    cat = str(catalog).strip().lower()
    if cat in {"rand", "randoms"}:
        cat = "random"
    if cat not in {"data", "random"}:
        raise ValueError("catalog must be either 'data' or 'random'")

    meta = None
    if sample is None:
        if config is None or data is None:
            raise ValueError("Provide either a prepared sample via 'sample' or (data, random, config).")
        if not bool(getattr(getattr(config, 'jackknife', None), 'enabled', False)):
            raise ValueError("plot_jk_regions requires config.jackknife.enabled=True when preparing regions internally.")
        from .angular.models import AngularAutoConfig
        from .projected.models import ProjectedAutoConfig
        if isinstance(config, AngularAutoConfig):
            from .angular.prepare import prepare_angular_auto
            data_p, rand_p, meta = prepare_angular_auto(data, random if random is not None else data, config)
        elif isinstance(config, ProjectedAutoConfig):
            from .projected.prepare import prepare_projected_auto
            data_p, rand_p, meta = prepare_projected_auto(data, random if random is not None else data, config)
        else:
            raise TypeError("plot_jk_regions currently supports AngularAutoConfig and ProjectedAutoConfig.")
        sample = data_p if cat == 'data' else rand_p
    if sample is None:
        raise ValueError("Could not determine which sample to plot.")
    if getattr(sample, 'region_id', None) is None:
        raise ValueError("Prepared sample does not contain jackknife region ids.")

    ra = _as_1d_array(getattr(sample, 'ra'), name='ra')
    dec = _as_1d_array(getattr(sample, 'dec'), name='dec')
    region_id = np.asarray(getattr(sample, 'region_id'), dtype=int)
    if region_id.ndim != 1 or len(region_id) != len(ra):
        raise ValueError("region_id must be a one-dimensional array matching the sample length.")
    nreg = int(region_id.max()) + 1 if region_id.size else 0

    scatter_opts = {"linewidths": 0.0, "alpha": alpha}
    scatter_opts["s"] = (3.0 if cat == 'data' else 1.0) if size is None else size
    if scatter_kwargs:
        scatter_opts.update(scatter_kwargs)

    cmap_obj = plt.get_cmap(cmap, max(nreg, 1))
    sc = ax.scatter(ra, dec, c=region_id, cmap=cmap_obj, vmin=-0.5, vmax=max(nreg - 0.5, 0.5), **scatter_opts)
    ax.set_xlabel(_sky_xlabel() if xlabel is None else xlabel)
    ax.set_ylabel(_sky_ylabel() if ylabel is None else ylabel)
    if title is None:
        source = None if meta is None else meta.get('jk_region_source')
        suffix = f" ({source})" if source else ""
        title = f"Jackknife regions on {cat}{suffix}"
    ax.set_title(title)
    if colorbar:
        cbar = ax.figure.colorbar(sc, ax=ax)
        cbar.set_label(colorbar_label)
    return ax


def _as_2d_array(values: Any, *, name: str) -> np.ndarray:
    """
    Convert the supplied value to 2d array.
    
    Parameters
    ----------
    values : object
        Value for ``values``.
    name : object
        Value for ``name``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D; got shape {arr.shape}")
    return arr


def _infer_projected_estimator(result: Any, estimator: str | None) -> str:
    """
    Infer projected estimator.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    estimator : object
        Value for ``estimator``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if estimator is not None:
        return estimator
    if hasattr(result, "estimator"):
        est = getattr(result, "estimator")
        if est is not None:
            return str(est)
    metadata = getattr(result, "metadata", {}) or {}
    cfg = metadata.get("config") if isinstance(metadata, Mapping) else None
    if isinstance(cfg, Mapping):
        est = cfg.get("estimator")
        if est is not None:
            return str(est)
    raise ValueError("An estimator is required to plot xi(r_p, pi) from this object")


def _projected_xi2d_from_result(result: Any, *, estimator: str | None) -> tuple[np.ndarray, str]:
    """
    Projected xi2d from result.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    estimator : object
        Value for ``estimator``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    est = _infer_projected_estimator(result, estimator)
    counts = result.counts if isinstance(result, ProjectedCorrelationResult) else result
    metadata = getattr(result, "metadata", {}) or {}
    counts_meta = getattr(counts, "metadata", {}) or {}

    if isinstance(counts, ProjectedAutoCounts):
        sum_w = metadata.get("sum_w_data", counts_meta.get("sum_w_data"))
        sum_w2 = metadata.get("sum_w2_data", counts_meta.get("sum_w2_data"))
        weighted = bool(metadata.get("weighted", counts_meta.get("data_weighted", False)))
        if weighted and sum_w is None and sum_w2 is None:
            raise ValueError(
                "Weighted projected auto xi(r_p, pi) plotting requires stored weight sums; "
                "plot the correlation result directly or rerun with a result that preserves metadata."
            )
        xi2d = compute_auto_xi2d(counts, estimator=est, sum_w_data=sum_w, sum_w2_data=sum_w2)
        return xi2d, est

    if isinstance(counts, ProjectedCrossCounts):
        sum_w1 = metadata.get("sum_w1", counts_meta.get("sum_w1"))
        sum_w2 = metadata.get("sum_w2", counts_meta.get("sum_w2"))
        weighted = bool(metadata.get("weighted", False) or sum_w1 is not None or sum_w2 is not None)
        if weighted and sum_w1 is None and sum_w2 is None:
            raise ValueError(
                "Weighted projected cross xi(r_p, pi) plotting requires stored weight sums; "
                "plot the correlation result directly or rerun with preserved metadata."
            )
        xi2d = compute_cross_xi2d(counts, estimator=est, sum_w1=sum_w1, sum_w2=sum_w2)
        return xi2d, est

    raise TypeError(f"Unsupported result type for 2D xi plotting: {type(result)!r}")


def _resolve_projected_2d_spec(
    result: Any,
    *,
    which: str | None = None,
    estimator: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    """
    Resolve projected 2d spec.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    which : object, optional
        Value for ``which``. This argument is keyword-only.
    estimator : object, optional
        Value for ``estimator``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if isinstance(result, ProjectedCorrelationResult):
        rp_edges = _as_1d_array(result.rp_edges, name="rp_edges")
        pi_edges = _as_1d_array(result.counts.pi_edges, name="pi_edges")
    else:
        rp_edges = _as_1d_array(getattr(result, "rp_edges"), name="rp_edges")
        pi_edges = _as_1d_array(getattr(result, "pi_edges"), name="pi_edges")
    field = which or ("xi" if isinstance(result, ProjectedCorrelationResult) else None)

    if field == "xi":
        z2d, used_estimator = _projected_xi2d_from_result(result, estimator=estimator)
        label = r"$\xi(r_p, \pi)$"
        return rp_edges, pi_edges, _as_2d_array(z2d, name="xi2d"), _projected_xlabel(), _projected_ylabel(), label

    if field is None:
        if isinstance(result, (ProjectedAutoCounts, ProjectedAutoCountsResult)):
            field = _pick_auto_count_field(result)
        elif isinstance(result, (ProjectedCrossCounts, ProjectedCrossCountsResult)):
            field = _pick_cross_count_field(result)
        else:
            raise TypeError(f"Unsupported result type for 2D plotting: {type(result)!r}")

    try:
        values = getattr(result, field)
    except AttributeError as exc:
        raise ValueError(f"Field {field!r} is not available for 2D plotting") from exc
    if values is None:
        raise ValueError(f"Field {field!r} is not available for 2D plotting")
    return rp_edges, pi_edges, _as_2d_array(values, name=field), _projected_xlabel(), _projected_ylabel(), _count_ylabel(field)


def _gaussian_smooth2d(values: np.ndarray, sigma: float | None) -> np.ndarray:
    """
    Gaussian smooth2d.
    
    Parameters
    ----------
    values : object
        Value for ``values``.
    sigma : object
        Value for ``sigma``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if sigma is None:
        return values
    sigma = float(sigma)
    if sigma <= 0.0:
        return values
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * np.square(x / sigma))
    kernel /= kernel.sum()
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    for axis in (0, 1):
        pad = [(0, 0)] * arr.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(arr, pad, mode="reflect")
        arr = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis, padded)
    return arr


def _mirror_projected_grid(rp_edges: np.ndarray, pi_edges: np.ndarray, z2d: np.ndarray, *, mirror: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mirror projected grid.
    
    Parameters
    ----------
    rp_edges : object
        Value for ``rp_edges``.
    pi_edges : object
        Value for ``pi_edges``.
    z2d : object
        Value for ``z2d``.
    mirror : object
        Value for ``mirror``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    mirror = mirror.lower()
    qtr = np.asarray(z2d, dtype=float).T
    if mirror == "none":
        return rp_edges, pi_edges, qtr
    if mirror == "pi":
        pi_m = np.concatenate((-pi_edges[:0:-1], pi_edges))
        qbot = np.flipud(qtr)
        return rp_edges, pi_m, np.vstack((qbot, qtr))
    if mirror == "four":
        rp_m = np.concatenate((-rp_edges[:0:-1], rp_edges))
        pi_m = np.concatenate((-pi_edges[:0:-1], pi_edges))
        qtl = np.fliplr(qtr)
        qbl = np.flipud(qtl)
        qbr = np.fliplr(qbl)
        qtop = np.hstack((qtl, qtr))
        qbot = np.hstack((qbl, qbr))
        return rp_m, pi_m, np.vstack((qbot, qtop))
    raise ValueError("mirror must be one of 'none', 'pi', or 'four'")


def _build_2d_norm(values: np.ndarray, *, color_scale: str, vmin: float | None, vmax: float | None, linthresh: float | None):
    """
    Build 2d norm.
    
    Parameters
    ----------
    values : object
        Value for ``values``.
    color_scale : object
        Value for ``color_scale``. This argument is keyword-only.
    vmin : object
        Value for ``vmin``. This argument is keyword-only.
    vmax : object
        Value for ``vmax``. This argument is keyword-only.
    linthresh : object
        Value for ``linthresh``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    scale = color_scale.lower()
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite values available to plot")
    if scale == "linear":
        return values, None
    if scale == "log-positive":
        positive = finite[finite > 0.0]
        if positive.size == 0:
            raise ValueError("color_scale='log-positive' requires at least one positive value")
        lo = float(vmin) if vmin is not None else float(np.min(positive))
        hi = float(vmax) if vmax is not None else float(np.max(positive))
        if not hi > lo:
            hi = lo * 10.0 if lo > 0.0 else 1.0
        masked = np.ma.masked_less_equal(values, 0.0)
        return masked, mcolors.LogNorm(vmin=lo, vmax=hi)
    if scale == "symlog":
        abs_positive = np.abs(finite[finite != 0.0])
        if abs_positive.size == 0:
            return values, None
        hi = float(vmax) if vmax is not None else float(np.max(abs_positive))
        lo = float(vmin) if vmin is not None else -hi
        if linthresh is None:
            linthresh = max(float(np.min(abs_positive)), hi * 1.0e-3)
        return values, mcolors.SymLogNorm(linthresh=float(linthresh), vmin=lo, vmax=hi)
    raise ValueError("color_scale must be one of 'linear', 'symlog', or 'log-positive'")


def _auto_contour_levels(values: np.ndarray, *, color_scale: str, levels: int | Sequence[float]):
    """
    Auto contour levels.
    
    Parameters
    ----------
    values : object
        Value for ``values``.
    color_scale : object
        Value for ``color_scale``. This argument is keyword-only.
    levels : object
        Value for ``levels``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if not isinstance(levels, int):
        return np.asarray(list(levels), dtype=float)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite values available to derive contour levels")
    if color_scale.lower() == "log-positive":
        positive = finite[finite > 0.0]
        if positive.size == 0:
            raise ValueError("Positive contour levels require at least one positive value")
        lo = float(np.min(positive))
        hi = float(np.max(positive))
        if not hi > lo:
            return np.asarray([lo], dtype=float)
        return np.unique(np.geomspace(lo, hi, int(levels)))
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if not hi > lo:
        return np.asarray([lo], dtype=float)
    return np.unique(np.linspace(lo, hi, int(levels)))

def _pick_auto_count_field(result) -> str:
    """
    Pick auto count field.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    for name in ("dd", "dr", "rr"):
        if getattr(result, name, None) is not None:
            return name
    raise ValueError("No count array available to plot")


def _pick_cross_count_field(result) -> str:
    """
    Pick cross count field.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    for name in ("d1d2", "d1r2", "r1d2", "r1r2"):
        if getattr(result, name, None) is not None:
            return name
    raise ValueError("No cross-count array available to plot")


def _maybe_get_projected_spec(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str, str] | None:
    """
    Maybe get projected spec.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if hasattr(result, "rp_centers") and hasattr(result, "wp"):
        return (
            _as_1d_array(getattr(result, "rp_centers"), name="rp_centers"),
            _as_1d_array(getattr(result, "wp"), name="wp"),
            _as_1d_array(getattr(result, "wp_err"), name="wp_err") if getattr(result, "wp_err", None) is not None else None,
            r"$r_p\,[h^{-1}\,\mathrm{Mpc}]$",
            r"$w_p(r_p)$",
        )
    return None


def _result_plot_spec(
    result: Any,
    *,
    which: str | None = None,
    angunit: str = "deg",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str, str]:
    """
    Result plot spec.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    which : object, optional
        Value for ``which``. This argument is keyword-only.
    angunit : object, optional
        Value for ``angunit``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    projected = _maybe_get_projected_spec(result)
    if projected is not None:
        return projected

    if isinstance(result, AngularCorrelationResult):
        x = _as_1d_array(result.theta_centers, name="theta_centers") * _ANGULAR_UNIT_SCALE[angunit]
        y = _as_1d_array(result.wtheta, name="wtheta")
        yerr = _as_1d_array(result.wtheta_err, name="wtheta_err") if result.wtheta_err is not None else None
        default_xlabel = _angular_xlabel(angunit)
        default_ylabel = r"$w(\theta)$"
    elif isinstance(result, (ProjectedCorrelationResult,)) or (hasattr(result, 'wp') and hasattr(result, 'rp_centers')):
        x = _as_1d_array(result.rp_centers, name="rp_centers")
        y = _as_1d_array(result.wp, name="wp")
        yerr = _as_1d_array(result.wp_err, name="wp_err") if getattr(result, 'wp_err', None) is not None else None
        default_xlabel = _projected_xlabel()
        default_ylabel = r"$w_p(r_p)$"
    elif isinstance(result, (AngularAutoCounts, AngularAutoCountsResult)):
        field = which or _pick_auto_count_field(result)
        x = _as_1d_array(result.theta_centers, name="theta_centers") * _ANGULAR_UNIT_SCALE[angunit]
        y = _as_1d_array(getattr(result, field), name=field)
        yerr = None
        default_xlabel = _angular_xlabel(angunit)
        default_ylabel = _count_ylabel(field)
    elif isinstance(result, (AngularCrossCounts, AngularCrossCountsResult)):
        field = which or _pick_cross_count_field(result)
        x = _as_1d_array(result.theta_centers, name="theta_centers") * _ANGULAR_UNIT_SCALE[angunit]
        y = _as_1d_array(getattr(result, field), name=field)
        yerr = None
        default_xlabel = _angular_xlabel(angunit)
        default_ylabel = _count_ylabel(field)
    elif isinstance(result, (ProjectedAutoCounts, ProjectedAutoCountsResult)) or (hasattr(result, 'dd') and hasattr(result, 'rp_centers')):
        field = which or _pick_auto_count_field(result)
        x = _as_1d_array(result.rp_centers, name="rp_centers")
        y = _as_1d_array(getattr(result, field), name=field)
        yerr = None
        default_xlabel = _projected_xlabel()
        default_ylabel = _count_ylabel(field)
    elif isinstance(result, (ProjectedCrossCounts, ProjectedCrossCountsResult)) or (hasattr(result, 'd1d2') and hasattr(result, 'rp_centers')):
        field = which or _pick_cross_count_field(result)
        x = _as_1d_array(result.rp_centers, name="rp_centers")
        y = _as_1d_array(getattr(result, field), name=field)
        yerr = None
        default_xlabel = _projected_xlabel()
        default_ylabel = _count_ylabel(field)
    else:
        raise TypeError(f"Unsupported result type for plotting: {type(result)!r}")

    return x, y, yerr, default_xlabel, default_ylabel


def _normalize_curve_specs(curves: Any) -> list[dict[str, Any]]:
    """
    Normalize curve specs.
    
    Parameters
    ----------
    curves : object
        Value for ``curves``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if isinstance(curves, Mapping):
        items = [{"key": str(key), "result": value, "label": str(key)} for key, value in curves.items()]
    elif isinstance(curves, Sequence) and not isinstance(curves, (str, bytes)):
        items = []
        for idx, entry in enumerate(curves):
            if isinstance(entry, Mapping):
                if "result" not in entry:
                    raise ValueError("Each curve spec must include a 'result' entry")
                spec = dict(entry)
                spec.setdefault("key", str(spec.get("label", idx)))
                items.append(spec)
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 2:
                key, result = entry
                items.append({"key": str(key), "result": result, "label": str(key)})
            else:
                items.append({"key": str(idx), "result": entry, "label": None})
    else:
        raise TypeError("curves must be a mapping or a sequence of curve specifications")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for idx, spec in enumerate(items):
        key = str(spec.get("key", idx))
        if key in seen:
            raise ValueError(f"Duplicate curve key: {key!r}")
        seen.add(key)
        result = spec["result"]
        norm = {
            "key": key,
            "result": result,
            "which": spec.get("which"),
            "label": spec.get("label"),
            "xlabel": spec.get("xlabel"),
            "ylabel": spec.get("ylabel"),
            "angunit": spec.get("angunit", "deg"),
            "errors": spec.get("errors"),
            "shift": float(spec.get("shift", 0.0)),
            "line_kwargs": dict(spec.get("line_kwargs") or {}),
            "errorbar_kwargs": dict(spec.get("errorbar_kwargs") or {}),
            "band_kwargs": dict(spec.get("band_kwargs") or {}),
        }
        reserved = {
            "key",
            "result",
            "which",
            "label",
            "xlabel",
            "ylabel",
            "angunit",
            "errors",
            "shift",
            "line_kwargs",
            "errorbar_kwargs",
            "band_kwargs",
        }
        plot_kwargs = {k: v for k, v in spec.items() if k not in reserved}
        norm["plot_kwargs"] = plot_kwargs
        normalized.append(norm)
    if not normalized:
        raise ValueError("At least one curve is required")
    return normalized


def _normalize_ratio_specs(ratios: Any) -> list[dict[str, Any]]:
    """
    Normalize ratio specs.
    
    Parameters
    ----------
    ratios : object
        Value for ``ratios``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if ratios is None:
        raise ValueError("ratios must be provided")
    if isinstance(ratios, Mapping):
        iterable = [ratios]
    else:
        iterable = list(ratios)
    if not iterable:
        raise ValueError("At least one ratio specification is required")

    normalized: list[dict[str, Any]] = []
    for idx, entry in enumerate(iterable):
        if isinstance(entry, Mapping):
            if "numerator" in entry:
                numerator = entry["numerator"]
            elif "num" in entry:
                numerator = entry["num"]
            else:
                raise ValueError("Each ratio spec must include 'numerator' (or 'num')")
            if "denominator" in entry:
                denominator = entry["denominator"]
            elif "den" in entry:
                denominator = entry["den"]
            else:
                raise ValueError("Each ratio spec must include 'denominator' (or 'den')")
            spec = dict(entry)
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 2:
            numerator, denominator = entry
            spec = {}
        else:
            raise TypeError("Each ratio spec must be a mapping or a (numerator, denominator) pair")

        norm = {
            "numerator": str(numerator),
            "denominator": str(denominator),
            "label": spec.get("label"),
            "errors": spec.get("errors"),
            "shift": float(spec.get("shift", 0.0)),
            "line_kwargs": dict(spec.get("line_kwargs") or {}),
            "errorbar_kwargs": dict(spec.get("errorbar_kwargs") or {}),
            "band_kwargs": dict(spec.get("band_kwargs") or {}),
        }
        reserved = {
            "numerator",
            "num",
            "denominator",
            "den",
            "label",
            "errors",
            "shift",
            "line_kwargs",
            "errorbar_kwargs",
            "band_kwargs",
        }
        plot_kwargs = {k: v for k, v in spec.items() if k not in reserved}
        norm["plot_kwargs"] = plot_kwargs
        normalized.append(norm)
    return normalized


def _compute_ratio_yerr(
    num_y: np.ndarray,
    den_y: np.ndarray,
    num_yerr: np.ndarray | None,
    den_yerr: np.ndarray | None,
) -> np.ndarray | None:
    """
    Compute ratio yerr.
    
    Parameters
    ----------
    num_y : object
        Value for ``num_y``.
    den_y : object
        Value for ``den_y``.
    num_yerr : object
        Value for ``num_yerr``.
    den_yerr : object
        Value for ``den_yerr``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if num_yerr is None and den_yerr is None:
        return None
    rel2 = np.zeros_like(num_y, dtype=float)
    if num_yerr is not None:
        rel2 += np.square(num_yerr / num_y)
    if den_yerr is not None:
        rel2 += np.square(den_yerr / den_y)
    return np.abs(num_y / den_y) * np.sqrt(rel2)


def _prepare_plotcf_data(
    x: Any,
    y: Any,
    yerr: Any = None,
    *,
    loglog: bool = True,
    xscale: str | None = None,
    yscale: str | None = None,
    errors: str = "bar",
    shift: float = 0.0,
    mask_nonpositive: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str, str]:
    """
    Prepare plotcf data.
    
    Parameters
    ----------
    x : object
        Value for ``x``.
    y : object
        Value for ``y``.
    yerr : object, optional
        Value for ``yerr``.
    loglog : object, optional
        Value for ``loglog``. This argument is keyword-only.
    xscale : object, optional
        Value for ``xscale``. This argument is keyword-only.
    yscale : object, optional
        Value for ``yscale``. This argument is keyword-only.
    errors : object, optional
        Value for ``errors``. This argument is keyword-only.
    shift : object, optional
        Value for ``shift``. This argument is keyword-only.
    mask_nonpositive : object, optional
        Value for ``mask_nonpositive``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    x_arr = _as_1d_array(x, name="x")
    y_arr = _as_1d_array(y, name="y")
    if len(x_arr) != len(y_arr):
        raise ValueError(f"x has length {len(x_arr)} but y has length {len(y_arr)}")
    yerr_arr = None if yerr is None else _broadcast_yerr(yerr, len(y_arr))

    final_xscale = xscale or ("log" if loglog else "linear")
    final_yscale = yscale or ("log" if loglog else "linear")
    x_arr = _shift_log_x(x_arr, shift)
    x_arr, y_arr, yerr_arr = _mask_plot_values(
        x_arr,
        y_arr,
        yerr_arr,
        xscale=final_xscale,
        yscale=final_yscale,
        errors=errors,
        mask_nonpositive=mask_nonpositive,
    )
    return x_arr, y_arr, yerr_arr, final_xscale, final_yscale


def _match_common_x(
    num_x: np.ndarray,
    den_x: np.ndarray,
    *,
    rtol: float = 1.0e-12,
    atol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match common x.
    
    Parameters
    ----------
    num_x : object
        Value for ``num_x``.
    den_x : object
        Value for ``den_x``.
    rtol : object, optional
        Value for ``rtol``. This argument is keyword-only.
    atol : object, optional
        Value for ``atol``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if num_x.shape == den_x.shape and np.allclose(num_x, den_x, rtol=rtol, atol=atol):
        idx = np.arange(num_x.size, dtype=int)
        return num_x.copy(), idx, idx

    matches_num: list[int] = []
    matches_den: list[int] = []
    common_x: list[float] = []
    j_start = 0
    for i, xval in enumerate(num_x):
        for j in range(j_start, len(den_x)):
            if np.isclose(xval, den_x[j], rtol=rtol, atol=atol):
                matches_num.append(i)
                matches_den.append(j)
                common_x.append(float(xval))
                j_start = j + 1
                break

    if not common_x:
        raise ValueError("Ratio numerator and denominator do not share any plotted x values")

    return np.asarray(common_x, dtype=float), np.asarray(matches_num, dtype=int), np.asarray(matches_den, dtype=int)


def plotcf(
    x: Any,
    y: Any,
    yerr: Any = None,
    *,
    ax=None,
    label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    loglog: bool = True,
    xscale: str | None = None,
    yscale: str | None = None,
    errors: str = "bar",
    shift: float = 0.0,
    mask_nonpositive: bool = True,
    line_kwargs: dict[str, Any] | None = None,
    errorbar_kwargs: dict[str, Any] | None = None,
    band_kwargs: dict[str, Any] | None = None,
    **plot_kwargs,
):
    """Plot a single correlation-like curve on an existing axes.

    This is the lowest-level plotting helper exposed by nuGUNDAM. It accepts
    already prepared ``x``/``y`` arrays plus optional symmetric uncertainties
    and draws a single curve on the supplied axes. Higher-level helpers such as
    :func:`plot_result` and :func:`plot_compare_ratio` build on top of this
    function.

    Parameters
    ----------
    x, y : array-like
        One-dimensional coordinates of the curve to plot.
    yerr : array-like or scalar, optional
        Symmetric uncertainties for ``y``. When provided, they can be drawn as
        error bars or as a filled band depending on ``errors``.
    ax : matplotlib.axes.Axes, optional
        Axes that receive the plot. When omitted, the current axes are used.
    label : str, optional
        Legend label for the main curve.
    xlabel, ylabel : str, optional
        Axis labels. Labels are only set when the corresponding argument is not
        ``None``.
    loglog : bool, default=True
        Convenience switch that sets both axes to logarithmic scale unless one
        of ``xscale`` or ``yscale`` is passed explicitly.
    xscale, yscale : {"linear", "log"}, optional
        Explicit axis-scale overrides.
    errors : {"bar", "band", "none"}, default="bar"
        Uncertainty rendering mode. ``"bar"`` uses ``Axes.errorbar``,
        ``"band"`` uses ``Axes.fill_between``, and ``"none"`` suppresses the
        visual uncertainty layer even when ``yerr`` is available.
    shift : float, default=0.0
        Horizontal offset expressed in units of the median logarithmic bin
        spacing. This is useful when overlaying several measurements on the
        same bins and wanting to separate the markers visually.
    mask_nonpositive : bool, default=True
        When ``True``, points that cannot be shown on the chosen axis scales are
        dropped before plotting. On logarithmic y-axes this also removes points
        for which ``y - yerr <= 0`` when uncertainties are drawn as bars or
        bands.
    line_kwargs, errorbar_kwargs, band_kwargs : dict, optional
        Optional keyword dictionaries routed respectively to ``Axes.plot``,
        ``Axes.errorbar`` and ``Axes.fill_between``. Any additional keyword
        arguments passed directly to :func:`plotcf` are forwarded to the main
        line call as well.
    **plot_kwargs
        Extra matplotlib keyword arguments forwarded to ``Axes.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that received the plot.

    Raises
    ------
    ValueError
        If the input arrays do not have matching lengths, the uncertainty mode
        is invalid, or no plottable points remain after masking.
    """
    if ax is None:
        ax = plt.gca()

    x_arr, y_arr, yerr_arr, xscale, yscale = _prepare_plotcf_data(
        x,
        y,
        yerr=yerr,
        loglog=loglog,
        xscale=xscale,
        yscale=yscale,
        errors=errors,
        shift=shift,
        mask_nonpositive=mask_nonpositive,
    )
    if x_arr.size == 0:
        raise ValueError("No plottable points remain after masking")

    line_opts = _merge_kwargs(plot_kwargs, line_kwargs)
    line_list = ax.plot(x_arr, y_arr, label=label, **line_opts)
    line = line_list[0]

    if yerr_arr is not None and errors != "none":
        color = line.get_color()
        if errors == "bar":
            err_opts = {"fmt": "none", "ecolor": color, "elinewidth": line.get_linewidth(), "capsize": 3}
            err_opts.update(errorbar_kwargs or {})
            ax.errorbar(x_arr, y_arr, yerr=yerr_arr, **err_opts)
        elif errors == "band":
            fill_opts = {"color": color, "alpha": 0.2}
            fill_opts.update(band_kwargs or {})
            ax.fill_between(x_arr, y_arr - yerr_arr, y_arr + yerr_arr, **fill_opts)
        else:
            raise ValueError("errors must be one of 'bar', 'band', or 'none'")
    elif errors not in {"bar", "band", "none"}:
        raise ValueError("errors must be one of 'bar', 'band', or 'none'")

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax


def plot_result(
    result: Any,
    *,
    ax=None,
    which: str | None = None,
    label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    angunit: str = "deg",
    errors: str = "bar",
    shift: float = 0.0,
    loglog: bool = True,
    xscale: str | None = None,
    yscale: str | None = None,
    mask_nonpositive: bool = True,
    line_kwargs: dict[str, Any] | None = None,
    errorbar_kwargs: dict[str, Any] | None = None,
    band_kwargs: dict[str, Any] | None = None,
    **plot_kwargs,
):
    """Plot a nuGUNDAM result or count object.

    This is the result-aware companion to :func:`plotcf`. It inspects the input
    object, extracts the appropriate x/y arrays, chooses sensible default axis
    labels, and then forwards the actual rendering to :func:`plotcf`.

    Parameters
    ----------
    result : object
        Supported inputs currently include :class:`AngularCorrelationResult`,
        :class:`AngularAutoCounts`, :class:`AngularCrossCounts`, and projected
        correlation-like objects exposing ``rp_centers``/``wp`` attributes.
    ax : matplotlib.axes.Axes, optional
        Axes that receive the plot. When omitted, the current axes are used.
    which : str, optional
        For count objects, name of the count array to plot. When omitted,
        nuGUNDAM chooses the first available standard field.
    label, xlabel, ylabel : str, optional
        Legend label and axis-label overrides.
    angunit : {"deg", "arcmin", "arcsec"}, default="deg"
        Angular unit used when plotting angular results.
    errors, shift, loglog, xscale, yscale, mask_nonpositive : optional
        Plotting controls forwarded to :func:`plotcf`.
    line_kwargs, errorbar_kwargs, band_kwargs : dict, optional
        Optional matplotlib keyword dictionaries forwarded to the lower-level
        plotting calls.
    **plot_kwargs
        Extra matplotlib keyword arguments forwarded to the main line call.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that received the plot.
    """
    x, y, yerr, default_xlabel, default_ylabel = _result_plot_spec(
        result,
        which=which,
        angunit=angunit,
    )
    return plotcf(
        x,
        y,
        yerr=yerr,
        ax=ax,
        label=label,
        xlabel=default_xlabel if xlabel is None else xlabel,
        ylabel=default_ylabel if ylabel is None else ylabel,
        loglog=loglog,
        xscale=xscale,
        yscale=yscale,
        errors=errors,
        shift=shift,
        mask_nonpositive=mask_nonpositive,
        line_kwargs=line_kwargs,
        errorbar_kwargs=errorbar_kwargs,
        band_kwargs=band_kwargs,
        **plot_kwargs,
    )


def plotcf2d(
    rp_edges: Any,
    pi_edges: Any,
    z2d: Any,
    *,
    ax=None,
    mirror: str = "four",
    smoothing: float | None = None,
    contours: bool = True,
    levels: int | Sequence[float] = 15,
    cmap: str = "RdBu_r",
    color_scale: str = "log-positive",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    linthresh: float | None = None,
    contour_kwargs: dict[str, Any] | None = None,
    pcolor_kwargs: dict[str, Any] | None = None,
):
    r"""Plot a projected-space 2D field on an ``(r_p, \pi)`` grid.

    Parameters
    ----------
    rp_edges, pi_edges : array-like
        One-dimensional bin-edge arrays for projected and line-of-sight
        separations.
    z2d : array-like, shape (nrp, npi)
        Two-dimensional field defined on the positive-quadrant nuGUNDAM grid.
    mirror : {"none", "pi", "four"}, default="four"
        Mirroring mode applied for display only.
    smoothing : float, optional
        Gaussian smoothing sigma in display-pixel units. ``None`` leaves the
        field untouched.
    contours : bool, default=True
        Overlay contour lines computed from the displayed field.
    levels : int or sequence, default=15
        Contour level specification.
    color_scale : {"linear", "symlog", "log-positive"}, default="log-positive"
        Color normalization applied to the heatmap.
    """
    if ax is None:
        ax = plt.gca()
    rp_e = _as_1d_array(rp_edges, name="rp_edges")
    pi_e = _as_1d_array(pi_edges, name="pi_edges")
    z_arr = _as_2d_array(z2d, name="z2d")
    if z_arr.shape != (len(rp_e) - 1, len(pi_e) - 1):
        raise ValueError(
            f"z2d has shape {z_arr.shape} but expected {(len(rp_e) - 1, len(pi_e) - 1)} "
            "for the supplied bin edges"
        )

    x_edges, y_edges, z_plot = _mirror_projected_grid(rp_e, pi_e, z_arr, mirror=mirror)
    z_plot = _gaussian_smooth2d(z_plot, smoothing)
    color_values, norm = _build_2d_norm(z_plot, color_scale=color_scale, vmin=vmin, vmax=vmax, linthresh=linthresh)

    mesh_opts = {"shading": "auto", "cmap": cmap}
    if pcolor_kwargs:
        mesh_opts.update(pcolor_kwargs)
    mesh = ax.pcolormesh(x_edges, y_edges, color_values, norm=norm, **mesh_opts)

    if contours:
        contour_opts = {"colors": "k", "linewidths": 0.8, "linestyles": "solid"}
        if contour_kwargs:
            contour_opts.update(contour_kwargs)
        contour_levels = _auto_contour_levels(z_plot, color_scale=color_scale, levels=levels)
        finite_z = np.asarray(z_plot, dtype=float)
        finite_z = finite_z[np.isfinite(finite_z)]
        if contour_levels.size > 0 and finite_z.size > 0 and float(np.min(finite_z)) != float(np.max(finite_z)):
            ax.contour(x_edges[:-1], y_edges[:-1], z_plot, levels=contour_levels, **contour_opts)

    ax.set_xlabel(_projected_xlabel() if xlabel is None else xlabel)
    ax.set_ylabel(_projected_ylabel() if ylabel is None else ylabel)

    if colorbar:
        cbar = ax.figure.colorbar(mesh, ax=ax)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)
    return ax


def plot_result2d(
    result: Any,
    *,
    ax=None,
    which: str | None = None,
    estimator: str | None = None,
    mirror: str = "four",
    smoothing: float | None = None,
    contours: bool = True,
    levels: int | Sequence[float] = 15,
    cmap: str = "RdBu_r",
    color_scale: str = "log-positive",
    colorbar: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    linthresh: float | None = None,
    contour_kwargs: dict[str, Any] | None = None,
    pcolor_kwargs: dict[str, Any] | None = None,
):
    """
    Plot a projected 2D field extracted from a result or count object.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    ax : object, optional
        Value for ``ax``. This argument is keyword-only.
    which : object, optional
        Value for ``which``. This argument is keyword-only.
    estimator : object, optional
        Value for ``estimator``. This argument is keyword-only.
    mirror : object, optional
        Value for ``mirror``. This argument is keyword-only.
    smoothing : object, optional
        Value for ``smoothing``. This argument is keyword-only.
    contours : object, optional
        Value for ``contours``. This argument is keyword-only.
    levels : object, optional
        Value for ``levels``. This argument is keyword-only.
    cmap : object, optional
        Value for ``cmap``. This argument is keyword-only.
    color_scale : object, optional
        Value for ``color_scale``. This argument is keyword-only.
    colorbar : object, optional
        Value for ``colorbar``. This argument is keyword-only.
    xlabel : object, optional
        Value for ``xlabel``. This argument is keyword-only.
    ylabel : object, optional
        Value for ``ylabel``. This argument is keyword-only.
    colorbar_label : object, optional
        Value for ``colorbar_label``. This argument is keyword-only.
    vmin : object, optional
        Value for ``vmin``. This argument is keyword-only.
    vmax : object, optional
        Value for ``vmax``. This argument is keyword-only.
    linthresh : object, optional
        Value for ``linthresh``. This argument is keyword-only.
    contour_kwargs : object, optional
        Value for ``contour_kwargs``. This argument is keyword-only.
    pcolor_kwargs : object, optional
        Value for ``pcolor_kwargs``. This argument is keyword-only.
    
    Returns
    -------
    matplotlib.axes.Axes
        Object returned by this helper.
    """
    rp_edges, pi_edges, z2d, default_xlabel, default_ylabel, default_clabel = _resolve_projected_2d_spec(
        result,
        which=which,
        estimator=estimator,
    )
    return plotcf2d(
        rp_edges,
        pi_edges,
        z2d,
        ax=ax,
        mirror=mirror,
        smoothing=smoothing,
        contours=contours,
        levels=levels,
        cmap=cmap,
        color_scale=color_scale,
        colorbar=colorbar,
        colorbar_label=default_clabel if colorbar_label is None else colorbar_label,
        xlabel=default_xlabel if xlabel is None else xlabel,
        ylabel=default_ylabel if ylabel is None else ylabel,
        vmin=vmin,
        vmax=vmax,
        linthresh=linthresh,
        contour_kwargs=contour_kwargs,
        pcolor_kwargs=pcolor_kwargs,
    )


def plot_compare_ratio(
    curves: Any,
    *,
    ratios: Any,
    axes: tuple[Any, Any] | None = None,
    figsize: tuple[float, float] = (7.0, 6.0),
    height_ratios: tuple[float, float] = (3.0, 1.0),
    hspace: float = 0.05,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ratio_ylabel: str = "ratio",
    angunit: str = "deg",
    loglog: bool = True,
    xscale: str | None = None,
    yscale: str | None = None,
    ratio_xscale: str | None = None,
    ratio_yscale: str = "linear",
    top_errors: str = "bar",
    ratio_errors: str = "bar",
    mask_nonpositive: bool = True,
    ratio_mask_nonpositive: bool = True,
    ratio_reference: float | None = 1.0,
    ratio_reference_kwargs: dict[str, Any] | None = None,
    legend: bool = True,
    legend_kwargs: dict[str, Any] | None = None,
    ratio_legend: bool = False,
    ratio_legend_kwargs: dict[str, Any] | None = None,
):
    """Plot multiple curves in a top panel and selected ratios below.

    This is a figure-level helper for the common workflow of comparing several
    correlation measurements in one panel and showing only a chosen subset of
    their ratios in a second panel with a shared x-axis. The upper panel is
    built from explicit curve specifications, while the lower panel is driven by
    explicit numerator/denominator key pairs. Ratios are computed only from the
    bins that are actually plottable in the upper panel for both participating
    curves, so the points shown downstairs always correspond to points that are
    visible upstairs.

    Parameters
    ----------
    curves : mapping or sequence
        Curves to draw in the upper panel. Supported forms are:

        - ``{"key": result, ...}``
        - ``[("key", result), ...]``
        - ``[{"key": ..., "result": ..., ...}, ...]``

        Each curve dictionary must include ``result`` and may also include any
        keyword understood by :func:`plot_result`, plus the bookkeeping fields
        ``key`` and ``label``. The ``key`` is what ratio specifications use to
        refer to that curve.
    ratios : sequence or mapping
        Ratio specifications for the lower panel. Each entry can be either
        ``("num_key", "den_key")`` or a dictionary with ``numerator`` /
        ``denominator`` (aliases ``num`` / ``den``). Ratio dictionaries may
        additionally provide ``label``, ``errors``, ``shift``, ``line_kwargs``,
        ``errorbar_kwargs`` and ``band_kwargs`` plus ordinary matplotlib line
        keyword arguments.
    axes : tuple(matplotlib.axes.Axes, matplotlib.axes.Axes), optional
        Existing ``(top_ax, ratio_ax)`` pair. When omitted, a new figure with a
        shared x-axis is created.
    figsize : tuple of float, default=(7.0, 6.0)
        Figure size in inches when a new figure is created.
    height_ratios : tuple of float, default=(3.0, 1.0)
        Relative heights of the upper and lower panels.
    hspace : float, default=0.05
        Vertical spacing between the two panels when a new figure is created.
    xlabel, ylabel : str, optional
        Optional axis-label overrides for the upper-panel x/y labels. By
        default these are inferred from the first curve.
    ratio_ylabel : str, default="ratio"
        Y label for the ratio panel.
    angunit : {"deg", "arcmin", "arcsec"}, default="deg"
        Default angular unit used for angular results unless overridden inside a
        curve specification.
    loglog, xscale, yscale : optional
        Upper-panel scaling controls forwarded to :func:`plot_result`.
    ratio_xscale, ratio_yscale : optional
        Lower-panel axis scales. ``ratio_xscale`` defaults to the upper-panel
        x-scale and ``ratio_yscale`` defaults to linear.
    top_errors, ratio_errors : {"bar", "band", "none"}
        Default uncertainty display modes for the upper and lower panels.
        Individual curve or ratio specifications may override them with their
        own ``errors`` entry.
    mask_nonpositive, ratio_mask_nonpositive : bool, default=True
        Masking policies for upper and lower panels. The ratio panel is always
        built from the subset of bins that survive the upper-panel masking for
        both the numerator and denominator curves.
    ratio_reference : float or None, default=1.0
        Horizontal reference line drawn on the lower panel. Pass ``None`` to
        disable the guide line.
    ratio_reference_kwargs : dict, optional
        Additional matplotlib keyword arguments for the horizontal reference
        line.
    legend, ratio_legend : bool, default=True and False
        Whether to draw legends on the upper and lower panels.
    legend_kwargs, ratio_legend_kwargs : dict, optional
        Keyword arguments forwarded to the corresponding legend calls.

    Returns
    -------
    tuple
        ``(fig, (ax_top, ax_ratio))`` where ``fig`` is the matplotlib figure
        and the two axes correspond to the upper and lower panels.

    Notes
    -----
    Ratio uncertainties are propagated with standard first-order independent
    error propagation for ``R = A / B``. Any covariance between the numerator
    and denominator measurements is ignored.
    """
    curve_specs = _normalize_curve_specs(curves)
    ratio_specs = _normalize_ratio_specs(ratios)

    top_xscale = xscale or ("log" if loglog else "linear")
    top_yscale = yscale or ("log" if loglog else "linear")
    ratio_xscale = ratio_xscale or top_xscale

    if axes is None:
        fig, (ax_top, ax_ratio) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=figsize,
            gridspec_kw={"height_ratios": list(height_ratios), "hspace": hspace},
        )
    else:
        ax_top, ax_ratio = axes
        fig = ax_top.figure

    curve_data: dict[str, dict[str, Any]] = {}
    default_xlabel = None
    default_ylabel = None
    for spec in curve_specs:
        local_angunit = spec["angunit"] if spec["angunit"] is not None else angunit
        x, y, yerr, curve_xlabel, curve_ylabel = _result_plot_spec(
            spec["result"],
            which=spec["which"],
            angunit=local_angunit,
        )
        if default_xlabel is None:
            default_xlabel = curve_xlabel
        if default_ylabel is None:
            default_ylabel = curve_ylabel

        top_errors_mode = top_errors if spec["errors"] is None else spec["errors"]
        plot_x, plot_y, plot_yerr, _, _ = _prepare_plotcf_data(
            x,
            y,
            yerr=yerr,
            loglog=loglog,
            xscale=top_xscale,
            yscale=top_yscale,
            errors=top_errors_mode,
            shift=0.0,
            mask_nonpositive=mask_nonpositive,
        )
        curve_data[spec["key"]] = {
            "x": plot_x,
            "y": plot_y,
            "yerr": plot_yerr,
            "xlabel": curve_xlabel,
            "ylabel": curve_ylabel,
        }
        plot_result(
            spec["result"],
            ax=ax_top,
            which=spec["which"],
            label=spec["label"],
            xlabel=None,
            ylabel=None,
            angunit=local_angunit,
            errors=top_errors_mode,
            shift=spec["shift"],
            loglog=loglog,
            xscale=top_xscale,
            yscale=top_yscale,
            mask_nonpositive=mask_nonpositive,
            line_kwargs=spec["line_kwargs"],
            errorbar_kwargs=spec["errorbar_kwargs"],
            band_kwargs=spec["band_kwargs"],
            **spec["plot_kwargs"],
        )

    ax_top.set_ylabel(default_ylabel if ylabel is None else ylabel)
    ax_top.tick_params(labelbottom=False)

    for spec in ratio_specs:
        num_key = spec["numerator"]
        den_key = spec["denominator"]
        try:
            num = curve_data[num_key]
        except KeyError as exc:
            raise KeyError(f"Unknown numerator curve key: {num_key!r}") from exc
        try:
            den = curve_data[den_key]
        except KeyError as exc:
            raise KeyError(f"Unknown denominator curve key: {den_key!r}") from exc

        common_x, num_idx, den_idx = _match_common_x(num["x"], den["x"])

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_y = num["y"][num_idx] / den["y"][den_idx]
            ratio_yerr = _compute_ratio_yerr(
                num["y"][num_idx],
                den["y"][den_idx],
                None if num["yerr"] is None else num["yerr"][num_idx],
                None if den["yerr"] is None else den["yerr"][den_idx],
            )

        plotcf(
            common_x,
            ratio_y,
            yerr=ratio_yerr,
            ax=ax_ratio,
            label=spec["label"],
            xlabel=None,
            ylabel=None,
            loglog=False,
            xscale=ratio_xscale,
            yscale=ratio_yscale,
            errors=ratio_errors if spec["errors"] is None else spec["errors"],
            shift=spec["shift"],
            mask_nonpositive=ratio_mask_nonpositive,
            line_kwargs=spec["line_kwargs"],
            errorbar_kwargs=spec["errorbar_kwargs"],
            band_kwargs=spec["band_kwargs"],
            **spec["plot_kwargs"],
        )

    if ratio_reference is not None:
        ref_opts = {"color": "0.5", "linestyle": "--", "linewidth": 1.0}
        ref_opts.update(ratio_reference_kwargs or {})
        ax_ratio.axhline(float(ratio_reference), **ref_opts)

    ax_ratio.set_xlabel(default_xlabel if xlabel is None else xlabel)
    ax_ratio.set_ylabel(ratio_ylabel)

    if legend:
        handles, labels = ax_top.get_legend_handles_labels()
        if labels:
            ax_top.legend(**(legend_kwargs or {}))
    if ratio_legend:
        handles, labels = ax_ratio.get_legend_handles_labels()
        if labels:
            ax_ratio.legend(**(ratio_legend_kwargs or {}))

    return fig, (ax_top, ax_ratio)

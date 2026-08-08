"""Standalone utilities for inspecting and exporting projected PDF GMM compressions."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .gmm_compress import compress_pdf_segments
from .models import DistanceSpec
from .pdf_common import _as_grid_centers, _centers_from_edges, load_array_spec, resolve_common_chi_grid, resolve_common_chi_edges


def _distance_spec_or_default(distance: DistanceSpec | None) -> DistanceSpec:
    """Return a validated distance specification for standalone PDF utilities."""
    if distance is None:
        return DistanceSpec()
    if not isinstance(distance, DistanceSpec):
        raise TypeError("distance must be a nugundam.projected.models.DistanceSpec instance or None.")
    return distance


def _resolve_pdf_matrix_input(pdfs, *, label: str = "pdfs") -> np.ndarray:
    """Resolve a PDF matrix from memory or a simple array file."""
    matrix = load_array_spec(pdfs, label=label)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{label} must resolve to a 2-D array with shape (nobj, ngrid).")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{label} cannot be empty.")
    if np.any(matrix < 0.0):
        raise ValueError(f"{label} must be non-negative.")
    row_sum = np.asarray(matrix.sum(axis=1), dtype=np.float64)
    if np.any(row_sum <= 0.0):
        raise ValueError(f"{label} rows must have strictly positive total probability.")
    return matrix


def _resolve_chi_grid_centers(*, z_grid=None, chi_grid=None, distance: DistanceSpec | None = None, grid_kind: str = "centers", label: str = "pdf") -> np.ndarray:
    """Resolve a one-dimensional chi-center grid for standalone utilities."""
    dist = _distance_spec_or_default(distance)
    cfg = SimpleNamespace(distance=dist)
    return np.asarray(
        resolve_common_chi_grid(z_grid=z_grid, chi_grid=chi_grid, config=cfg, grid_kind=grid_kind, label=label),
        dtype=np.float64,
    )


def _resolve_chi_grid_edges(*, z_grid=None, chi_grid=None, distance: DistanceSpec | None = None, grid_kind: str = "centers", label: str = "pdf") -> np.ndarray | None:
    """Resolve chi edges for standalone utilities when native edge grids exist."""
    dist = _distance_spec_or_default(distance)
    cfg = SimpleNamespace(distance=dist)
    out = resolve_common_chi_edges(z_grid=z_grid, chi_grid=chi_grid, config=cfg, grid_kind=grid_kind, label=label)
    return None if out is None else np.asarray(out, dtype=np.float64)


def _resolve_chi_plot_grid(*, z_grid=None, chi_grid=None, distance: DistanceSpec | None = None, grid_kind: str = "centers") -> tuple[np.ndarray, np.ndarray, str]:
    """Resolve chi centers and widths for empirical PDF plotting."""
    kind = str(grid_kind).strip().lower()
    dist = _distance_spec_or_default(distance)
    if chi_grid is not None:
        chi_raw = load_array_spec(chi_grid, label="chi_grid")
        chi_raw = np.asarray(chi_raw, dtype=np.float64)
        if kind == "edges":
            chi_cent = _centers_from_edges(chi_raw)
            dchi = np.diff(chi_raw)
        else:
            chi_cent = _as_grid_centers(chi_raw, grid_kind=kind)
            dchi = np.gradient(chi_cent)
        return np.asarray(chi_cent, dtype=np.float64), np.asarray(dchi, dtype=np.float64), kind

    if z_grid is None:
        raise ValueError("A common z_grid or chi_grid must be provided.")

    z_raw = load_array_spec(z_grid, label="z_grid")
    z_raw = np.asarray(z_raw, dtype=np.float64)
    try:
        from astropy.cosmology import LambdaCDM
    except Exception as exc:  # pragma: no cover
        raise ImportError("plot_gmm_for_object requires astropy to convert z_grid to comoving distance.") from exc

    cosmo = LambdaCDM(H0=dist.h0, Om0=dist.omegam, Ode0=dist.omegal)
    if kind == "edges":
        if z_raw.ndim != 1 or z_raw.size < 2:
            raise ValueError("z_grid edges must be one-dimensional with at least two entries.")
        chi_edges = np.asarray(cosmo.comoving_distance(z_raw).value, dtype=np.float64)
        chi_cent = _centers_from_edges(chi_edges)
        dchi = np.diff(chi_edges)
    else:
        z_cent = _as_grid_centers(z_raw, grid_kind=kind)
        chi_cent = np.asarray(cosmo.comoving_distance(z_cent).value, dtype=np.float64)
        dchi = np.gradient(chi_cent)
    return np.asarray(chi_cent, dtype=np.float64), np.asarray(dchi, dtype=np.float64), kind


@dataclass(slots=True)
class CompressedPdfGMM:
    """Compressed chi-space Gaussian-mixture representation of common-grid PDFs.

    Parameters
    ----------
    alpha, mu, sigma : ndarray
        GMM mixture weights, means, and standard deviations with shape
        ``(k, nobj)``.
    chi_grid : ndarray
        Shared one-dimensional chi-grid centers aligned with the original PDF
        matrix columns.
    z_grid : object, optional
        Original common redshift grid used to derive ``chi_grid`` when
        available.
    grid_kind : {'centers', 'edges'}
        Whether the original support grid was supplied as bin centers or bin
        edges.
    compressor : str
        Compression scheme used to build the GMM.
    eps : float
        Additive floor applied before row renormalization.
    sigma_floor : float
        Minimum Gaussian width enforced by the compressor.
    ids : array-like, optional
        Optional object identifiers aligned with the columns of ``alpha``,
        ``mu``, and ``sigma``.
    """

    alpha: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    chi_grid: np.ndarray
    z_grid: object | None = None
    grid_kind: str = "centers"
    compressor: str = "segments_equal_mass"
    eps: float = 0.0
    sigma_floor: float = 1.0e-6
    ids: object | None = None

    @property
    def k(self) -> int:
        """Number of Gaussian components per object."""
        return int(self.alpha.shape[0])

    @property
    def nobj(self) -> int:
        """Number of compressed objects."""
        return int(self.alpha.shape[1])

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the raw ``(alpha, mu, sigma)`` arrays."""
        return self.alpha, self.mu, self.sigma

    def to_table(
        self,
        *,
        layout: str = "wide",
        alpha_prefix: str = "alpha",
        mu_prefix: str = "mu",
        sigma_prefix: str = "sigma",
        index_base: int = 0,
    ):
        """Export the compressed GMM catalog as an Astropy table.

        Parameters
        ----------
        layout : {'wide', 'long'}, default='wide'
            Output table layout. ``'wide'`` stores one row per object and one
            set of columns per component. ``'long'`` stores one row per object
            and component.
        alpha_prefix, mu_prefix, sigma_prefix : str, default='alpha'/'mu'/'sigma'
            Column prefixes used in ``layout='wide'``.
        index_base : int, default=0
            Starting component index used in the column suffixes or the
            ``component`` column.

        Returns
        -------
        astropy.table.Table
            Tabular view of the compressed GMM catalog.
        """
        layout_norm = str(layout).strip().lower()
        ids = None if self.ids is None else np.asarray(self.ids)
        object_index = np.arange(self.nobj, dtype=np.int64)

        if layout_norm == "wide":
            data = {"object_index": object_index}
            if ids is not None:
                data["object_id"] = ids
            for j in range(self.k):
                suffix = str(index_base + j)
                data[f"{alpha_prefix}{suffix}"] = np.asarray(self.alpha[j], dtype=np.float64)
                data[f"{mu_prefix}{suffix}"] = np.asarray(self.mu[j], dtype=np.float64)
                data[f"{sigma_prefix}{suffix}"] = np.asarray(self.sigma[j], dtype=np.float64)
        elif layout_norm == "long":
            comp = np.repeat(np.arange(index_base, index_base + self.k, dtype=np.int64), self.nobj)
            obj = np.tile(object_index, self.k)
            data = {
                "object_index": obj,
                "component": comp,
                "alpha": np.asarray(self.alpha, dtype=np.float64).reshape(-1, order="C"),
                "mu": np.asarray(self.mu, dtype=np.float64).reshape(-1, order="C"),
                "sigma": np.asarray(self.sigma, dtype=np.float64).reshape(-1, order="C"),
            }
            if ids is not None:
                data["object_id"] = np.tile(ids, self.k)
        else:
            raise ValueError("layout must be either 'wide' or 'long'.")

        try:
            from astropy.table import Table
            return Table(data)
        except Exception:
            pass

        try:
            import pandas as pd
            return pd.DataFrame(data)
        except Exception:
            pass

        names = list(data.keys())
        arrays = [np.asarray(data[name]) for name in names]
        dtype = [(name, arr.dtype) for name, arr in zip(names, arrays)]
        out = np.empty(arrays[0].shape[0], dtype=dtype)
        for name, arr in zip(names, arrays):
            out[name] = arr
        return out


@dataclass(slots=True)
class PiMaxEstimate:
    """Estimated LOS integration scale implied by empirical photo-z PDFs.

    Parameters
    ----------
    zc, chi, mu_chi, sig_chi : ndarray or None
        Support-grid and per-object radial-moment quantities for the primary
        photometric sample.
    zc2, chi2, mu_chi2, sig_chi2 : ndarray or None, optional
        Same quantities for an optional secondary photometric sample used in a
        photo-photo cross-correlation estimate.
    sigma_chi_eff_1, sigma_chi_eff_2 : float or None
        Effective single-sample comoving radial uncertainty inferred from the
        chosen aggregation rule. ``sigma_chi_eff_2`` is ``None`` unless a
        second photometric sample was supplied.
    sigma_pw_eff : float
        Effective pairwise radial uncertainty for the selected case.
    pi_max_guess : float
        Suggested line-of-sight integration limit computed as
        ``multiplier * sigma_pw_eff``.
    case : {'auto', 'cross_photo_photo', 'cross_spec_photo'}
        Correlation case used to build ``sigma_pw_eff``.
    multiplier : float
        Factor used to transform ``sigma_pw_eff`` into ``pi_max_guess``.
    pdf_normalization, pdf_normalization2 : str
        Interpretation used for the supplied PDF rows: ``'density'`` or
        ``'probability'``.
    statistic : str
        Aggregation rule used to compress per-object uncertainties into the
        effective pairwise scale.
    """

    zc: np.ndarray | None
    chi: np.ndarray
    mu_chi: np.ndarray
    sig_chi: np.ndarray
    sigma_pw_eff: float
    pi_max_guess: float
    zc2: np.ndarray | None = None
    chi2: np.ndarray | None = None
    mu_chi2: np.ndarray | None = None
    sig_chi2: np.ndarray | None = None
    sigma_chi_eff_1: float | None = None
    sigma_chi_eff_2: float | None = None
    case: str = "auto"
    multiplier: float = 2.5
    pdf_normalization: str = "probability"
    pdf_normalization2: str | None = None
    statistic: str = "median_variance"

    @property
    def sigma_pw_eff_auto(self) -> float:
        """Backward-friendly alias for the effective pair scale."""
        return float(self.sigma_pw_eff)

    @property
    def sigma_pw_eff_cross(self) -> float:
        """Alias for the effective pair scale in cross-correlation cases."""
        return float(self.sigma_pw_eff)

    def as_dict(self) -> dict[str, object]:
        """Return the estimate as a plain dictionary."""
        return {
            "zc": self.zc,
            "chi": self.chi,
            "mu_chi": self.mu_chi,
            "sig_chi": self.sig_chi,
            "zc2": self.zc2,
            "chi2": self.chi2,
            "mu_chi2": self.mu_chi2,
            "sig_chi2": self.sig_chi2,
            "sigma_chi_eff_1": None if self.sigma_chi_eff_1 is None else float(self.sigma_chi_eff_1),
            "sigma_chi_eff_2": None if self.sigma_chi_eff_2 is None else float(self.sigma_chi_eff_2),
            "sigma_pw_eff": float(self.sigma_pw_eff),
            "sigma_pw_eff_auto": float(self.sigma_pw_eff),
            "sigma_pw_eff_cross": float(self.sigma_pw_eff),
            "pi_max_guess": float(self.pi_max_guess),
            "case": self.case,
            "multiplier": float(self.multiplier),
            "pdf_normalization": self.pdf_normalization,
            "pdf_normalization2": self.pdf_normalization2,
            "statistic": self.statistic,
        }


def _grid_centers_and_widths(grid, *, grid_kind: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Resolve grid centers and approximate bin widths."""
    kind = str(grid_kind).strip().lower()
    arr = load_array_spec(grid, label=label)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional grid.")
    if kind == "edges":
        if arr.size < 2:
            raise ValueError(f"{label} edges must contain at least two values.")
        centers = _centers_from_edges(arr)
        widths = np.diff(arr)
    elif kind == "centers":
        centers = arr
        widths = np.gradient(arr)
    else:
        raise ValueError("grid_kind must be either 'centers' or 'edges'.")
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError(f"{label} must be strictly increasing.")
    return np.asarray(centers, dtype=np.float64), np.asarray(widths, dtype=np.float64)


def _resolve_pdf_weight_mode(matrix: np.ndarray, widths: np.ndarray, mode: str) -> str:
    """Infer whether the supplied rows are discrete probabilities or densities."""
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"probability", "density"}:
        return mode_norm
    if mode_norm != "auto":
        raise ValueError("pdf_normalization must be 'auto', 'probability', or 'density'.")
    row_sum = np.asarray(matrix.sum(axis=1), dtype=np.float64)
    row_int = np.asarray((matrix * widths[None, :]).sum(axis=1), dtype=np.float64)
    prob_err = abs(float(np.nanmedian(row_sum)) - 1.0)
    dens_err = abs(float(np.nanmedian(row_int)) - 1.0)
    return "probability" if prob_err <= dens_err else "density"


def _aggregate_single_sample_sigma(sig_chi: np.ndarray, *, statistic: str) -> float:
    """Aggregate per-object radial widths into an effective single-sample scale."""
    stat = str(statistic).strip().lower()
    sig2 = np.asarray(sig_chi, dtype=np.float64) ** 2
    if sig2.ndim != 1 or sig2.size == 0:
        raise ValueError("sig_chi must be a non-empty one-dimensional array.")
    if stat in {"median_variance", "median"}:
        base = float(np.median(sig2))
    elif stat in {"mean_variance", "mean"}:
        base = float(np.mean(sig2))
    else:
        raise ValueError("statistic must be 'median_variance' or 'mean_variance'.")
    return float(np.sqrt(max(base, 0.0)))


def _aggregate_pairwise_sigma(sig_chi: np.ndarray, *, statistic: str) -> float:
    """Aggregate per-object radial widths into an effective auto pairwise scale."""
    sigma_single = _aggregate_single_sample_sigma(sig_chi, statistic=statistic)
    return float(np.sqrt(2.0) * sigma_single)


def estimate_pi_max_from_pdfs(
    pdfs,
    *,
    pdfs2=None,
    z_grid=None,
    chi_grid=None,
    z_grid2=None,
    chi_grid2=None,
    grid_kind: str = "centers",
    grid_kind2: str | None = None,
    distance: DistanceSpec | None = None,
    pdf_normalization: str = "auto",
    pdf_normalization2: str | None = None,
    statistic: str = "median_variance",
    multiplier: float = 2.5,
    output: str = "result",
    sample2_kind: str = "photo",
):
    """Estimate a sensible ``pi_max`` scale from common-grid empirical PDFs.

    This helper summarizes the line-of-sight smearing implied by per-object
    photo-z PDFs and returns a default projected-correlation integration limit.
    The primary sample is always photometric and represented by ``pdfs``.
    Optionally, a secondary photometric sample (``pdfs2``) or a spectroscopic
    secondary sample (``sample2_kind='spec'``) can be selected to estimate
    photo-photo and spec-photo cross-correlation scales.

    Parameters
    ----------
    pdfs : array-like or path-like
        Empirical PDF matrix with shape ``(nobj, ngrid)`` for the primary
        photometric sample. The input may be an in-memory array or a path to a
        ``.npy``, ``.npz``, or whitespace-delimited text file.
    pdfs2 : array-like or path-like, optional
        Empirical PDF matrix for an optional secondary photometric sample. When
        supplied, the helper returns a photo-photo cross-correlation estimate.
        Leave as ``None`` to compute an auto-correlation estimate, or combine
        with ``sample2_kind='spec'`` to represent a spectroscopic secondary
        sample with negligible radial uncertainty.
    z_grid, chi_grid : array-like or path-like, optional
        Shared support grid for the primary sample expressed either in redshift
        or in comoving distance. When both are supplied, ``chi_grid`` takes
        precedence.
    z_grid2, chi_grid2 : array-like or path-like, optional
        Optional support grid for ``pdfs2``. When omitted, the primary-sample
        grid is reused. ``chi_grid2`` takes precedence over ``z_grid2``.
    grid_kind : {'centers', 'edges'}, default='centers'
        Whether the supplied primary-sample support grid contains bin centers
        or bin edges.
    grid_kind2 : {'centers', 'edges'}, optional
        Grid-kind override for the secondary photometric sample. When omitted,
        ``grid_kind`` is reused.
    distance : DistanceSpec, optional
        Cosmology used when converting ``z_grid`` inputs to comoving distance.
        When omitted, :class:`~nugundam.projected.models.DistanceSpec` defaults
        are used.
    pdf_normalization : {'auto', 'probability', 'density'}, default='auto'
        Interpretation of each primary-sample PDF row. ``'probability'`` means
        each row already stores discrete bin probabilities. ``'density'`` means
        the rows store a density on the supplied support grid and are
        multiplied by the corresponding bin widths before renormalization.
        ``'auto'`` picks the interpretation whose median row total is closest
        to unity.
    pdf_normalization2 : {'auto', 'probability', 'density'}, optional
        Normalization override for ``pdfs2``. When omitted, the primary value
        of ``pdf_normalization`` is reused.
    statistic : {'median_variance', 'mean_variance'}, default='median_variance'
        Rule used to compress the per-object radial widths into the effective
        single-sample and pairwise scales.
    multiplier : float, default=2.5
        Factor used to transform the effective pairwise radial scale into the
        suggested ``pi_max``.
    output : {'result', 'dict'}, default='result'
        Return a :class:`PiMaxEstimate` or a plain dictionary.
    sample2_kind : {'photo', 'spec'}, default='photo'
        Type of the optional secondary sample. Use ``'photo'`` for a second PDF
        matrix in ``pdfs2`` or ``'spec'`` to represent a spectroscopic sample
        with negligible radial uncertainty.

    Returns
    -------
    PiMaxEstimate or dict
        Container with the shared grids, per-object radial moments, the
        effective pairwise width, and the suggested ``pi_max``.
    """
    matrix1 = _resolve_pdf_matrix_input(pdfs, label="pdfs")
    dist = _distance_spec_or_default(distance)

    zc1 = None
    if chi_grid is not None:
        chi_cent1, native_widths1 = _grid_centers_and_widths(chi_grid, grid_kind=grid_kind, label="chi_grid")
    else:
        if z_grid is None:
            raise ValueError("A common z_grid or chi_grid must be provided.")
        zc1, native_widths1 = _grid_centers_and_widths(z_grid, grid_kind=grid_kind, label="z_grid")
        chi_cent1 = _resolve_chi_grid_centers(z_grid=z_grid, chi_grid=None, distance=dist, grid_kind=grid_kind, label="pdf")

    if matrix1.shape[1] != chi_cent1.size:
        raise ValueError(
            f"pdfs column count {matrix1.shape[1]} does not match the resolved support-grid length {chi_cent1.size}."
        )

    norm_mode1 = _resolve_pdf_weight_mode(matrix1, native_widths1, pdf_normalization)
    if norm_mode1 == "density":
        weights1 = np.asarray(matrix1 * native_widths1[None, :], dtype=np.float64)
    else:
        weights1 = np.asarray(matrix1, dtype=np.float64)

    row_sum1 = np.asarray(weights1.sum(axis=1), dtype=np.float64)
    if np.any(row_sum1 <= 0.0):
        raise ValueError("pdfs rows must have strictly positive total probability after normalization handling.")
    weights1 = np.asarray(weights1 / row_sum1[:, None], dtype=np.float64)

    mu_chi1 = np.sum(weights1 * chi_cent1[None, :], axis=1)
    var_chi1 = np.sum(weights1 * (chi_cent1[None, :] - mu_chi1[:, None]) ** 2, axis=1)
    sig_chi1 = np.sqrt(np.maximum(var_chi1, 0.0))
    sigma_chi_eff_1 = _aggregate_single_sample_sigma(sig_chi1, statistic=statistic)

    case = str(sample2_kind).strip().lower()
    if case not in {"photo", "spec"}:
        raise ValueError("sample2_kind must be either 'photo' or 'spec'.")

    zc2 = None
    chi_cent2 = None
    mu_chi2 = None
    sig_chi2 = None
    norm_mode2 = None
    sigma_chi_eff_2 = None

    if pdfs2 is None:
        if case == "spec":
            case_name = "cross_spec_photo"
            sigma_pw_eff = float(sigma_chi_eff_1)
        else:
            case_name = "auto"
            sigma_pw_eff = float(np.sqrt(2.0) * sigma_chi_eff_1)
    else:
        if case == "spec":
            raise ValueError("pdfs2 must be omitted when sample2_kind='spec'.")

        matrix2 = _resolve_pdf_matrix_input(pdfs2, label="pdfs2")
        grid_kind_2 = grid_kind if grid_kind2 is None else grid_kind2
        pdf_norm_2 = pdf_normalization if pdf_normalization2 is None else pdf_normalization2

        if chi_grid2 is not None:
            chi_cent2, native_widths2 = _grid_centers_and_widths(chi_grid2, grid_kind=grid_kind_2, label="chi_grid2")
        elif z_grid2 is not None:
            zc2, native_widths2 = _grid_centers_and_widths(z_grid2, grid_kind=grid_kind_2, label="z_grid2")
            chi_cent2 = _resolve_chi_grid_centers(z_grid=z_grid2, chi_grid=None, distance=dist, grid_kind=grid_kind_2, label="pdf2")
        else:
            zc2 = None if zc1 is None else np.asarray(zc1, dtype=np.float64)
            chi_cent2 = np.asarray(chi_cent1, dtype=np.float64)
            native_widths2 = np.asarray(native_widths1, dtype=np.float64)

        if matrix2.shape[1] != chi_cent2.size:
            raise ValueError(
                f"pdfs2 column count {matrix2.shape[1]} does not match the resolved support-grid length {chi_cent2.size}."
            )

        norm_mode2 = _resolve_pdf_weight_mode(matrix2, native_widths2, pdf_norm_2)
        if norm_mode2 == "density":
            weights2 = np.asarray(matrix2 * native_widths2[None, :], dtype=np.float64)
        else:
            weights2 = np.asarray(matrix2, dtype=np.float64)

        row_sum2 = np.asarray(weights2.sum(axis=1), dtype=np.float64)
        if np.any(row_sum2 <= 0.0):
            raise ValueError("pdfs2 rows must have strictly positive total probability after normalization handling.")
        weights2 = np.asarray(weights2 / row_sum2[:, None], dtype=np.float64)

        mu_chi2 = np.sum(weights2 * chi_cent2[None, :], axis=1)
        var_chi2 = np.sum(weights2 * (chi_cent2[None, :] - mu_chi2[:, None]) ** 2, axis=1)
        sig_chi2 = np.sqrt(np.maximum(var_chi2, 0.0))
        sigma_chi_eff_2 = _aggregate_single_sample_sigma(sig_chi2, statistic=statistic)

        case_name = "cross_photo_photo"
        sigma_pw_eff = float(np.sqrt(sigma_chi_eff_1 ** 2 + sigma_chi_eff_2 ** 2))

    mult = float(multiplier)
    if not np.isfinite(mult) or mult <= 0.0:
        raise ValueError("multiplier must be a strictly positive finite number.")
    result = PiMaxEstimate(
        zc=None if zc1 is None else np.asarray(zc1, dtype=np.float64),
        chi=np.asarray(chi_cent1, dtype=np.float64),
        mu_chi=np.asarray(mu_chi1, dtype=np.float64),
        sig_chi=np.asarray(sig_chi1, dtype=np.float64),
        sigma_pw_eff=float(sigma_pw_eff),
        pi_max_guess=float(mult * sigma_pw_eff),
        zc2=None if zc2 is None else np.asarray(zc2, dtype=np.float64),
        chi2=None if chi_cent2 is None else np.asarray(chi_cent2, dtype=np.float64),
        mu_chi2=None if mu_chi2 is None else np.asarray(mu_chi2, dtype=np.float64),
        sig_chi2=None if sig_chi2 is None else np.asarray(sig_chi2, dtype=np.float64),
        sigma_chi_eff_1=float(sigma_chi_eff_1),
        sigma_chi_eff_2=None if sigma_chi_eff_2 is None else float(sigma_chi_eff_2),
        case=case_name,
        multiplier=mult,
        pdf_normalization=norm_mode1,
        pdf_normalization2=norm_mode2,
        statistic=str(statistic).strip().lower(),
    )
    out = str(output).strip().lower()
    if out == "result":
        return result
    if out == "dict":
        return result.as_dict()
    raise ValueError("output must be either 'result' or 'dict'.")


def compress_pdfs_to_gmm(
    pdfs,
    *,
    z_grid=None,
    chi_grid=None,
    grid_kind: str = "centers",
    distance: DistanceSpec | None = None,
    k: int = 3,
    compressor: str = "segments_equal_mass",
    eps: float = 0.0,
    sigma_floor: float = 1.0e-6,
    edge_moments: bool | None = None,
    ids=None,
    output: str = "result",
    table_layout: str = "wide",
    alpha_prefix: str = "alpha",
    mu_prefix: str = "mu",
    sigma_prefix: str = "sigma",
    index_base: int = 0,
):
    """Compress common-grid empirical PDFs into chi-space Gaussian mixtures.

    This helper is a standalone public wrapper around
    :func:`nugundam.projected.gmm_compress.compress_pdf_segments`. It accepts
    the original empirical PDF matrix on a common redshift or chi grid,
    resolves the chi-space grid internally using the supplied cosmology, and
    returns the Gaussian-mixture parameters for all objects.

    Parameters
    ----------
    pdfs : array-like or path-like
        Empirical PDF matrix with shape ``(nobj, ngrid)``. The input may be an
        in-memory array or a path to a ``.npy``, ``.npz``, or whitespace-
        delimited text file.
    z_grid, chi_grid : array-like or path-like, optional
        Shared support grid expressed either in redshift or in chi. When both
        are supplied, ``chi_grid`` takes precedence. ``z_grid`` is converted to
        chi with the cosmology defined by ``distance``.
    grid_kind : {'centers', 'edges'}, default='centers'
        Whether the supplied support grid contains bin centers or bin edges.
    distance : DistanceSpec, optional
        Cosmology used when converting ``z_grid`` to comoving distance. When
        omitted, :class:`~nugundam.projected.models.DistanceSpec` defaults are
        used.
    k : int, default=3
        Number of Gaussian components per object.
    compressor : str, default='segments_equal_mass'
        Compression scheme passed to :func:`compress_pdf_segments`.
    eps : float, default=0.0
        Additive probability floor applied before row renormalization.
    sigma_floor : float, default=1e-6
        Minimum Gaussian width enforced by the compressor.
    edge_moments : bool or None, optional
        If True and ``grid_kind='edges'``, treat each input PDF value as
        probability uniformly distributed inside its chi bin when computing
        GMM moments. If None, this is enabled automatically for edge grids.
    ids : array-like, optional
        Optional object identifiers aligned with the rows of ``pdfs``.
    output : {'result', 'arrays', 'table'}, default='result'
        Return a :class:`CompressedPdfGMM`, a raw ``(alpha, mu, sigma)`` tuple,
        or an Astropy table.
    table_layout : {'wide', 'long'}, default='wide'
        Table layout used when ``output='table'``.
    alpha_prefix, mu_prefix, sigma_prefix : str, default='alpha'/'mu'/'sigma'
        Column prefixes used by ``output='table', table_layout='wide'``.
    index_base : int, default=0
        Starting component index for table output.

    Returns
    -------
    CompressedPdfGMM or tuple of ndarray or astropy.table.Table
        Compressed GMM representation of the full PDF catalog.
    """
    matrix = _resolve_pdf_matrix_input(pdfs, label="pdfs")
    kind = str(grid_kind).strip().lower()
    chi_cent = _resolve_chi_grid_centers(z_grid=z_grid, chi_grid=chi_grid, distance=distance, grid_kind=kind, label="pdf")
    chi_edges = _resolve_chi_grid_edges(z_grid=z_grid, chi_grid=chi_grid, distance=distance, grid_kind=kind, label="pdf")
    if matrix.shape[1] != chi_cent.size:
        raise ValueError(
            f"pdfs column count {matrix.shape[1]} does not match the resolved chi-grid length {chi_cent.size}."
        )
    if ids is not None:
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.size != matrix.shape[0]:
            raise ValueError(f"ids must be one-dimensional with length {matrix.shape[0]}.")

    use_edge_moments = (kind == "edges") if edge_moments is None else (bool(edge_moments) and kind == "edges")
    alpha, mu, sigma = compress_pdf_segments(
        matrix,
        chi_cent,
        k=int(k),
        compressor=compressor,
        eps=float(eps),
        sigma_floor=float(sigma_floor),
        chi_edges=chi_edges,
        edge_moments=use_edge_moments,
    )
    result = CompressedPdfGMM(
        alpha=np.asarray(alpha, dtype=np.float64, order="F"),
        mu=np.asarray(mu, dtype=np.float64, order="F"),
        sigma=np.asarray(sigma, dtype=np.float64, order="F"),
        chi_grid=np.asarray(chi_cent, dtype=np.float64),
        z_grid=z_grid,
        grid_kind=kind,
        compressor=str(compressor),
        eps=float(eps),
        sigma_floor=float(sigma_floor),
        ids=ids,
    )
    output_norm = str(output).strip().lower()
    if output_norm == "result":
        return result
    if output_norm == "arrays":
        return result.as_arrays()
    if output_norm == "table":
        return result.to_table(
            layout=table_layout,
            alpha_prefix=alpha_prefix,
            mu_prefix=mu_prefix,
            sigma_prefix=sigma_prefix,
            index_base=index_base,
        )
    raise ValueError("output must be one of 'result', 'arrays', or 'table'.")





def _resolve_redshift_axis_grid(*, z_grid=None, compressed=None, grid_kind: str = "centers") -> np.ndarray | None:
    """Resolve the redshift grid used to build the optional top-axis mapping."""
    z_source = z_grid
    if z_source is None and isinstance(compressed, CompressedPdfGMM):
        z_source = compressed.z_grid
    if z_source is None:
        return None
    z_raw = load_array_spec(z_source, label="z_grid")
    z_raw = np.asarray(z_raw, dtype=np.float64)
    return np.asarray(_as_grid_centers(z_raw, grid_kind=str(grid_kind).strip().lower()), dtype=np.float64)


def _build_chi_redshift_mapping(*, z_grid=None, compressed=None, distance: DistanceSpec | None = None, grid_kind: str = "centers"):
    """Build forward and inverse interpolation functions between chi and redshift."""
    z_cent = _resolve_redshift_axis_grid(z_grid=z_grid, compressed=compressed, grid_kind=grid_kind)
    if z_cent is None:
        return None, None, None, None

    dist = _distance_spec_or_default(distance)
    try:
        from astropy.cosmology import LambdaCDM
    except Exception as exc:  # pragma: no cover
        raise ImportError("plot_gmm_for_object requires astropy to convert z_grid to comoving distance.") from exc

    cosmo = LambdaCDM(H0=dist.h0, Om0=dist.omegam, Ode0=dist.omegal)
    z_dense = np.linspace(float(np.min(z_cent)), float(np.max(z_cent)), 4096)
    chi_dense = np.asarray(cosmo.comoving_distance(z_dense).value, dtype=np.float64)

    def chi_to_z(chi_vals):
        chi_vals = np.asarray(chi_vals, dtype=np.float64)
        chi_clip = np.clip(chi_vals, chi_dense[0], chi_dense[-1])
        return np.interp(chi_clip, chi_dense, z_dense)

    def z_to_chi(z_vals):
        z_vals = np.asarray(z_vals, dtype=np.float64)
        z_clip = np.clip(z_vals, z_dense[0], z_dense[-1])
        return np.interp(z_clip, z_dense, chi_dense)

    return chi_to_z, z_to_chi, z_dense, chi_dense


def _normal_pdf(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Evaluate a normal density."""
    sig = float(sig)
    return np.exp(-0.5 * ((x - float(mu)) / sig) ** 2) / (np.sqrt(2.0 * np.pi) * sig)



def _coerce_compressed(compressed) -> CompressedPdfGMM:
    """Coerce supported compressed-GMM inputs to ``CompressedPdfGMM``."""
    if isinstance(compressed, CompressedPdfGMM):
        return compressed
    if isinstance(compressed, (tuple, list)) and len(compressed) == 3:
        alpha, mu, sigma = compressed
        alpha = np.asarray(alpha, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        if alpha.ndim != 2 or mu.shape != alpha.shape or sigma.shape != alpha.shape:
            raise ValueError("compressed tuples must be (alpha, mu, sigma) arrays with identical 2-D shapes.")
        return CompressedPdfGMM(
            alpha=alpha,
            mu=mu,
            sigma=sigma,
            chi_grid=np.array([], dtype=np.float64),
        )
    raise TypeError("compressed must be a CompressedPdfGMM instance or an (alpha, mu, sigma) tuple.")



def plot_gmm_for_object(
    obj_index,
    pdfs=None,
    *,
    z_grid=None,
    chi_grid=None,
    grid_kind: str = "centers",
    distance: DistanceSpec | None = None,
    compressed=None,
    k: int = 3,
    compressor: str = "segments_equal_mass",
    eps: float = 0.0,
    sigma_floor: float = 1.0e-6,
    x=None,
    nplot: int = 1200,
    xlim=None,
    ax=None,
    show_empirical: bool = True,
    show_components: bool = True,
    show_total: bool = True,
    legend: bool = True,
    show_redshift_axis: bool = True,
    return_data: bool = False,
):
    """Plot the empirical PDF and GMM approximation for one object.

    Parameters
    ----------
    obj_index : int
        Row index of the object to inspect.
    pdfs : array-like or path-like, optional
        Empirical PDF matrix with shape ``(nobj, ngrid)``. Required when the
        empirical curve is requested or when ``compressed`` is not supplied.
    z_grid, chi_grid : array-like or path-like, optional
        Common support grid for ``pdfs``. ``chi_grid`` takes precedence when
        both are supplied.
    grid_kind : {'centers', 'edges'}, default='centers'
        Whether the supplied support grid contains bin centers or bin edges.
    distance : DistanceSpec, optional
        Cosmology used to convert ``z_grid`` to chi.
    compressed : CompressedPdfGMM or tuple, optional
        Precomputed compressed GMM catalog. When omitted, the helper computes it
        internally from ``pdfs``.
    k, compressor, eps, sigma_floor
        Compression parameters used only when ``compressed`` is omitted.
    x : ndarray, optional
        Explicit chi grid used to draw the smooth GMM curves.
    nplot : int, default=1200
        Number of samples used when constructing the smooth GMM curves when
        ``x`` is not supplied.
    xlim : tuple, optional
        Optional x-axis limits passed to Matplotlib.
    ax : matplotlib.axes.Axes, optional
        Axes used for the plot. When omitted, a new figure and axes are
        created.
    show_empirical, show_components, show_total : bool, default=True
        Toggle the empirical curve, the individual Gaussian components, and the
        total GMM curve.
    legend : bool, default=True
        Draw the legend when True.
    show_redshift_axis : bool, default=True
        Add a secondary top x-axis labeled in redshift when a ``z_grid`` is
        available either explicitly or through ``compressed``. When only a
        ``chi_grid`` is available, no top axis is added.
    return_data : bool, default=False
        When True, return ``(ax, payload)`` where ``payload`` contains the
        plotted arrays and component parameters.

    Returns
    -------
    matplotlib.axes.Axes or tuple
        The plot axes, or ``(axes, payload)`` when ``return_data=True``.
    """
    obj_index = int(obj_index)

    matrix = None
    chi_cent = None
    dchi = None
    kind = str(grid_kind).strip().lower()
    if pdfs is not None:
        matrix = _resolve_pdf_matrix_input(pdfs, label="pdfs")
        chi_cent, dchi, kind = _resolve_chi_plot_grid(
            z_grid=z_grid,
            chi_grid=chi_grid,
            distance=distance,
            grid_kind=grid_kind,
        )
        if matrix.shape[1] != chi_cent.size:
            raise ValueError(
                f"pdfs column count {matrix.shape[1]} does not match the resolved chi-grid length {chi_cent.size}."
            )
        if obj_index < 0 or obj_index >= matrix.shape[0]:
            raise IndexError(f"obj_index={obj_index} is out of bounds for a PDF matrix with {matrix.shape[0]} rows.")
    elif show_empirical:
        raise ValueError("pdfs must be provided when show_empirical=True.")

    if compressed is None:
        if matrix is None:
            raise ValueError("pdfs plus z_grid or chi_grid are required when compressed is not supplied.")
        compressed_obj = compress_pdfs_to_gmm(
            matrix,
            z_grid=z_grid,
            chi_grid=chi_grid,
            grid_kind=kind,
            distance=distance,
            k=int(k),
            compressor=compressor,
            eps=float(eps),
            sigma_floor=float(sigma_floor),
            output="result",
        )
    else:
        compressed_obj = _coerce_compressed(compressed)
        if obj_index < 0 or obj_index >= compressed_obj.nobj:
            raise IndexError(
                f"obj_index={obj_index} is out of bounds for compressed arrays with {compressed_obj.nobj} objects."
            )
        if matrix is not None and compressed_obj.nobj != matrix.shape[0]:
            raise ValueError("compressed object count does not match the number of PDF rows.")

    a = np.asarray(compressed_obj.alpha[:, obj_index], dtype=np.float64)
    m = np.asarray(compressed_obj.mu[:, obj_index], dtype=np.float64)
    s = np.asarray(compressed_obj.sigma[:, obj_index], dtype=np.float64)

    if x is None:
        if chi_cent is None:
            chi_base = np.asarray(compressed_obj.chi_grid, dtype=np.float64)
            if chi_base.size == 0:
                xmin = np.min(m - 4.0 * s)
                xmax = np.max(m + 4.0 * s)
            else:
                xmin = float(np.min(chi_base))
                xmax = float(np.max(chi_base))
            x = np.linspace(xmin, xmax, int(nplot))
        else:
            x = np.linspace(float(np.min(chi_cent)), float(np.max(chi_cent)), int(nplot))
    else:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1 or x.size == 0:
            raise ValueError("x must be a non-empty one-dimensional array.")

    comp_curves = np.asarray([a[j] * _normal_pdf(x, m[j], s[j]) for j in range(a.size)], dtype=np.float64)
    gmm_total = np.sum(comp_curves, axis=0)

    pchi_emp = None
    if matrix is not None:
        pchi_emp = np.asarray(matrix[obj_index], dtype=np.float64) / np.asarray(dchi, dtype=np.float64)
        area_emp = np.trapezoid(pchi_emp, chi_cent)
        if area_emp > 0.0:
            pchi_emp = pchi_emp / area_emp

    if ax is None:
        import matplotlib.pyplot as plt
        _fig, ax = plt.subplots()

    if show_empirical and pchi_emp is not None:
        if kind == "edges":
            ax.step(chi_cent, pchi_emp, where="mid", color="gray", lw=0.8, label="Empirical PDF")
        else:
            ax.plot(chi_cent, pchi_emp, marker="o", ms=3.0, color="gray", lw=0.8, label="Empirical PDF")

    if show_components:
        for j in range(a.size):
            ax.plot(x, comp_curves[j], lw=0.9, ls="--", label=f"Component {j + 1} (α={a[j]:.3f})")

    if show_total:
        ax.plot(x, gmm_total, lw=1.2, label="GMM total")

    ax.set_xlabel(r"$\chi$")
    ax.set_ylabel(r"$p(\chi)$")
    ax.set_title(f"Object {obj_index}: empirical PDF vs GMM")
    if xlim is not None:
        ax.set_xlim(xlim)

    secax = None
    z_dense = None
    chi_dense = None
    if show_redshift_axis:
        chi_to_z, z_to_chi, z_dense, chi_dense = _build_chi_redshift_mapping(
            z_grid=z_grid,
            compressed=compressed_obj,
            distance=distance,
            grid_kind=kind,
        )
        if chi_to_z is not None and z_to_chi is not None:
            secax = ax.secondary_xaxis("top", functions=(chi_to_z, z_to_chi))
            secax.set_xlabel(r"$z$")

    if legend:
        ax.legend()

    payload = {
        "chi_cent": chi_cent,
        "pchi_emp": pchi_emp,
        "alpha": a,
        "mu": m,
        "sigma": s,
        "x": x,
        "component_curves": comp_curves,
        "gmm_total": gmm_total,
        "grid_kind": kind,
        "compressed": compressed_obj,
        "z_dense": z_dense,
        "chi_dense": chi_dense,
        "ax": ax,
        "fig": ax.figure,
        "secax": secax,
    }
    if return_data:
        return ax, payload
    return ax


__all__ = ["CompressedPdfGMM", "PiMaxEstimate", "compress_pdfs_to_gmm", "estimate_pi_max_from_pdfs", "plot_gmm_for_object"]

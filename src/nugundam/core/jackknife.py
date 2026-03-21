"""Shared jackknife helpers for region generation and covariance assembly."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np


def validate_resampling_choice(bootstrap, jackknife) -> None:
    """Validate the resampling configuration.

    Parameters
    ----------
    bootstrap, jackknife : object
        Resampling configuration objects exposing an ``enabled`` attribute.

    Raises
    ------
    ValueError
        If both bootstrap and jackknife are enabled simultaneously.
    """
    if bootstrap is not None and jackknife is not None:
        if bool(getattr(bootstrap, "enabled", False)) and bool(getattr(jackknife, "enabled", False)):
            raise ValueError("Bootstrap and jackknife cannot be enabled simultaneously.")


def choose_default_nregions(p: int) -> int:
    """Choose a practical default number of jackknife regions.

    Parameters
    ----------
    p : int
        Number of bins in the statistic whose covariance will be estimated.

    Returns
    -------
    int
        Default number of jackknife regions.
    """
    p = max(int(p), 1)
    return int(min(100, max(36, 2 * p)))


def normalize_region_labels(labels) -> np.ndarray:
    """Map arbitrary labels to contiguous integer region ids.

    Parameters
    ----------
    labels : array-like
        One-dimensional region labels supplied by the user.

    Returns
    -------
    ndarray
        Integer region identifiers running from ``0`` to ``nregions - 1``.
    """
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("Region labels must be one-dimensional.")
    _, inv = np.unique(arr, return_inverse=True)
    return np.asarray(inv, dtype=np.int32)


def _stack_xyz(catalogs: Iterable[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    pts: list[np.ndarray] = []
    for ra, dec in catalogs:
        ra = np.asarray(ra, dtype=np.float64)
        dec = np.asarray(dec, dtype=np.float64)
        if ra.size == 0:
            continue
        rar = np.deg2rad(ra)
        decr = np.deg2rad(dec)
        c = np.cos(decr)
        xyz = np.column_stack((c * np.cos(rar), c * np.sin(rar), np.sin(decr)))
        pts.append(xyz)
    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    out = np.concatenate(pts, axis=0)
    norms = np.linalg.norm(out, axis=1)
    norms[norms == 0.0] = 1.0
    return out / norms[:, None]


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1)
    norms[norms == 0.0] = 1.0
    return arr / norms[:, None]


def _init_centers(points: np.ndarray, nregions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    npts = points.shape[0]
    nregions = min(max(1, int(nregions)), npts)
    if nregions == 1:
        return _normalize_rows(points[[0]].copy())
    idx = np.empty(nregions, dtype=np.int64)
    idx[0] = int(rng.integers(0, npts))
    sim = points @ points[idx[0]]
    for i in range(1, nregions):
        idx[i] = int(np.argmin(sim))
        sim = np.maximum(sim, points @ points[idx[i]])
    return _normalize_rows(points[idx].copy())


def build_shared_sky_regions(
    geometry_catalogs: Iterable[tuple[np.ndarray, np.ndarray]],
    assignment_catalogs: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    nregions: int,
    seed: int = 12345,
    max_iter: int = 20,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Generate shared compact sky regions for one or more catalogs.

    Parameters
    ----------
    geometry_catalogs : iterable of tuple of ndarray
        Catalogs used to determine the common patch centers. Each element is a
        ``(ra, dec)`` pair in degrees.
    assignment_catalogs : iterable of tuple of ndarray
        Catalogs that receive region ids from the fitted centers. Each element
        is a ``(ra, dec)`` pair in degrees.
    nregions : int
        Requested number of regions.
    seed : int, default=12345
        Random seed used by the spherical k-means initializer.
    max_iter : int, default=20
        Maximum number of Lloyd iterations.

    Returns
    -------
    assignments : list of ndarray
        Region identifiers for each catalog in ``assignment_catalogs``.
    centers : ndarray
        Normalized Cartesian patch centers with shape ``(nregions, 3)``.

    Raises
    ------
    ValueError
        If the geometry catalog set is empty.
    """
    points = _stack_xyz(geometry_catalogs)
    if points.shape[0] == 0:
        raise ValueError("Cannot generate jackknife regions from an empty geometry catalog set.")
    nregions = min(max(1, int(nregions)), points.shape[0])
    centers = _init_centers(points, nregions, seed)
    prev = None
    for _ in range(max_iter):
        labels = np.argmax(points @ centers.T, axis=1)
        if prev is not None and np.array_equal(labels, prev):
            break
        prev = labels
        new_centers = centers.copy()
        for k in range(nregions):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = points[mask].mean(axis=0)
            else:
                far_idx = int(np.argmin(np.max(points @ centers.T, axis=1)))
                new_centers[k] = points[far_idx]
        centers = _normalize_rows(new_centers)
    assignments = [assign_regions_from_centers(ra, dec, centers) for ra, dec in assignment_catalogs]
    return assignments, centers


def assign_regions_from_centers(ra, dec, centers: np.ndarray) -> np.ndarray:
    """Assign sky coordinates to the nearest spherical region center.

    Parameters
    ----------
    ra, dec : array-like
        Right ascension and declination coordinates in degrees.
    centers : ndarray
        Array of normalized Cartesian patch centers with shape ``(nregions, 3)``.

    Returns
    -------
    ndarray
        Integer region identifiers for the supplied coordinates.
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    if ra.size == 0:
        return np.empty(0, dtype=np.int32)
    rar = np.deg2rad(ra)
    decr = np.deg2rad(dec)
    c = np.cos(decr)
    xyz = np.column_stack((c * np.cos(rar), c * np.sin(rar), np.sin(decr)))
    xyz = _normalize_rows(xyz)
    return np.argmax(xyz @ np.asarray(centers, dtype=np.float64).T, axis=1).astype(np.int32)


def jackknife_cov(realizations: np.ndarray) -> np.ndarray:
    """Return the delete-one jackknife covariance matrix.

    Parameters
    ----------
    realizations : ndarray, shape (nregions, nbins)
        Leave-one-region-out realizations of the measured statistic.

    Returns
    -------
    ndarray
        Jackknife covariance matrix with shape ``(nbins, nbins)``.
    """
    arr = np.asarray(realizations, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Jackknife realizations must have shape (n_regions, n_bins).")
    n = arr.shape[0]
    if n <= 1:
        return np.zeros((arr.shape[1], arr.shape[1]), dtype=np.float64)
    mean = np.mean(arr, axis=0)
    delta = arr - mean[None, :]
    return ((n - 1.0) / n) * (delta.T @ delta)

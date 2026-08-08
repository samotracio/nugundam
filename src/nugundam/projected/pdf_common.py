"""Shared helpers for projected empirical-PDF inputs on a common grid."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..core.catalogs import catalog_column_names, catalog_get_column


def load_array_spec(value, *, label: str = "grid") -> np.ndarray:
    """Resolve an array-like spec from memory or disk.

    Parameters
    ----------
    value : object
        In-memory array-like object or a path to ``.npy``, ``.npz``, or a
        whitespace-delimited text file.
    label : str, default='grid'
        Human-readable label used in error messages.

    Returns
    -------
    ndarray
        One-dimensional or multi-dimensional float64 array loaded from the
        supplied spec.
    """
    if value is None:
        raise ValueError(f"A common {label} must be provided.")
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


def _centers_from_edges(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError("Common PDF grids must be one-dimensional.")
    if grid.size < 2:
        raise ValueError("Grid edges must contain at least two values.")
    return np.asarray(0.5 * (grid[:-1] + grid[1:]), dtype=np.float64)


def _as_grid_centers(grid: np.ndarray, *, grid_kind: str) -> np.ndarray:
    kind = str(grid_kind).strip().lower()
    arr = np.asarray(grid, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Common PDF grids must be one-dimensional.")
    if arr.size == 0:
        raise ValueError("Common PDF grids cannot be empty.")
    if kind == "centers":
        return arr
    if kind == "edges":
        return _centers_from_edges(arr)
    raise ValueError("grid_kind must be either 'centers' or 'edges'.")


def zgrid_to_chi(z_grid: np.ndarray, config, *, grid_kind: str = "centers", label: str = "pdf") -> np.ndarray:
    """Convert a common redshift grid to comoving-distance grid centers.

    The output uses the same cosmology and distance units as the standard
    projected preparation layer, so it is consistent with ``distance.calcdist``.
    """
    try:
        from astropy.cosmology import LambdaCDM
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            f"Projected {label} handling requires astropy to convert the shared z-grid to comoving distance."
        ) from exc
    z_centers = _as_grid_centers(np.asarray(z_grid, dtype=np.float64), grid_kind=grid_kind)
    cosmo = LambdaCDM(H0=config.distance.h0, Om0=config.distance.omegam, Ode0=config.distance.omegal)
    return np.asarray(cosmo.comoving_distance(z_centers).value, dtype=np.float64)



def zgrid_to_chi_edges(z_grid: np.ndarray, config, *, label: str = "pdf") -> np.ndarray:
    """Convert a redshift edge grid to comoving-distance edges."""
    try:
        from astropy.cosmology import LambdaCDM
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            f"Projected {label} handling requires astropy to convert the shared z-grid to comoving distance."
        ) from exc
    z_edges = np.asarray(z_grid, dtype=np.float64)
    if z_edges.ndim != 1 or z_edges.size < 2:
        raise ValueError("Redshift edge grids must be one-dimensional with at least two entries.")
    cosmo = LambdaCDM(H0=config.distance.h0, Om0=config.distance.omegam, Ode0=config.distance.omegal)
    return np.asarray(cosmo.comoving_distance(z_edges).value, dtype=np.float64)


def resolve_common_chi_edges(*, z_grid=None, chi_grid=None, config, grid_kind: str = "centers", label: str = "pdf") -> np.ndarray | None:
    """Resolve chi bin edges for edge-grid empirical PDFs.

    Returns ``None`` for center grids, because no unambiguous native bin edges
    were supplied. For ``grid_kind='edges'`` the returned array has length
    ``ngrid + 1`` and is in the same distance units used elsewhere in the
    projected run.
    """
    kind = str(grid_kind).strip().lower()
    if kind == "centers":
        return None
    if kind != "edges":
        raise ValueError("grid_kind must be either 'centers' or 'edges'.")
    if chi_grid is not None:
        edges = load_array_spec(chi_grid, label=f"{label} chi_grid")
        edges = np.asarray(edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("chi_grid edges must be one-dimensional with at least two entries.")
        return edges
    if z_grid is None:
        raise ValueError(f"A common z_grid or chi_grid must be provided for {label} mode.")
    return zgrid_to_chi_edges(load_array_spec(z_grid, label=f"{label} z_grid"), config, label=label)

def resolve_common_chi_grid(*, z_grid=None, chi_grid=None, config, grid_kind: str = "centers", label: str = "pdf") -> np.ndarray:
    """Resolve the shared chi grid used by empirical-PDF inputs.

    Parameters
    ----------
    z_grid, chi_grid : object, optional
        Shared support grid expressed either in redshift or in comoving distance.
        When both are supplied, ``chi_grid`` takes precedence.
    config : object
        Projected correlation configuration providing the cosmology used when
        converting redshift to comoving distance.
    grid_kind : {'centers', 'edges'}, default='centers'
        Whether the supplied grid contains bin centers or bin edges.
    label : str, default='pdf'
        Human-readable label used in error messages.

    Returns
    -------
    ndarray
        One-dimensional float64 array of chi-grid centers in the same distance
        units used elsewhere in the projected run.
    """
    if chi_grid is not None:
        return _as_grid_centers(load_array_spec(chi_grid, label=f"{label} chi_grid"), grid_kind=grid_kind)
    if z_grid is None:
        raise ValueError(f"A common z_grid or chi_grid must be provided for {label} mode.")
    return zgrid_to_chi(load_array_spec(z_grid, label=f"{label} z_grid"), config, grid_kind=grid_kind, label=label)


def get_raw_column(table, name: str):
    """Return a catalog column without coercing its dtype."""
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


def vector_to_matrix(values) -> np.ndarray:
    """Convert vector-valued rows into a dense 2-D float64 matrix."""
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 2 and arr.dtype != object:
        return np.asarray(arr, dtype=np.float64)
    rows = []
    for item in arr:
        rows.append(np.asarray(item, dtype=np.float64))
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(rows).astype(np.float64, copy=False)


def read_parquet_dataframe(path: str, columns: list[str] | None = None):
    """Read a parquet table with pandas when that dependency is available."""
    try:  # pragma: no cover - depends on optional parquet stack
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise ImportError("Reading projected PDF parquet sources requires pandas with parquet support.") from exc
    return pd.read_parquet(path, columns=columns)


def columns_from_prefix(names: tuple[str, ...], prefix: str) -> list[str]:
    """Resolve PDF scalar-column names from a shared prefix."""
    cols = [str(name) for name in names if str(name).startswith(prefix)]
    if not cols:
        raise ValueError(f"No PDF columns found with prefix {prefix!r}.")
    try:
        cols.sort(key=lambda s: int(s[len(prefix):]))
    except Exception:
        cols.sort()
    return cols


def align_matrix_rows(matrix: np.ndarray, source_ids, catalog_ids, *, label: str = "pdf") -> np.ndarray:
    """Reorder PDF rows to match the associated catalog row IDs."""
    if source_ids is None or catalog_ids is None:
        return np.asarray(matrix, dtype=np.float64)
    source_ids = np.asarray(source_ids)
    catalog_ids = np.asarray(catalog_ids)
    lookup = {val: i for i, val in enumerate(source_ids.tolist())}
    try:
        order = np.asarray([lookup[val] for val in catalog_ids.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise KeyError(
            f"{label} row alignment failed because catalog id {exc.args[0]!r} is missing from the PDF source."
        ) from exc
    return np.asarray(matrix[order], dtype=np.float64)


def load_pdf_matrix(source, catalog, *, nrows: int, label: str = "pdf") -> np.ndarray:
    """Resolve an empirical PDF source into a normalized ``(nobj, ngrid)`` matrix."""
    if source is None:
        raise ValueError(f"{label} source is required for the selected catalog.")
    matrix = None
    source_ids = None
    if source.matrix is not None:
        matrix = np.asarray(source.matrix, dtype=np.float64)
    elif source.path:
        path = str(source.path)
        suffix = Path(path).suffix.lower()
        if suffix == ".npy":
            matrix = np.asarray(np.load(path), dtype=np.float64)
        elif suffix == ".npz":
            obj = np.load(path)
            key = source.array_key or next(iter(obj.files))
            matrix = np.asarray(obj[key], dtype=np.float64)
        elif suffix == ".parquet":
            if source.kind == "vector_column" or (source.column is not None and source.columns is None and source.prefix is None):
                cols = [str(source.column)]
                if source.id_column is not None:
                    cols.append(str(source.id_column))
                df = read_parquet_dataframe(path, columns=cols)
                matrix = vector_to_matrix(df[str(source.column)].to_numpy())
                if source.id_column is not None:
                    source_ids = np.asarray(df[str(source.id_column)])
            else:
                if source.columns is None:
                    df = read_parquet_dataframe(path, columns=None)
                    cols = columns_from_prefix(tuple(str(c) for c in df.columns), str(source.prefix))
                    use_cols = list(cols)
                    if source.id_column is not None:
                        use_cols.append(str(source.id_column))
                    df = df[use_cols]
                else:
                    use_cols = list(source.columns)
                    if source.id_column is not None:
                        use_cols.append(str(source.id_column))
                    df = read_parquet_dataframe(path, columns=use_cols)
                    cols = list(source.columns)
                matrix = np.column_stack([np.asarray(df[str(c)], dtype=np.float64) for c in cols])
                if source.id_column is not None:
                    source_ids = np.asarray(df[str(source.id_column)])
        else:
            matrix = np.asarray(np.loadtxt(path), dtype=np.float64)
    else:
        if source.kind == "vector_column":
            matrix = vector_to_matrix(get_raw_column(catalog, str(source.column)))
        else:
            names = catalog_column_names(catalog)
            cols = list(source.columns) if source.columns is not None else columns_from_prefix(names, str(source.prefix))
            matrix = np.column_stack([catalog_get_column(catalog, str(c), dtype=np.float64) for c in cols])
        if source.id_column is not None:
            source_ids = np.asarray(catalog_get_column(catalog, str(source.id_column)))
    if matrix.ndim != 2:
        raise ValueError(f"{label} sources must resolve to a 2-D matrix with shape (nobj, ngrid).")
    if source.id_column is not None and source.catalog_id_column is not None:
        catalog_ids = np.asarray(catalog_get_column(catalog, str(source.catalog_id_column)))
        matrix = align_matrix_rows(matrix, source_ids, catalog_ids, label=label)
    if matrix.shape[0] != int(nrows):
        raise ValueError(f"{label} matrix row count {matrix.shape[0]} does not match catalog length {nrows}.")
    if np.any(matrix < 0.0):
        raise ValueError(f"{label} matrices must be non-negative.")
    row_sum = np.asarray(matrix.sum(axis=1), dtype=np.float64)
    if np.any(row_sum <= 0.0):
        raise ValueError(f"{label} matrices must have strictly positive row sums.")
    return np.asarray(matrix / row_sum[:, None], dtype=np.float64)

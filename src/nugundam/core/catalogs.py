"""Helpers for adapting supported tabular inputs to 1-D NumPy columns.

These helpers keep the public API flexible while ensuring the heavy pair-
counting code operates on plain NumPy arrays after preparation. Supported
inputs include Astropy tables, pandas DataFrames, PyArrow tables, NumPy
structured arrays/recarrays, and plain mappings of column names to arrays.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def catalog_backend_name(table) -> str:
    """Return a short backend label for a supported catalog object."""
    if hasattr(table, "colnames"):
        return "astropy"
    if hasattr(table, "column_names") and hasattr(table, "num_rows"):
        return "pyarrow"
    if hasattr(table, "columns") and hasattr(table, "iloc"):
        return "pandas"
    if isinstance(table, Mapping):
        return "mapping"
    if isinstance(table, np.ndarray) and table.dtype.names is not None:
        return "structured"
    return type(table).__name__


def catalog_column_names(table) -> tuple[str, ...]:
    """Return the available column names for a supported catalog object."""
    if hasattr(table, "colnames"):
        return tuple(str(name) for name in table.colnames)
    if hasattr(table, "column_names") and hasattr(table, "num_rows"):
        return tuple(str(name) for name in table.column_names)
    if hasattr(table, "columns") and hasattr(table, "iloc"):
        return tuple(str(name) for name in table.columns)
    if isinstance(table, Mapping):
        return tuple(str(name) for name in table.keys())
    if isinstance(table, np.ndarray) and table.dtype.names is not None:
        return tuple(str(name) for name in table.dtype.names)
    raise TypeError(
        "Unsupported catalog input type. Expected an Astropy table, pandas "
        "DataFrame, PyArrow table, NumPy structured array, or mapping of "
        "column names to arrays."
    )


def catalog_has_column(table, name: str) -> bool:
    """Return whether a supported catalog exposes the requested column name."""
    return str(name) in catalog_column_names(table)


def catalog_nrows(table) -> int:
    """Return the number of rows in a supported catalog."""
    if hasattr(table, "num_rows"):
        return int(table.num_rows)
    if isinstance(table, Mapping):
        if not table:
            return 0
        first = next(iter(table.values()))
        return int(np.ravel(np.asarray(first)).shape[0])
    try:
        return int(len(table))
    except Exception as exc:  # pragma: no cover - defensive path
        raise TypeError(f"Cannot determine row count for catalog type {type(table)!r}.") from exc


def _materialize_column(value):
    if isinstance(value, np.ndarray):
        return value
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        for kwargs in ({"copy": False}, {"zero_copy_only": False}, {}):
            try:
                return np.asarray(to_numpy(**kwargs))
            except TypeError:
                continue
            except Exception:
                pass
    data = getattr(value, "data", None)
    if data is not None and not isinstance(data, memoryview):
        try:
            return np.asarray(data)
        except Exception:
            pass
    return np.asarray(value)


def catalog_get_column(table, name: str, *, dtype=None) -> np.ndarray:
    """Return a 1-D NumPy view/copy of a named catalog column.

    Parameters
    ----------
    table : object
        Supported tabular input.
    name : str
        Column name.
    dtype : data-type, optional
        Optional dtype requested for the returned array.
    """
    key = str(name)
    if hasattr(table, "columns") and hasattr(table, "iloc"):
        if key not in table.columns:
            raise KeyError(key)
        value = table[key]
    else:
        try:
            value = table[key]
        except Exception as exc:
            raise KeyError(key) from exc
    arr = _materialize_column(value)
    if dtype is not None:
        arr = np.asarray(arr, dtype=dtype)
    else:
        arr = np.asarray(arr)
    return np.ravel(arr)


__all__ = [
    "catalog_backend_name",
    "catalog_column_names",
    "catalog_get_column",
    "catalog_has_column",
    "catalog_nrows",
]

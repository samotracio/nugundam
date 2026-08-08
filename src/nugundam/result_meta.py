"""Metadata helpers attached to saved and plotted nuGUNDAM results."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import platform
import sys
from typing import Any

import numpy as np


_CONFIG_LEVELS = {"none", "compact", "full"}
_SMALL_ARRAY_MAX = 64


def _package_version() -> str:
    """
    Return the installed nuGUNDAM package version when available.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    try:
        from importlib.metadata import version
        return version("nugundam")
    except Exception:
        return "unknown"


def provenance_dict(run_kind: str) -> dict[str, Any]:
    """
    Build a provenance dictionary describing the current run and software environment.
    
    Parameters
    ----------
    run_kind : object
        Value for ``run_kind``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    return {
        "run_kind": run_kind,
        "package": "nugundam",
        "package_version": _package_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def _normalize_store_config(store_config: str | None, config: Any) -> str:
    level = store_config
    if level is None:
        level = getattr(config, "store_config", "compact")
    level = str(level).strip().lower()
    if level not in _CONFIG_LEVELS:
        raise ValueError(f"store_config must be one of {sorted(_CONFIG_LEVELS)}.")
    return level


def _compact_array(arr: np.ndarray) -> Any:
    arr = np.asarray(arr)
    if arr.size <= _SMALL_ARRAY_MAX:
        return arr.tolist()
    return {
        "__kind__": "ndarray_summary",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }


def _serialize_config_value(value: Any, *, level: str) -> Any:
    if level == "none":
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value if level == "full" else _compact_array(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _serialize_config_value(getattr(value, f.name), level=level) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _serialize_config_value(v, level=level) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_config_value(v, level=level) for v in value]
    if isinstance(value, tuple):
        return tuple(_serialize_config_value(v, level=level) for v in value)
    return value


def attach_roundtrip_context(
    obj: Any,
    *,
    config: Any,
    provenance: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
    store_config: str | None = None,
) -> Any:
    """
    Attach configuration and provenance metadata to a result object.
    
    Parameters
    ----------
    obj : object
        Value for ``obj``.
    config : object
        Value for ``config``. This argument is keyword-only.
    provenance : object
        Value for ``provenance``. This argument is keyword-only.
    extra_metadata : object, optional
        Value for ``extra_metadata``. This argument is keyword-only.
    store_config : {"none", "compact", "full"} or None, optional
        Level of configuration snapshot stored in ``metadata``. When omitted,
        the helper reads ``config.store_config`` when present and otherwise
        defaults to ``"compact"``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    metadata = dict(getattr(obj, "metadata", {}) or {})
    if extra_metadata:
        metadata.update(extra_metadata)
    level = _normalize_store_config(store_config, config)
    if level == "none":
        metadata.pop("config", None)
    else:
        metadata["config"] = _serialize_config_value(config, level=level)
    metadata["config_store"] = level
    metadata["provenance"] = provenance
    obj.metadata = metadata
    return obj

"""Metadata helpers attached to saved and plotted nuGUNDAM results."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import platform
import sys
from typing import Any


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


def attach_roundtrip_context(obj: Any, *, config: Any, provenance: dict[str, Any], extra_metadata: dict[str, Any] | None = None) -> Any:
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
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    metadata = dict(getattr(obj, "metadata", {}) or {})
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata["config"] = asdict(config) if is_dataclass(config) else config
    metadata["provenance"] = provenance
    obj.metadata = metadata
    return obj

"""Round-trip serialization helpers for nuGUNDAM dataclass results."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import importlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

_MANIFEST_KEY = "__gundam_manifest__"
_FORMAT_VERSION = 1


def _array_key(prefix: str) -> str:
    """
    Array key.
    
    Parameters
    ----------
    prefix : object
        Value for ``prefix``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    key = prefix.replace('.', '__').replace('[', '_').replace(']', '_')
    return key.strip('_') or 'root'


def _serialize_value(value: Any, arrays: dict[str, np.ndarray], prefix: str):
    """
    Serialize value into the manifest representation.
    
    Parameters
    ----------
    value : object
        Value for ``value``.
    arrays : object
        Value for ``arrays``.
    prefix : object
        Value for ``prefix``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if isinstance(value, np.ndarray):
        key = _array_key(prefix)
        arrays[key] = np.asarray(value)
        return {"__kind__": "ndarray", "key": key}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {
            "__kind__": "dataclass",
            "module": value.__class__.__module__,
            "class": value.__class__.__name__,
            "fields": {
                field.name: _serialize_value(getattr(value, field.name), arrays, f"{prefix}.{field.name}")
                for field in fields(value)
            },
        }
    if isinstance(value, dict):
        return {
            "__kind__": "dict",
            "items": [
                [_serialize_value(key, arrays, f"{prefix}.key{i}"), _serialize_value(val, arrays, f"{prefix}.{key}")]
                for i, (key, val) in enumerate(value.items())
            ],
        }
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [_serialize_value(v, arrays, f"{prefix}[{i}]") for i, v in enumerate(value)]}
    if isinstance(value, list):
        return {"__kind__": "list", "items": [_serialize_value(v, arrays, f"{prefix}[{i}]") for i, v in enumerate(value)]}
    if isinstance(value, Path):
        return {"__kind__": "path", "value": str(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported value in nuGUNDAM result serialization: {type(value)!r}")


def _resolve_dataclass(module_name: str, class_name: str):
    """
    Resolve dataclass.
    
    Parameters
    ----------
    module_name : object
        Value for ``module_name``.
    class_name : object
        Value for ``class_name``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if not module_name.startswith('nugundam.'):
        raise ValueError(f"Refusing to reconstruct unsupported dataclass module: {module_name!r}")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not hasattr(cls, '__dataclass_fields__'):
        raise TypeError(f"Serialized class is not a dataclass: {module_name}.{class_name}")
    return cls


def _deserialize_value(node, archive):
    """
    Deserialize value from the manifest representation.
    
    Parameters
    ----------
    node : object
        Value for ``node``.
    archive : object
        Value for ``archive``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if isinstance(node, dict) and '__kind__' in node:
        kind = node['__kind__']
        if kind == 'ndarray':
            return np.asarray(archive[node['key']])
        if kind == 'dataclass':
            cls = _resolve_dataclass(node['module'], node['class'])
            kwargs = {name: _deserialize_value(value, archive) for name, value in node['fields'].items()}
            return cls(**kwargs)
        if kind == 'dict':
            return {
                _deserialize_value(key, archive): _deserialize_value(value, archive)
                for key, value in node['items']
            }
        if kind == 'tuple':
            return tuple(_deserialize_value(v, archive) for v in node['items'])
        if kind == 'list':
            return [_deserialize_value(v, archive) for v in node['items']]
        if kind == 'path':
            return Path(node['value'])
        raise ValueError(f"Unknown serialized nuGUNDAM node kind: {kind!r}")
    return node


def save_result(result: Any, path: str | Path) -> None:
    """
    Serialize a nuGUNDAM result to the native compressed ``.npz`` format.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    path : object
        Value for ``path``.
    
    Returns
    -------
    None
        Object returned by this helper.
    """
    path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    manifest = {
        'format': 'nugundam-result',
        'format_version': _FORMAT_VERSION,
        'root': _serialize_value(result, arrays, 'root'),
    }
    payload = {_MANIFEST_KEY: np.array(json.dumps(manifest, separators=(',', ':')))}
    payload.update(arrays)
    with path.open('wb') as fp:
        np.savez_compressed(fp, **payload)


def read_result(path: str | Path):
    """
    Read a nuGUNDAM result from disk and reconstruct the saved object.
    
    Parameters
    ----------
    path : object
        Value for ``path``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if _MANIFEST_KEY in archive:
                manifest = json.loads(np.asarray(archive[_MANIFEST_KEY]).item())
                if manifest.get('format') != 'nugundam-result':
                    raise ValueError('Unsupported nuGUNDAM result format')
                return _deserialize_value(manifest['root'], archive)
    except Exception:
        pass
    with path.open('rb') as fp:
        return pickle.load(fp)


def write_result(result: Any, path: str | Path) -> None:
    """
    Alias for :func:`save_result` using the legacy verb ``write``.
    
    Parameters
    ----------
    result : object
        Value for ``result``.
    path : object
        Value for ``path``.
    
    Returns
    -------
    None
        Object returned by this helper.
    """
    save_result(result, path)

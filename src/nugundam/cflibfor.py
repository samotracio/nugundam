"""Python wrapper around the compiled Fortran extension.

This module keeps the public import path ``nugundam.cflibfor`` stable while
loading the actual F2PY extension from the private module
``nugundam._cflibfor``. Keeping a Python wrapper in front of the compiled
module makes it safe for tests to monkeypatch ``cff.mod.skll2d`` and similar
helpers, because the wrapper can store Python-level overrides even when the
underlying F2PY object rejects attribute assignment.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised in editable/wheel installs
    from . import _cflibfor as _compiled
except Exception:  # pragma: no cover - source-tree tests without compiled ext
    _compiled = None


class _FortranModuleProxy:
    """Proxy exposing the compiled ``mod`` namespace with patchable overrides."""

    def __init__(self, compiled_mod: Any | None):
        object.__setattr__(self, '_compiled_mod', compiled_mod)
        object.__setattr__(self, '_overrides', {})

    def __getattr__(self, name: str) -> Any:
        overrides = object.__getattribute__(self, '_overrides')
        if name in overrides:
            return overrides[name]
        compiled_mod = object.__getattribute__(self, '_compiled_mod')
        if compiled_mod is None:
            raise ImportError(
                "nugundam._cflibfor compiled extension is not available. "
                "Build/install the package so the Fortran module is compiled "
                f"before calling '{name}'."
            )
        return getattr(compiled_mod, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        object.__getattribute__(self, '_overrides')[name] = value

    def __delattr__(self, name: str) -> None:
        overrides = object.__getattribute__(self, '_overrides')
        if name in overrides:
            del overrides[name]
            return
        raise AttributeError(name)

    def __dir__(self) -> list[str]:
        names = set(object.__getattribute__(self, '_overrides'))
        compiled_mod = object.__getattribute__(self, '_compiled_mod')
        if compiled_mod is not None:
            names.update(dir(compiled_mod))
        return sorted(names)


_compiled_mod = None if _compiled is None else getattr(_compiled, 'mod', None)
mod = _FortranModuleProxy(_compiled_mod)
compiled_available = _compiled_mod is not None

__all__ = ['mod', 'compiled_available']

"""Runtime backend detection.

Priority order when ``backend='auto'``:

1. ``cython``  -- compiled C extension (fastest, needs a C toolchain at build)
2. ``numba``   -- JIT compiled native code
3. ``python``  -- pure-Python loop reference (always available)
"""
from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    CYTHON = "cython"
    NUMBA = "numba"
    PYTHON = "python"


def detect_backend() -> Backend:
    try:
        from . import _cython_core  # noqa: F401
        return Backend.CYTHON
    except Exception:
        pass
    try:
        from ._numba import _HAVE_NUMBA
        if _HAVE_NUMBA:
            return Backend.NUMBA
    except Exception:
        pass
    return Backend.PYTHON


ACTIVE_BACKEND: Backend = detect_backend()


def resolve(backend: str | Backend) -> Backend:
    if isinstance(backend, Backend):
        return backend
    b = backend.lower()
    if b == "auto":
        return ACTIVE_BACKEND
    try:
        return Backend(b)
    except ValueError:
        raise ValueError(
            f"Unknown backend {backend!r}; choose from 'auto','cython','numba','python'."
        )

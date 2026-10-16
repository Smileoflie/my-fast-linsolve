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
        from ._kernels import _HAVE_NUMBA
        if _HAVE_NUMBA:
            return Backend.NUMBA
    except Exception:
        pass
    return Backend.PYTHON


ACTIVE_BACKEND: Backend = detect_backend()


def kernels_for(backend: "Backend"):
    """Return the namespace that holds the kernels for ``backend``.

    Every backend exposes the same kernel names (``gauss_solve``,
    ``lu_factor``, ``jacobi``, ...), so callers never need to know which
    implementation they are talking to.
    """
    if backend is Backend.CYTHON:
        from . import _cython_core
        return _cython_core
    from . import _kernels
    return _kernels.nb if backend is Backend.NUMBA else _kernels


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
        ) from None


def backend_info() -> dict:
    """Report availability of each backend, with the reason when unavailable.

    Useful for diagnosing silent degradation, e.g. a Cython extension whose
    ABI no longer matches the installed NumPy::

        >>> from linsolve import backend_info
        >>> backend_info()["cython"]
        (False, 'ImportError: ...')
    """
    info = {}
    try:
        from . import _cython_core  # noqa: F401
        info["cython"] = (True, "compiled extension available")
    except Exception as exc:
        info["cython"] = (False, f"{type(exc).__name__}: {exc}")
    try:
        from ._kernels import _HAVE_NUMBA
        info["numba"] = (
            (True, "numba available") if _HAVE_NUMBA
            else (False, "numba not installed")
        )
    except Exception as exc:
        info["numba"] = (False, f"{type(exc).__name__}: {exc}")
    info["python"] = (True, "pure-Python reference, always available")
    info["active"] = ACTIVE_BACKEND.value
    return info

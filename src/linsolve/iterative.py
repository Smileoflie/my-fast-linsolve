"""Iterative linear solvers: Jacobi, Gauss-Seidel, SOR."""
from __future__ import annotations

from typing import Tuple, Dict

import numpy as np

from . import _raw
from ._backend import Backend, resolve


def _as_float(A, b, x0):
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must have length equal to A.shape[0].")
    if x0 is None:
        x0 = np.zeros(A.shape[0], dtype=np.float64)
    else:
        x0 = np.ascontiguousarray(x0, dtype=np.float64)
    return A, b, x0


def _pack(x, it, r, converged) -> Tuple[np.ndarray, Dict]:
    return x, {"iterations": int(it), "residual": float(r), "converged": bool(converged)}


def jacobi(A, b, x0=None, tol: float = 1e-8, max_iter: int = 1000,
           backend: str = "auto"):
    """Jacobi iteration. Returns ``(x, info)``."""
    A, b, x0 = _as_float(A, b, x0)
    be = resolve(backend)
    if be is Backend.CYTHON:
        from . import _cython_core
        return _pack(*_cython_core.jacobi_cy(A, b, x0, tol, max_iter))
    if be is Backend.NUMBA:
        from ._numba import jacobi_nb
        return _pack(*jacobi_nb(A, b, x0, tol, max_iter))
    return _pack(*_raw.jacobi(A, b, x0, tol, max_iter))


def gauss_seidel(A, b, x0=None, tol: float = 1e-8, max_iter: int = 1000,
                 backend: str = "auto"):
    """Gauss-Seidel iteration. Returns ``(x, info)``."""
    A, b, x0 = _as_float(A, b, x0)
    be = resolve(backend)
    if be is Backend.CYTHON:
        from . import _cython_core
        return _pack(*_cython_core.gauss_seidel_cy(A, b, x0, tol, max_iter))
    if be is Backend.NUMBA:
        from ._numba import gauss_seidel_nb
        return _pack(*gauss_seidel_nb(A, b, x0, tol, max_iter))
    return _pack(*_raw.gauss_seidel(A, b, x0, tol, max_iter))


def sor(A, b, omega: float = 1.0, x0=None, tol: float = 1e-8,
        max_iter: int = 1000, backend: str = "auto"):
    """Successive Over-Relaxation iteration. Returns ``(x, info)``."""
    A, b, x0 = _as_float(A, b, x0)
    if not (0.0 < omega < 2.0):
        raise ValueError("omega must be in (0, 2) for convergence.")
    be = resolve(backend)
    if be is Backend.CYTHON:
        from . import _cython_core
        return _pack(*_cython_core.sor_cy(A, b, omega, x0, tol, max_iter))
    if be is Backend.NUMBA:
        from ._numba import sor_nb
        return _pack(*sor_nb(A, b, omega, x0, tol, max_iter))
    return _pack(*_raw.sor(A, b, omega, x0, tol, max_iter))

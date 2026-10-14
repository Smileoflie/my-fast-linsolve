"""Direct linear solvers: Gauss elimination, LU, Cholesky."""
from __future__ import annotations

import numpy as np

from . import _raw
from ._backend import Backend, resolve


def _as_float(A: np.ndarray, b: np.ndarray):
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must have length equal to A.shape[0].")
    return A, b


def gauss_solve(A, b, backend: str = "auto") -> np.ndarray:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting."""
    A, b = _as_float(A, b)
    be = resolve(backend)
    if be is Backend.CYTHON:
        from . import _cython_core
        return _cython_core.gauss_solve_cy(A, b)
    if be is Backend.NUMBA:
        from ._numba import gauss_solve_nb
        return gauss_solve_nb(A, b)
    return _raw.gauss_solve(A, b)


def lu_solve(A, b, backend: str = "auto") -> np.ndarray:
    """Solve ``A x = b`` by LU factorisation (PA = LU, Doolittle, partial pivoting)."""
    A, b = _as_float(A, b)
    be = resolve(backend)
    if be is Backend.CYTHON:
        from . import _cython_core
        return _cython_core.lu_solve_cy(A, b)
    if be is Backend.NUMBA:
        from ._numba import lu_solve_nb
        return lu_solve_nb(A, b)
    return _raw.lu_solve(A, b)


def cholesky_solve(A, b, backend: str = "auto") -> np.ndarray:
    """Solve ``A x = b`` by Cholesky factorisation for symmetric positive-definite A."""
    A, b = _as_float(A, b)
    be = resolve(backend)
    if be is Backend.CYTHON:
        from . import _cython_core
        return _cython_core.cholesky_solve_cy(A, b)
    if be is Backend.NUMBA:
        from ._numba import cholesky_solve_nb
        return cholesky_solve_nb(A, b)
    return _raw.cholesky_solve(A, b)

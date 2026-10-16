"""Direct linear solvers: Gauss elimination, LU, Cholesky.

Backend selection is a single lookup: every backend exposes the same kernel
names (``gauss_solve``, ``lu_factor``, ...), so the wrappers below only choose
*which* namespace to call into and never duplicate algorithm logic.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ._backend import kernels_for, resolve


def _as_float(A, b) -> Tuple[np.ndarray, np.ndarray]:
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    if b.ndim != 1:
        raise ValueError(f"b must be a 1-D vector, got {b.ndim}-D array.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must have length equal to A.shape[0].")
    return A, b


def _check_symmetric(A: np.ndarray):
    """Cholesky reads only the lower triangle; a non-symmetric A would
    otherwise silently produce a wrong answer instead of an error."""
    scale = float(np.abs(A).max()) if A.size else 1.0
    if not np.allclose(A, A.T, rtol=1e-10, atol=1e-12 * max(scale, 1.0)):
        raise ValueError(
            "cholesky_solve requires a symmetric matrix; "
            "use lu_solve/gauss_solve for non-symmetric A."
        )


def _as_rhs(b, n: int):
    """Normalise a right-hand side to ``(B, squeeze)`` with ``B`` shaped (n, k).

    Accepts a 1-D vector of length n, or a 2-D (n, k) matrix so several
    right-hand sides can be solved against a single factorisation.
    """
    B = np.asarray(b, dtype=np.float64)
    if B.ndim == 1:
        if B.shape[0] != n:
            raise ValueError("b must have length equal to A.shape[0].")
        # np.ascontiguousarray: b may be a strided view (e.g. B[:, 0]), and the
        # compiled backends require C-contiguous buffers
        return np.ascontiguousarray(B).reshape(n, 1), True
    if B.ndim == 2:
        if B.shape[0] != n:
            raise ValueError("b must have A.shape[0] rows.")
        return np.ascontiguousarray(B), False
    raise ValueError(f"b must be a 1-D vector or a 2-D (n, k) matrix, got {B.ndim}-D.")


# ---------------------------------------------------------------------------
# Factorisations (reusable across right-hand sides)
# ---------------------------------------------------------------------------
def lu_factor(A, backend: str = "auto"):
    """Factorise ``A`` so that ``P A = L U``.

    Returns ``(LU, piv)``. ``LU`` packs the unit lower-triangular multipliers
    and the upper triangle; ``piv[k]`` records the row interchanged with row
    ``k``. Feed the pair to :func:`lu_solve_factorized` -- the factorisation is
    O(n^3) once, each further solve only O(n^2)::

        LU, piv = lu_factor(A)
        x1 = lu_solve_factorized(LU, piv, b1)
        x2 = lu_solve_factorized(LU, piv, b2)   # no re-factorisation
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    return kernels_for(resolve(backend)).lu_factor(A)


def lu_solve_factorized(LU, piv, b, backend: str = "auto"):
    """Solve ``A x = b`` from an :func:`lu_factor` result.

    ``b`` may be 1-D ``(n,)`` or 2-D ``(n, k)``; the result has the same shape.
    """
    LU = np.ascontiguousarray(LU, dtype=np.float64)
    if LU.ndim != 2 or LU.shape[0] != LU.shape[1]:
        raise ValueError("LU must be a square 2D matrix.")
    piv = np.ascontiguousarray(piv, dtype=np.int64)
    B, squeeze = _as_rhs(b, LU.shape[0])
    X = kernels_for(resolve(backend)).lu_solve_from_factor(LU, piv, B)
    return X[:, 0] if squeeze else X


def cholesky_factor(A, backend: str = "auto"):
    """Cholesky factorisation ``A = L L^T`` for symmetric positive-definite A.

    Returns the lower-triangular ``L``; feed it to
    :func:`cholesky_solve_factorized` to reuse it across right-hand sides.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    _check_symmetric(A)
    return kernels_for(resolve(backend)).cholesky_factor(A)


def cholesky_solve_factorized(L, b, backend: str = "auto"):
    """Solve ``A x = b`` from a :func:`cholesky_factor` result."""
    L = np.ascontiguousarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square 2D matrix.")
    B, squeeze = _as_rhs(b, L.shape[0])
    X = kernels_for(resolve(backend)).cholesky_solve_from_factor(L, B)
    return X[:, 0] if squeeze else X


# ---------------------------------------------------------------------------
# One-shot solvers
# ---------------------------------------------------------------------------
def gauss_solve(A, b, backend: str = "auto") -> np.ndarray:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting."""
    A, b = _as_float(A, b)
    return kernels_for(resolve(backend)).gauss_solve(A, b)


def lu_solve(A, b, backend: str = "auto") -> np.ndarray:
    """Solve ``A x = b`` by LU factorisation (PA = LU, Doolittle, partial pivoting)."""
    A, b = _as_float(A, b)
    be = resolve(backend)
    K = kernels_for(be)
    LU, piv = K.lu_factor(A)
    return K.lu_solve_from_factor(LU, piv, b.reshape(-1, 1))[:, 0]

def cholesky_solve(A, b, backend: str = "auto") -> np.ndarray:
    """Solve ``A x = b`` by Cholesky factorisation for symmetric positive-definite A."""
    A, b = _as_float(A, b)
    _check_symmetric(A)
    be = resolve(backend)
    K = kernels_for(be)
    L = K.cholesky_factor(A)
    return K.cholesky_solve_from_factor(L, b.reshape(-1, 1))[:, 0]

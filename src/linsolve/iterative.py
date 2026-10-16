"""Iterative linear solvers: Jacobi, Gauss-Seidel, SOR."""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np

from ._backend import kernels_for, resolve


def _as_float(A, b, x0):
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    if b.ndim != 1:
        raise ValueError(f"b must be a 1-D vector, got {b.ndim}-D array.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must have length equal to A.shape[0].")
    if x0 is None:
        x0 = np.zeros(A.shape[0], dtype=np.float64)
    else:
        x0 = np.ascontiguousarray(x0, dtype=np.float64)
        if x0.ndim != 1 or x0.shape[0] != A.shape[0]:
            raise ValueError("x0 must be a 1-D vector of length A.shape[0].")
    return A, b, x0


def _check_diag(A: np.ndarray):
    """A zero diagonal makes every sweep divide by zero and silently
    produces inf/nan; reject it up front."""
    if np.any(np.diagonal(A) == 0.0):
        raise ValueError(
            "A has a zero diagonal entry; Jacobi/Gauss-Seidel/SOR require "
            "A[i, i] != 0."
        )


def _threshold(b, rtol, atol, tol):
    """Convert rtol/atol into the absolute residual threshold used by the
    kernels: ||r|| <= atol + rtol * ||b||  (SciPy convention)."""
    if tol is not None:
        warnings.warn(
            "`tol` is deprecated and treated as an absolute tolerance; "
            "use `rtol`/`atol` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        atol, rtol = float(tol), 0.0
    bnorm = float(np.linalg.norm(b))
    return atol + rtol * bnorm, bnorm


def _pack(x, it, r, converged, bnorm) -> Tuple[np.ndarray, Dict]:
    # info["residual"] is the *relative* residual ||b - Ax|| / ||b||
    rel = (r / bnorm) if bnorm > 0.0 else (0.0 if r == 0.0 else float("inf"))
    return x, {
        "iterations": int(it),
        "residual": float(rel),
        "residual_abs": float(r),
        "converged": bool(converged),
    }


def jacobi(A, b, x0=None, *, rtol: float = 1e-8, atol: float = 0.0,
           max_iter: int = 1000, tol=None, backend: str = "auto"):
    """Jacobi iteration. Returns ``(x, info)``.

    Convergence: ``||b - A x|| <= atol + rtol * ||b||``.
    ``info["residual"]`` is the relative residual ``||b - A x|| / ||b||``;
    ``info["residual_abs"]`` is the absolute norm. ``tol`` is a deprecated
    alias for an absolute tolerance.
    """
    A, b, x0 = _as_float(A, b, x0)
    _check_diag(A)
    thr, bnorm = _threshold(b, rtol, atol, tol)
    K = kernels_for(resolve(backend))
    return _pack(*K.jacobi(A, b, x0, thr, max_iter), bnorm)


def gauss_seidel(A, b, x0=None, *, rtol: float = 1e-8, atol: float = 0.0,
                 max_iter: int = 1000, tol=None, backend: str = "auto"):
    """Gauss-Seidel iteration. Returns ``(x, info)``. See :func:`jacobi`."""
    A, b, x0 = _as_float(A, b, x0)
    _check_diag(A)
    thr, bnorm = _threshold(b, rtol, atol, tol)
    K = kernels_for(resolve(backend))
    return _pack(*K.gauss_seidel(A, b, x0, thr, max_iter), bnorm)


def sor(A, b, omega: float = 1.0, x0=None, *, rtol: float = 1e-8,
        atol: float = 0.0, max_iter: int = 1000, tol=None,
        backend: str = "auto"):
    """Successive Over-Relaxation iteration. Returns ``(x, info)``. See :func:`jacobi`."""
    A, b, x0 = _as_float(A, b, x0)
    _check_diag(A)
    if not (0.0 < omega < 2.0):
        raise ValueError("omega must be in (0, 2) for convergence.")
    thr, bnorm = _threshold(b, rtol, atol, tol)
    K = kernels_for(resolve(backend))
    return _pack(*K.sor(A, b, omega, x0, thr, max_iter), bnorm)

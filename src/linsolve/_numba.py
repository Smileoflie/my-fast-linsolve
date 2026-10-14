"""Numba JIT implementations.

Same algorithms as :mod:`linsolve._raw`, decorated with ``@nj`` so the inner
Python loops compile to native machine code on first call.  Importing this
module is safe even if Numba is absent; the public wrappers in
:mod:`linsolve.direct` / :mod:`linsolve.iterative` detect availability.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba missing
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(func):
            return func
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return wrap


# ---------------------------------------------------------------------------
# Direct methods
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def gauss_solve_nb(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    M = A.astype(np.float64).copy()
    y = b.astype(np.float64).copy()

    for k in range(n):
        pivot = k
        best = abs(M[k, k])
        for i in range(k + 1, n):
            v = abs(M[i, k])
            if v > best:
                best = v
                pivot = i
        if best < 1e-15:
            raise ValueError("Matrix is singular to working precision.")
        if pivot != k:
            for j in range(n):
                tmp = M[k, j]; M[k, j] = M[pivot, j]; M[pivot, j] = tmp
            tmp = y[k]; y[k] = y[pivot]; y[pivot] = tmp

        aik_inv = 1.0 / M[k, k]
        for i in range(k + 1, n):
            factor = M[i, k] * aik_inv
            M[i, k] = 0.0
            for j in range(k + 1, n):
                M[i, j] -= factor * M[k, j]
            y[i] -= factor * y[k]

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x


@njit(cache=True, fastmath=True)
def lu_solve_nb(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    M = A.astype(np.float64).copy()
    y = b.astype(np.float64).copy()

    for k in range(n):
        pivot = k
        best = abs(M[k, k])
        for i in range(k + 1, n):
            v = abs(M[i, k])
            if v > best:
                best = v
                pivot = i
        if best < 1e-15:
            raise ValueError("Matrix is singular to working precision.")
        if pivot != k:
            for j in range(n):
                tmp = M[k, j]; M[k, j] = M[pivot, j]; M[pivot, j] = tmp
            tmp = y[k]; y[k] = y[pivot]; y[pivot] = tmp

        inv = 1.0 / M[k, k]
        for i in range(k + 1, n):
            M[i, k] *= inv
            lik = M[i, k]
            for j in range(k + 1, n):
                M[i, j] -= lik * M[k, j]

    for i in range(1, n):
        s = y[i]
        for j in range(i):
            s -= M[i, j] * y[j]
        y[i] = s

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x


@njit(cache=True, fastmath=True)
def cholesky_solve_nb(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    raise ValueError("Matrix is not positive definite.")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s / L[i, i]

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= L[j, i] * x[j]
        x[i] = s / L[i, i]
    return x


# ---------------------------------------------------------------------------
# Iterative methods (fully inlined for numba)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _residual_norm(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    n = A.shape[0]
    acc = 0.0
    for i in range(n):
        s = b[i]
        for j in range(n):
            s -= A[i, j] * x[j]
        acc += s * s
    return acc ** 0.5


@njit(cache=True, fastmath=True)
def jacobi_nb(A: np.ndarray, b: np.ndarray, x0: np.ndarray,
              tol: float, max_iter: int):
    n = A.shape[0]
    x = x0.astype(np.float64).copy()
    x_new = np.zeros(n, dtype=np.float64)
    for it in range(1, max_iter + 1):
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i, j] * x[j]
            x_new[i] = s / A[i, i]
        x[:] = x_new
        r = _residual_norm(A, b, x)
        if r < tol:
            return x, it, r, True
    return x, max_iter, _residual_norm(A, b, x), False


@njit(cache=True, fastmath=True)
def gauss_seidel_nb(A: np.ndarray, b: np.ndarray, x0: np.ndarray,
                    tol: float, max_iter: int):
    n = A.shape[0]
    x = x0.astype(np.float64).copy()
    for it in range(1, max_iter + 1):
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i, j] * x[j]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = s / A[i, i]
        r = _residual_norm(A, b, x)
        if r < tol:
            return x, it, r, True
    return x, max_iter, _residual_norm(A, b, x), False


@njit(cache=True, fastmath=True)
def sor_nb(A: np.ndarray, b: np.ndarray, omega: float, x0: np.ndarray,
           tol: float, max_iter: int):
    n = A.shape[0]
    x = x0.astype(np.float64).copy()
    for it in range(1, max_iter + 1):
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i, j] * x[j]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = x[i] + omega * (s / A[i, i] - x[i])
        r = _residual_norm(A, b, x)
        if r < tol:
            return x, it, r, True
    return x, max_iter, _residual_norm(A, b, x), False

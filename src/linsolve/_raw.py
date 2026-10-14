"""Pure-Python reference implementations (plain loops over numpy arrays).

These are deliberately *not* vectorised with numpy slicing: they use explicit
Python ``for`` loops with scalar indexing, which is the slow baseline that
Numba JIT and Cython static compilation are meant to beat.

All functions accept and return :class:`numpy.ndarray`.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Direct methods
# ---------------------------------------------------------------------------
def gauss_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gaussian elimination with partial pivoting (pure Python loop)."""
    n = A.shape[0]
    M = A.astype(np.float64, copy=True)
    y = b.astype(np.float64, copy=True)

    for k in range(n):
        # partial pivoting: find the row with largest |M[i, k]| in column k
        pivot = k
        for i in range(k + 1, n):
            if abs(M[i, k]) > abs(M[pivot, k]):
                pivot = i
        if abs(M[pivot, k]) < 1e-15:
            raise ValueError("Matrix is singular to working precision.")
        if pivot != k:
            M[[k, pivot]] = M[[pivot, k]]
            y[k], y[pivot] = y[pivot], y[k]

        aik_inv = 1.0 / M[k, k]
        for i in range(k + 1, n):
            factor = M[i, k] * aik_inv
            M[i, k] = 0.0
            for j in range(k + 1, n):
                M[i, j] -= factor * M[k, j]
            y[i] -= factor * y[k]

    # back substitution
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x


def lu_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """LU solve with partial pivoting (PA = LU, Doolittle form).

    On exit U is upper triangular, L is unit lower triangular (stored below
    the diagonal of the working matrix), then forward + back substitution.
    """
    n = A.shape[0]
    M = A.astype(np.float64, copy=True)
    y = b.astype(np.float64, copy=True)

    for k in range(n):
        pivot = k
        for i in range(k + 1, n):
            if abs(M[i, k]) > abs(M[pivot, k]):
                pivot = i
        if abs(M[pivot, k]) < 1e-15:
            raise ValueError("Matrix is singular to working precision.")
        if pivot != k:
            M[[k, pivot]] = M[[pivot, k]]
            y[k], y[pivot] = y[pivot], y[k]

        inv = 1.0 / M[k, k]
        for i in range(k + 1, n):
            M[i, k] *= inv  # multiplier -> L
            lik = M[i, k]
            for j in range(k + 1, n):
                M[i, j] -= lik * M[k, j]

    # forward substitution: Ly = Pb (L unit lower triangular)
    for i in range(1, n):
        s = y[i]
        for j in range(i):
            s -= M[i, j] * y[j]
        y[i] = s

    # back substitution: Ux = y
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x


def cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cholesky solve for symmetric positive-definite matrices (A = L L^T)."""
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

    # forward: L y = b
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s / L[i, i]

    # back: L^T x = y
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= L[j, i] * x[j]
        x[i] = s / L[i, i]
    return x


# ---------------------------------------------------------------------------
# Iterative methods
# ---------------------------------------------------------------------------
def _iterate(A, b, x0, tol, max_iter, update):
    n = A.shape[0]
    x = x0.astype(np.float64, copy=True)
    for it in range(1, max_iter + 1):
        update(A, b, x)
        # residual r = b - A x
        rnorm = 0.0
        for i in range(n):
            s = b[i]
            for j in range(n):
                s -= A[i, j] * x[j]
            rnorm += s * s
        rnorm = rnorm ** 0.5
        if rnorm < tol:
            return x, it, rnorm, True
    return x, max_iter, rnorm, False


def jacobi(A: np.ndarray, b: np.ndarray, x0: np.ndarray | None = None,
           tol: float = 1e-8, max_iter: int = 1000):
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n, dtype=np.float64)

    def update(A, b, x):
        x_new = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i, j] * x[j]
            x_new[i] = s / A[i, i]
        x[:] = x_new

    return _iterate(A, b, x0, tol, max_iter, update)


def gauss_seidel(A: np.ndarray, b: np.ndarray, x0: np.ndarray | None = None,
                 tol: float = 1e-8, max_iter: int = 1000):
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n, dtype=np.float64)

    def update(A, b, x):
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i, j] * x[j]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = s / A[i, i]

    return _iterate(A, b, x0, tol, max_iter, update)


def sor(A: np.ndarray, b: np.ndarray, omega: float = 1.0,
        x0: np.ndarray | None = None,
        tol: float = 1e-8, max_iter: int = 1000):
    if not (0.0 < omega < 2.0):
        raise ValueError("omega must be in (0, 2) for convergence.")
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n, dtype=np.float64)

    def update(A, b, x):
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i, j] * x[j]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = x[i] + omega * (s / A[i, i] - x[i])

    return _iterate(A, b, x0, tol, max_iter, update)

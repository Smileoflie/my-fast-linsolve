"""Shared computational kernels -- the single source of truth.

Every algorithm is written **once** here, as plain Python with explicit loops.
The very same source serves two backends:

* ``python`` -- the plain functions below, executed by CPython (slow baseline)
* ``numba``  -- ``kernels.nb.*``, the identical source compiled by ``@njit``

Two rules make a single source possible:

1. **No closures.** ``@njit`` cannot capture enclosing variables, so every
   function takes all of its inputs as arguments.
2. **No cross-kernel calls.** ``@njit`` can only call compiled callees, so
   short cross-cutting loops (the residual norm, the singularity threshold)
   are written inline in the functions that need them.

Historically these algorithms lived in three files (``_raw.py``,
``_numba.py`` and ``_cython_core.pyx``) and drifted apart -- that duplication is
exactly how a behavioural difference in ``max_iter=0`` slipped in. The Cython
core remains a second implementation because it is compiled ahead of time, but
it mirrors this file function for function.
"""
from __future__ import annotations

from math import sqrt
from types import SimpleNamespace

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


def _jit(fn):
    """Return a numba-compiled twin of ``fn`` (or ``fn`` itself without numba)."""
    if not _HAVE_NUMBA:
        return fn
    return njit(cache=True, fastmath=True)(fn)


# eps(float64), as a literal: np.finfo support in nopython mode varies across
# numba versions
_EPS = 2.220446049250313e-16


# ---------------------------------------------------------------------------
# Direct methods
# ---------------------------------------------------------------------------
def gauss_solve(A, b):
    """Gaussian elimination with partial pivoting."""
    n = A.shape[0]
    M = A.astype(np.float64)
    y = b.astype(np.float64)
    # relative singularity threshold eps * max|A| * n; an absolute threshold
    # would misjudge a merely scaled (but well-conditioned) matrix as singular
    sing_tol = _EPS * np.abs(M).max() * n

    for k in range(n):
        pivot = k
        best = abs(M[k, k])
        for i in range(k + 1, n):
            v = abs(M[i, k])
            if v > best:
                best = v
                pivot = i
        if best <= sing_tol:
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


def lu_factor(A):
    """PA = LU with partial pivoting (Doolittle form).

    Returns ``(LU, piv)``. ``LU`` packs the unit lower triangle (multipliers,
    diagonal implicitly 1) and the upper triangle; ``piv[k]`` is the row that
    was interchanged with row ``k`` at elimination step ``k``. Solving with the
    factorisation reuses it for any number of right-hand sides.
    """
    n = A.shape[0]
    M = A.astype(np.float64)
    piv = np.zeros(n, dtype=np.int64)
    sing_tol = _EPS * np.abs(M).max() * n

    for k in range(n):
        pivot = k
        best = abs(M[k, k])
        for i in range(k + 1, n):
            v = abs(M[i, k])
            if v > best:
                best = v
                pivot = i
        piv[k] = pivot
        if best <= sing_tol:
            raise ValueError("Matrix is singular to working precision.")
        if pivot != k:
            for j in range(n):
                tmp = M[k, j]; M[k, j] = M[pivot, j]; M[pivot, j] = tmp

        inv = 1.0 / M[k, k]
        for i in range(k + 1, n):
            M[i, k] *= inv
            lik = M[i, k]
            for j in range(k + 1, n):
                M[i, j] -= lik * M[k, j]
    return M, piv


def lu_solve_from_factor(M, piv, B):
    """Solve ``A X = B`` from an :func:`lu_factor` result. ``B`` is ``(n, k)``."""
    n = M.shape[0]
    k = B.shape[1]
    X = B.astype(np.float64)
    for c in range(k):
        # apply the recorded row interchanges to the right-hand side
        for i in range(n):
            p = piv[i]
            if p != i:
                tmp = X[i, c]; X[i, c] = X[p, c]; X[p, c] = tmp
        # forward substitution, Ly = Pb (L is unit lower triangular)
        for i in range(1, n):
            s = X[i, c]
            for j in range(i):
                s -= M[i, j] * X[j, c]
            X[i, c] = s
        # back substitution, Ux = y
        for i in range(n - 1, -1, -1):
            s = X[i, c]
            for j in range(i + 1, n):
                s -= M[i, j] * X[j, c]
            X[i, c] = s / M[i, i]
    return X


def cholesky_factor(A):
    """LL^T factorisation of a symmetric positive-definite ``A``; returns L."""
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
                L[i, j] = sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def cholesky_solve_from_factor(L, B):
    """Solve ``A X = B`` from a :func:`cholesky_factor` result. ``B`` is ``(n, k)``."""
    n = L.shape[0]
    k = B.shape[1]
    X = B.astype(np.float64)
    for c in range(k):
        # forward substitution: Ly = b
        for i in range(n):
            s = X[i, c]
            for j in range(i):
                s -= L[i, j] * X[j, c]
            X[i, c] = s / L[i, i]
        # back substitution: L^T x = y
        for i in range(n - 1, -1, -1):
            s = X[i, c]
            for j in range(i + 1, n):
                s -= L[j, i] * X[j, c]
            X[i, c] = s / L[i, i]
    return X


# ---------------------------------------------------------------------------
# Iterative methods
#
# ``thr`` is the absolute residual threshold the caller derived from
# rtol/atol. Each solver checks the initial guess before iterating, so an exact
# x0 converges in 0 iterations and max_iter=0 is well defined.
# ---------------------------------------------------------------------------
def jacobi(A, b, x0, thr, max_iter):
    n = A.shape[0]
    x = x0.astype(np.float64)
    x_new = np.zeros(n, dtype=np.float64)

    r = 0.0
    for i in range(n):
        s = b[i]
        for j in range(n):
            s -= A[i, j] * x[j]
        r += s * s
    r = sqrt(r)
    if r <= thr:
        return x, 0, r, True

    for it in range(1, max_iter + 1):
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i, j] * x[j]
            x_new[i] = s / A[i, i]
        for i in range(n):
            x[i] = x_new[i]
        r = 0.0
        for i in range(n):
            s = b[i]
            for j in range(n):
                s -= A[i, j] * x[j]
            r += s * s
        r = sqrt(r)
        if r <= thr:
            return x, it, r, True
    return x, max_iter, r, False


def gauss_seidel(A, b, x0, thr, max_iter):
    n = A.shape[0]
    x = x0.astype(np.float64)

    r = 0.0
    for i in range(n):
        s = b[i]
        for j in range(n):
            s -= A[i, j] * x[j]
        r += s * s
    r = sqrt(r)
    if r <= thr:
        return x, 0, r, True

    for it in range(1, max_iter + 1):
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i, j] * x[j]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = s / A[i, i]
        r = 0.0
        for i in range(n):
            s = b[i]
            for j in range(n):
                s -= A[i, j] * x[j]
            r += s * s
        r = sqrt(r)
        if r <= thr:
            return x, it, r, True
    return x, max_iter, r, False


def sor(A, b, omega, x0, thr, max_iter):
    n = A.shape[0]
    x = x0.astype(np.float64)

    r = 0.0
    for i in range(n):
        s = b[i]
        for j in range(n):
            s -= A[i, j] * x[j]
        r += s * s
    r = sqrt(r)
    if r <= thr:
        return x, 0, r, True

    for it in range(1, max_iter + 1):
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i, j] * x[j]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = x[i] + omega * (s / A[i, i] - x[i])
        r = 0.0
        for i in range(n):
            s = b[i]
            for j in range(n):
                s -= A[i, j] * x[j]
            r += s * s
        r = sqrt(r)
        if r <= thr:
            return x, it, r, True
    return x, max_iter, r, False


# ---------------------------------------------------------------------------
# The numba twins: the same source, compiled on first call.
# ---------------------------------------------------------------------------
nb = SimpleNamespace(
    gauss_solve=_jit(gauss_solve),
    lu_factor=_jit(lu_factor),
    lu_solve_from_factor=_jit(lu_solve_from_factor),
    cholesky_factor=_jit(cholesky_factor),
    cholesky_solve_from_factor=_jit(cholesky_solve_from_factor),
    jacobi=_jit(jacobi),
    gauss_seidel=_jit(gauss_seidel),
    sor=_jit(sor),
)

__all__ = ["nb", "_HAVE_NUMBA"]

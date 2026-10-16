# cython: language_level=3
"""Cython statically-typed core routines.

These mirror :mod:`linsolve._kernels` function for function, but use C-level
typed loops (``ctypedef double``, contiguous typed buffers) so the compiler
emits tight native loops with no Python object overhead inside the kernel.
The function names deliberately match the plain-Python kernels so the
dispatcher can select a backend with a single lookup.
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt

np.import_array()

ctypedef np.float64_t D

# eps(float64)
cdef double _EPS = 2.220446049250313e-16


# ---------------------------------------------------------------------------
# Direct methods
# ---------------------------------------------------------------------------
def gauss_solve(np.ndarray[D, ndim=2, mode="c"] A not None,
                np.ndarray[D, ndim=1, mode="c"] b not None):
    cdef:
        Py_ssize_t n = A.shape[0]
        Py_ssize_t i, j, k, pivot
        double factor, aik_inv, best, v, s, tmp, amax = 0.0, sing_tol

    cdef np.ndarray[D, ndim=2, mode="c"] M = A.copy()
    cdef np.ndarray[D, ndim=1, mode="c"] y = b.copy()
    cdef np.ndarray[D, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)

    for i in range(n):
        for j in range(n):
            v = fabs(M[i, j])
            if v > amax:
                amax = v
    sing_tol = _EPS * amax * n

    for k in range(n):
        pivot = k
        best = fabs(M[k, k])
        for i in range(k + 1, n):
            v = fabs(M[i, k])
            if v > best:
                best = v; pivot = i
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

    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x


def lu_factor(np.ndarray[D, ndim=2, mode="c"] A not None):
    """PA = LU with partial pivoting; returns (LU, piv)."""
    cdef:
        Py_ssize_t n = A.shape[0]
        Py_ssize_t i, j, k, pivot
        double lik, inv, best, v, tmp, amax = 0.0, sing_tol

    cdef np.ndarray[D, ndim=2, mode="c"] M = A.copy()
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] piv = np.zeros(n, dtype=np.int64)

    for i in range(n):
        for j in range(n):
            v = fabs(M[i, j])
            if v > amax:
                amax = v
    sing_tol = _EPS * amax * n

    for k in range(n):
        pivot = k
        best = fabs(M[k, k])
        for i in range(k + 1, n):
            v = fabs(M[i, k])
            if v > best:
                best = v; pivot = i
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


def lu_solve_from_factor(np.ndarray[D, ndim=2, mode="c"] M not None,
                         np.ndarray[np.int64_t, ndim=1, mode="c"] piv not None,
                         np.ndarray[D, ndim=2, mode="c"] B not None):
    """Solve A X = B (B is (n, k)) from an lu_factor result."""
    cdef:
        Py_ssize_t n = M.shape[0], k = B.shape[1]
        Py_ssize_t i, j, c, p
        double s, tmp

    cdef np.ndarray[D, ndim=2, mode="c"] X = B.copy()

    for c in range(k):
        for i in range(n):
            p = piv[i]
            if p != i:
                tmp = X[i, c]; X[i, c] = X[p, c]; X[p, c] = tmp
        for i in range(1, n):
            s = X[i, c]
            for j in range(i):
                s -= M[i, j] * X[j, c]
            X[i, c] = s
        for i in range(n - 1, -1, -1):
            s = X[i, c]
            for j in range(i + 1, n):
                s -= M[i, j] * X[j, c]
            X[i, c] = s / M[i, i]
    return X


def cholesky_factor(np.ndarray[D, ndim=2, mode="c"] A not None):
    """LL^T factorisation of a symmetric positive-definite A; returns L."""
    cdef:
        Py_ssize_t n = A.shape[0]
        Py_ssize_t i, j, k
        double s
    cdef np.ndarray[D, ndim=2, mode="c"] L = np.zeros((n, n), dtype=np.float64)

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


def cholesky_solve_from_factor(np.ndarray[D, ndim=2, mode="c"] L not None,
                               np.ndarray[D, ndim=2, mode="c"] B not None):
    """Solve A X = B (B is (n, k)) from a cholesky_factor result."""
    cdef:
        Py_ssize_t n = L.shape[0], k = B.shape[1]
        Py_ssize_t i, j, c
        double s

    cdef np.ndarray[D, ndim=2, mode="c"] X = B.copy()

    for c in range(k):
        for i in range(n):
            s = X[i, c]
            for j in range(i):
                s -= L[i, j] * X[j, c]
            X[i, c] = s / L[i, i]
        for i in range(n - 1, -1, -1):
            s = X[i, c]
            for j in range(i + 1, n):
                s -= L[j, i] * X[j, c]
            X[i, c] = s / L[i, i]
    return X


# ---------------------------------------------------------------------------
# Iterative methods
# ---------------------------------------------------------------------------
cdef double _resid(np.ndarray[D, ndim=2, mode="c"] A,
                   np.ndarray[D, ndim=1, mode="c"] b,
                   np.ndarray[D, ndim=1, mode="c"] x) except? -1.0:
    cdef Py_ssize_t n = A.shape[0], i, j
    cdef double s, acc = 0.0
    for i in range(n):
        s = b[i]
        for j in range(n):
            s -= A[i, j] * x[j]
        acc += s * s
    return sqrt(acc)


def jacobi(np.ndarray[D, ndim=2, mode="c"] A not None,
           np.ndarray[D, ndim=1, mode="c"] b not None,
           np.ndarray[D, ndim=1, mode="c"] x0 not None,
           double thr, long max_iter):
    cdef Py_ssize_t n = A.shape[0], i, j, it
    cdef double s, r
    cdef np.ndarray[D, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[D, ndim=1, mode="c"] x_new = np.zeros(n, dtype=np.float64)

    r = _resid(A, b, x)
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
        r = _resid(A, b, x)
        if r <= thr:
            return x, it, r, True
    return x, max_iter, r, False


def gauss_seidel(np.ndarray[D, ndim=2, mode="c"] A not None,
                 np.ndarray[D, ndim=1, mode="c"] b not None,
                 np.ndarray[D, ndim=1, mode="c"] x0 not None,
                 double thr, long max_iter):
    cdef Py_ssize_t n = A.shape[0], i, j, it
    cdef double s, r
    cdef np.ndarray[D, ndim=1, mode="c"] x = x0.copy()

    r = _resid(A, b, x)
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
        r = _resid(A, b, x)
        if r <= thr:
            return x, it, r, True
    return x, max_iter, r, False


def sor(np.ndarray[D, ndim=2, mode="c"] A not None,
        np.ndarray[D, ndim=1, mode="c"] b not None,
        double omega,
        np.ndarray[D, ndim=1, mode="c"] x0 not None,
        double thr, long max_iter):
    cdef Py_ssize_t n = A.shape[0], i, j, it
    cdef double s, r
    cdef np.ndarray[D, ndim=1, mode="c"] x = x0.copy()

    r = _resid(A, b, x)
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
        r = _resid(A, b, x)
        if r <= thr:
            return x, it, r, True
    return x, max_iter, r, False

# cython: language_level=3
"""Cython statically-typed core routines.

These mirror ``_numba.py`` but use C-level typed loops (``ctypedef double``,
contiguous ``double[:,:]`` / ``double[:]`` memoryviews) so the compiler can
emit tight native loops with no Python object overhead inside the kernel.
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt

np.import_array()

ctypedef np.float64_t D


# ---------------------------------------------------------------------------
# Direct methods
# ---------------------------------------------------------------------------
def gauss_solve_cy(np.ndarray[D, ndim=2, mode="c"] A not None,
                   np.ndarray[D, ndim=1, mode="c"] b not None):
    cdef:
        Py_ssize_t n = A.shape[0]
        Py_ssize_t i, j, k, pivot
        double factor, aik_inv, best, v, s, tmp

    # copy inputs (caller may reuse them)
    cdef np.ndarray[D, ndim=2, mode="c"] M = A.copy()
    cdef np.ndarray[D, ndim=1, mode="c"] y = b.copy()
    cdef np.ndarray[D, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)

    for k in range(n):
        pivot = k
        best = fabs(M[k, k])
        for i in range(k + 1, n):
            v = fabs(M[i, k])
            if v > best:
                best = v; pivot = i
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

    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x


def lu_solve_cy(np.ndarray[D, ndim=2, mode="c"] A not None,
                np.ndarray[D, ndim=1, mode="c"] b not None):
    cdef:
        Py_ssize_t n = A.shape[0]
        Py_ssize_t i, j, k, pivot
        double lik, inv, best, v, s, tmp

    cdef np.ndarray[D, ndim=2, mode="c"] M = A.copy()
    cdef np.ndarray[D, ndim=1, mode="c"] y = b.copy()
    cdef np.ndarray[D, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)

    for k in range(n):
        pivot = k
        best = fabs(M[k, k])
        for i in range(k + 1, n):
            v = fabs(M[i, k])
            if v > best:
                best = v; pivot = i
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

    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x


def cholesky_solve_cy(np.ndarray[D, ndim=2, mode="c"] A not None,
                      np.ndarray[D, ndim=1, mode="c"] b not None):
    cdef:
        Py_ssize_t n = A.shape[0]
        Py_ssize_t i, j, k
        double s
    cdef np.ndarray[D, ndim=2, mode="c"] L = np.zeros((n, n), dtype=np.float64)
    cdef np.ndarray[D, ndim=1, mode="c"] y = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[D, ndim=1, mode="c"] x = np.zeros(n, dtype=np.float64)

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

    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s / L[i, i]

    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= L[j, i] * x[j]
        x[i] = s / L[i, i]
    return x


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


def jacobi_cy(np.ndarray[D, ndim=2, mode="c"] A not None,
              np.ndarray[D, ndim=1, mode="c"] b not None,
              np.ndarray[D, ndim=1, mode="c"] x0 not None,
              double tol, long max_iter):
    cdef Py_ssize_t n = A.shape[0], i, j, it
    cdef double s, r
    cdef np.ndarray[D, ndim=1, mode="c"] x = x0.copy()
    cdef np.ndarray[D, ndim=1, mode="c"] x_new = np.zeros(n, dtype=np.float64)
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
        if r < tol:
            return x, it, r, True
    return x, max_iter, _resid(A, b, x), False


def gauss_seidel_cy(np.ndarray[D, ndim=2, mode="c"] A not None,
                    np.ndarray[D, ndim=1, mode="c"] b not None,
                    np.ndarray[D, ndim=1, mode="c"] x0 not None,
                    double tol, long max_iter):
    cdef Py_ssize_t n = A.shape[0], i, j, it
    cdef double s, r
    cdef np.ndarray[D, ndim=1, mode="c"] x = x0.copy()
    for it in range(1, max_iter + 1):
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i, j] * x[j]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = s / A[i, i]
        r = _resid(A, b, x)
        if r < tol:
            return x, it, r, True
    return x, max_iter, _resid(A, b, x), False


def sor_cy(np.ndarray[D, ndim=2, mode="c"] A not None,
           np.ndarray[D, ndim=1, mode="c"] b not None,
           double omega,
           np.ndarray[D, ndim=1, mode="c"] x0 not None,
           double tol, long max_iter):
    cdef Py_ssize_t n = A.shape[0], i, j, it
    cdef double s, r
    cdef np.ndarray[D, ndim=1, mode="c"] x = x0.copy()
    for it in range(1, max_iter + 1):
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i, j] * x[j]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = x[i] + omega * (s / A[i, i] - x[i])
        r = _resid(A, b, x)
        if r < tol:
            return x, it, r, True
    return x, max_iter, _resid(A, b, x), False

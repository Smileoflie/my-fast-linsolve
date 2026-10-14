"""linsolve: high-performance linear-system solvers in Python.

Six algorithms are provided, each available on three backends
(``cython`` -> ``numba`` -> ``python``, auto-selected):

Direct methods
    - :func:`gauss_solve`     Gaussian elimination with partial pivoting
    - :func:`lu_solve`       LU factorisation (PA = LU, Doolittle)
    - :func:`cholesky_solve` Cholesky factorisation (symmetric positive-definite)

Iterative methods
    - :func:`jacobi`         Jacobi iteration
    - :func:`gauss_seidel`   Gauss-Seidel iteration
    - :func:`sor`            Successive Over-Relaxation

Quick start
-----------
>>> import numpy as np
>>> from linsolve import solve
>>> A = np.array([[4, 1, 2], [0, 5, 1], [1, 0, 3]], dtype=float)
>>> b = np.array([7, 7, 4], dtype=float)
>>> x = solve(A, b, method="lu")
>>> np.allclose(A @ x, b)
True
"""
from __future__ import annotations

import numpy as np

from ._backend import ACTIVE_BACKEND, Backend, detect_backend
from .direct import cholesky_solve, gauss_solve, lu_solve
from .iterative import gauss_seidel, jacobi, sor

__version__ = "0.1.1"

__all__ = [
    "solve",
    "gauss_solve", "lu_solve", "cholesky_solve",
    "jacobi", "gauss_seidel", "sor",
    "ACTIVE_BACKEND", "Backend", "detect_backend",
    "__version__",
]


_DIRECT = {"gauss": gauss_solve, "lu": lu_solve, "cholesky": cholesky_solve}


def solve(A, b, method: str = "lu", backend: str = "auto", **kwargs):
    """Unified entry point for ``A x = b``.

    Parameters
    ----------
    method : str
        One of ``"gauss"``, ``"lu"``, ``"cholesky"`` (direct),
        ``"jacobi"``, ``"gauss_seidel"``, ``"sor"`` (iterative).
    backend : str
        ``"auto"`` (default), ``"cython"``, ``"numba"`` or ``"python"``.
    """
    m = method.lower()
    if m in _DIRECT:
        return _DIRECT[m](A, b, backend=backend)
    if m == "jacobi":
        return jacobi(A, b, backend=backend, **kwargs)
    if m == "gauss_seidel":
        return gauss_seidel(A, b, backend=backend, **kwargs)
    if m == "sor":
        return sor(A, b, backend=backend, **kwargs)
    raise ValueError(
        f"Unknown method {method!r}; choose from "
        f"{list(_DIRECT) + ['jacobi','gauss_seidel','sor']}."
    )

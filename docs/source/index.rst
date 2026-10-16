Welcome to linsolve's documentation!
=====================================

``linsolve`` is a high-performance linear-system solver library. It ships
three direct methods and three iterative methods, each available on three
backends (Cython / Numba / pure Python) that are selected automatically at
runtime.

Installation
------------

.. code-block:: bash

   pip install fast-linsolve
   pip install fast-linsolve[numba,cython]

Quick start
-----------

.. code-block:: python

   import numpy as np
   from linsolve import solve

   A = np.array([[4, 1, 2], [0, 5, 1], [1, 0, 3]], dtype=float)
   b = np.array([7, 7, 4], dtype=float)
   x = solve(A, b, method="lu")

Direct methods
--------------

.. autofunction:: linsolve.gauss_solve
.. autofunction:: linsolve.lu_solve
.. autofunction:: linsolve.cholesky_solve

Factorisations (reuse across right-hand sides)
----------------------------------------------

The O(n^3) factorisation and the O(n^2) triangular solves are separable, so a
single factorisation can serve any number of right-hand sides::

   from linsolve import lu_factor, lu_solve_factorized

   LU, piv = lu_factor(A)               # once
   x = lu_solve_factorized(LU, piv, b)  # every time after that

.. autofunction:: linsolve.lu_factor
.. autofunction:: linsolve.lu_solve_factorized
.. autofunction:: linsolve.cholesky_factor
.. autofunction:: linsolve.cholesky_solve_factorized

Iterative methods
-----------------

Convergence is tested as ``||b - A x|| <= atol + rtol * ||b||``;
``info["residual"]`` is the relative residual and ``info["residual_abs"]`` the
absolute one.

.. autofunction:: linsolve.jacobi
.. autofunction:: linsolve.gauss_seidel
.. autofunction:: linsolve.sor

Unified entry point
-------------------

.. autofunction:: linsolve.solve

Backend selection
-----------------

``linsolve.ACTIVE_BACKEND`` reports the backend chosen at import time;
:func:`linsolve.backend_info` explains why.

.. autofunction:: linsolve.backend_info
.. autofunction:: linsolve.detect_backend

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

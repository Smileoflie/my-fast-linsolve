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

Iterative methods
-----------------

.. autofunction:: linsolve.jacobi
.. autofunction:: linsolve.gauss_seidel
.. autofunction:: linsolve.sor

Unified entry point
-------------------

.. autofunction:: linsolve.solve

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

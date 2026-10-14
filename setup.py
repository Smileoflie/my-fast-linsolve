"""Conditional Cython build.

If Cython and a C compiler are available, the ``_cython_core`` extension is
compiled from ``src/linsolve/_cython_core.pyx``. Otherwise the package is built
as pure Python and the runtime layer falls back to Numba / NumPy automatically.
"""
import os
from setuptools import setup, Extension

CYTHON_SRC = os.path.join("src", "linsolve", "_cython_core.pyx")
ext_modules = []

try:
    from Cython.Build import cythonize
    import numpy as np

    if os.path.exists(CYTHON_SRC):
        ext_modules = cythonize(
            [
                Extension(
                    "linsolve._cython_core",
                    [CYTHON_SRC],
                    include_dirs=[np.get_include()],
                )
            ],
            compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False},
        )
except ImportError:
    # Cython not present at build time -> ship pure Python only.
    pass

setup(ext_modules=ext_modules)

"""Tests for direct solvers: Gauss, LU, Cholesky."""
import numpy as np
import pytest

from linsolve import gauss_solve, lu_solve, cholesky_solve
from linsolve._backend import Backend


BACKENDS = ["python", "numba"]  # cython only if compiled in this env
try:
    from linsolve import _cython_core  # noqa: F401
    BACKENDS.append("cython")
except Exception:
    pass


@pytest.mark.parametrize("backend", BACKENDS)
def test_gauss_matches_numpy(small_dense, backend):
    A, b, x_ref = small_dense
    x = gauss_solve(A, b, backend=backend)
    assert np.allclose(x, x_ref, atol=1e-8)
    assert np.allclose(A @ x, b, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lu_matches_numpy(small_dense, backend):
    A, b, x_ref = small_dense
    x = lu_solve(A, b, backend=backend)
    assert np.allclose(x, x_ref, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cholesky_on_spd(spd_matrix, backend):
    A, b, x_ref = spd_matrix
    x = cholesky_solve(A, b, backend=backend)
    assert np.allclose(x, x_ref, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_direct_methods_agree(small_dense, backend):
    A, b, _ = small_dense
    xg = gauss_solve(A, b, backend=backend)
    xl = lu_solve(A, b, backend=backend)
    assert np.allclose(xg, xl, atol=1e-9)


def test_cholesky_rejects_non_spd():
    # symmetric but not positive-definite: eigenvalues 1 and -1
    A = np.array([[1.0, 0.0], [0.0, -1.0]])
    b = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        cholesky_solve(A, b)


def test_singular_matrix_raises(rng):
    A = np.ones((3, 3))  # rank 1
    b = np.ones(3)
    with pytest.raises(ValueError):
        gauss_solve(A, b)
    with pytest.raises(ValueError):
        lu_solve(A, b)


def test_input_validation():
    with pytest.raises(ValueError):
        gauss_solve(np.zeros((3, 4)), np.zeros(3))
    with pytest.raises(ValueError):
        gauss_solve(np.zeros((3, 3)), np.zeros(4))


def test_backend_resolution():
    from linsolve._backend import resolve
    assert resolve("auto") in (Backend.CYTHON, Backend.NUMBA, Backend.PYTHON)
    with pytest.raises(ValueError):
        resolve("not_a_backend")

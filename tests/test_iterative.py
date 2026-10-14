"""Tests for iterative solvers: Jacobi, Gauss-Seidel, SOR."""
import numpy as np
import pytest

from linsolve import jacobi, gauss_seidel, sor


BACKENDS = ["python", "numba"]
try:
    from linsolve import _cython_core  # noqa: F401
    BACKENDS.append("cython")
except Exception:
    pass


@pytest.mark.parametrize("backend", BACKENDS)
def test_jacobi_converges(diag_dominant, backend):
    A, b, x_ref = diag_dominant
    x, info = jacobi(A, b, tol=1e-8, max_iter=5000, backend=backend)
    assert info["converged"] is True
    assert np.allclose(x, x_ref, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gauss_seidel_converges(diag_dominant, backend):
    A, b, x_ref = diag_dominant
    x, info = gauss_seidel(A, b, tol=1e-8, max_iter=5000, backend=backend)
    assert info["converged"] is True
    assert np.allclose(x, x_ref, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sor_converges(diag_dominant, backend):
    A, b, x_ref = diag_dominant
    x, info = sor(A, b, omega=1.2, tol=1e-8, max_iter=5000, backend=backend)
    assert info["converged"] is True
    assert np.allclose(x, x_ref, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sor_omega_one_matches_gauss_seidel(diag_dominant, backend):
    """omega = 1 should make SOR identical to Gauss-Seidel."""
    A, b, _ = diag_dominant
    xs, _ = sor(A, b, omega=1.0, tol=1e-10, max_iter=200, backend=backend)
    xg, _ = gauss_seidel(A, b, tol=1e-10, max_iter=200, backend=backend)
    assert np.allclose(xs, xg, atol=1e-10)


def test_bad_omega_raises():
    A = np.eye(3)
    b = np.ones(3)
    with pytest.raises(ValueError):
        sor(A, b, omega=0.0)
    with pytest.raises(ValueError):
        sor(A, b, omega=2.0)
    with pytest.raises(ValueError):
        sor(A, b, omega=3.5)


def test_max_iter_returns_nonconverged_flag(rng):
    # A matrix where Jacobi on a non-convergent problem does not reach tol.
    A = np.array([[1.0, 2.0], [3.0, 4.0]])  # not diagonally dominant
    b = np.array([1.0, 1.0])
    x, info = jacobi(A, b, tol=1e-12, max_iter=5)
    assert info["converged"] is False
    assert info["iterations"] == 5


def test_info_dict_shape(diag_dominant):
    A, b, _ = diag_dominant
    x, info = jacobi(A, b)
    assert set(info.keys()) == {"iterations", "residual", "converged"}
    assert isinstance(info["iterations"], int)
    assert isinstance(info["residual"], float)
    assert isinstance(info["converged"], bool)

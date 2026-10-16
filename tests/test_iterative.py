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
    x, info = jacobi(A, b, rtol=1e-8, max_iter=5000, backend=backend)
    assert info["converged"] is True
    assert np.allclose(x, x_ref, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gauss_seidel_converges(diag_dominant, backend):
    A, b, x_ref = diag_dominant
    x, info = gauss_seidel(A, b, rtol=1e-8, max_iter=5000, backend=backend)
    assert info["converged"] is True
    assert np.allclose(x, x_ref, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sor_converges(diag_dominant, backend):
    A, b, x_ref = diag_dominant
    x, info = sor(A, b, omega=1.2, rtol=1e-8, max_iter=5000, backend=backend)
    assert info["converged"] is True
    assert np.allclose(x, x_ref, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sor_omega_one_matches_gauss_seidel(diag_dominant, backend):
    """omega = 1 should make SOR identical to Gauss-Seidel."""
    A, b, _ = diag_dominant
    xs, _ = sor(A, b, omega=1.0, rtol=1e-10, max_iter=200, backend=backend)
    xg, _ = gauss_seidel(A, b, rtol=1e-10, max_iter=200, backend=backend)
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
    x, info = jacobi(A, b, rtol=1e-12, max_iter=5)
    assert info["converged"] is False
    assert info["iterations"] == 5


def test_info_dict_shape(diag_dominant):
    A, b, _ = diag_dominant
    x, info = jacobi(A, b)
    assert set(info.keys()) == {"iterations", "residual", "residual_abs", "converged"}
    assert isinstance(info["iterations"], int)
    assert isinstance(info["residual"], float)
    assert isinstance(info["residual_abs"], float)
    assert isinstance(info["converged"], bool)


# ---------------------------------------------------------------------------
# Regression tests for the 0.2.0 robustness fixes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_max_iter_zero_returns_initial_guess(diag_dominant, backend):
    """max_iter=0 must not crash and must report the x0 residual."""
    A, b, _ = diag_dominant
    for fn in (jacobi, gauss_seidel, sor):
        x, info = fn(A, b, max_iter=0, backend=backend)
        assert info["iterations"] == 0
        assert info["converged"] is False
        assert np.allclose(x, 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_exact_x0_converges_immediately(diag_dominant, backend):
    """An exact initial guess converges in 0 iterations."""
    A, b, x_ref = diag_dominant
    x, info = gauss_seidel(A, b, x0=x_ref, backend=backend)
    assert info["iterations"] == 0
    assert info["converged"] is True
    assert np.allclose(x, x_ref)


def test_zero_diagonal_raises():
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    b = np.ones(2)
    for fn in (jacobi, gauss_seidel, sor):
        with pytest.raises(ValueError, match="zero diagonal"):
            fn(A, b)


def test_convergence_is_scale_invariant(rng):
    """With a relative tolerance, iteration count must not depend on the
    magnitude of b (previously an absolute tol let tiny systems 'converge'
    with completely wrong answers)."""
    n = 20
    A = rng.standard_normal((n, n)) * 0.1
    np.fill_diagonal(A, np.abs(A).sum(axis=1) + 1.0)
    x_ref = rng.standard_normal(n)
    counts = []
    for scale in (1e-6, 1.0, 1e6):
        b = scale * (A @ x_ref)
        x, info = jacobi(A, b, rtol=1e-6, max_iter=10000)
        assert info["converged"] is True
        assert np.allclose(x / scale, x_ref, rtol=1e-4)
        counts.append(info["iterations"])
    assert max(counts) - min(counts) <= 2, f"iteration counts diverge: {counts}"


def test_relative_residual_reported(diag_dominant):
    A, b, _ = diag_dominant
    x, info = jacobi(A, b, rtol=1e-6, max_iter=5000)
    expected_rel = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    assert np.isclose(info["residual"], expected_rel, rtol=1e-12)
    assert info["residual"] <= 1e-6  # relative tolerance actually enforced


def test_tol_kwarg_backward_compatible(diag_dominant):
    """The deprecated absolute `tol` still works and warns."""
    A, b, _ = diag_dominant
    with pytest.warns(DeprecationWarning):
        x, info = jacobi(A, b, tol=1e-8, max_iter=5000)
    assert info["converged"] is True


def test_b_must_be_1d():
    with pytest.raises(ValueError, match="1-D"):
        jacobi(np.eye(2), np.ones((2, 1)))


def test_x0_wrong_length_raises():
    with pytest.raises(ValueError):
        jacobi(np.eye(2), np.ones(2), x0=np.ones(5))

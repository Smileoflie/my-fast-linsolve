"""Tests for direct solvers: Gauss, LU, Cholesky."""
import numpy as np
import pytest

from linsolve import (
    cholesky_factor,
    cholesky_solve,
    cholesky_solve_factorized,
    gauss_solve,
    lu_factor,
    lu_solve,
    lu_solve_factorized,
)
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


# ---------------------------------------------------------------------------
# Regression tests for the 0.2.0 robustness fixes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_cholesky_rejects_nonsymmetric(backend):
    """A non-symmetric matrix must not silently produce a wrong answer."""
    A = np.array([[4.0, 100.0], [1.0, 3.0]])
    with pytest.raises(ValueError, match="symmetric"):
        cholesky_solve(A, np.ones(2), backend=backend)


@pytest.mark.parametrize("scale", [1e-20, 1e-16, 1e-8, 1e6])
def test_scaled_matrix_not_misjudged_singular(scale):
    """A well-conditioned matrix stays solvable at any overall scale."""
    A = scale * np.array([[2.0, 1.0], [1.0, 3.0]])
    b = scale * np.array([1.0, 1.0])
    for fn in (gauss_solve, lu_solve):
        x = fn(A, b)
        assert np.allclose(x, [0.4, 0.2], atol=1e-10)


def test_b_must_be_1d():
    with pytest.raises(ValueError, match="1-D"):
        lu_solve(np.eye(2), np.ones((2, 1)))


def test_solve_rejects_kwargs_for_direct_methods():
    from linsolve import solve
    with pytest.raises(TypeError, match="direct solver"):
        solve(np.eye(2), np.ones(2), method="lu", tol=1e-10)


def test_backend_info_reports_all_backends():
    from linsolve import backend_info
    info = backend_info()
    assert info["active"] in ("cython", "numba", "python")
    assert info["python"] == (True, "pure-Python reference, always available")
    assert set(("cython", "numba", "python")) <= set(info)


def test_backend_info_numba_availability_is_truthful():
    """A stale import must not report an installed numba as unavailable:
    that would silently fall back to the slow pure-Python backend."""
    import importlib.util

    from linsolve import backend_info
    have_numba = importlib.util.find_spec("numba") is not None
    assert backend_info()["numba"][0] is have_numba, backend_info()["numba"]


# ---------------------------------------------------------------------------
# Factorisation reuse API
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_lu_factor_reconstructs_pa_eq_lu(rng, backend):
    """The documented contract is P A = L U; verify it directly."""
    A = rng.standard_normal((6, 6))
    A = np.tril(A) + np.tril(A, -1).T + 6 * np.eye(6)  # forces row swaps
    LU, piv = lu_factor(A, backend=backend)
    n = A.shape[0]
    L = np.tril(LU, -1) + np.eye(n)
    U = np.triu(LU)
    P = np.eye(n)
    for k in range(n):
        P[[k, piv[k]]] = P[[piv[k], k]]
    assert np.allclose(P @ A, L @ U, atol=1e-9)
    assert np.allclose(L @ U, P @ A, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lu_solve_factorized_matches_one_shot(small_dense, backend):
    A, b, x_ref = small_dense
    LU, piv = lu_factor(A, backend=backend)
    x = lu_solve_factorized(LU, piv, b, backend=backend)
    assert np.allclose(x, x_ref, atol=1e-8)
    assert np.allclose(x, lu_solve(A, b, backend=backend), atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lu_factorized_multiple_rhs(rng, backend):
    A = rng.standard_normal((7, 7)) + 7 * np.eye(7)
    B = rng.standard_normal((7, 4))
    LU, piv = lu_factor(A, backend=backend)
    X = lu_solve_factorized(LU, piv, B, backend=backend)
    assert X.shape == (7, 4)
    assert np.allclose(A @ X, B, atol=1e-8)
    # a strided view must work too (regression: C-contiguity in compiled backends)
    x0 = lu_solve_factorized(LU, piv, B[:, 0], backend=backend)
    assert x0.shape == (7,)
    assert np.allclose(x0, X[:, 0], atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cholesky_factor_reconstructs(spd_matrix, backend):
    A, b, x_ref = spd_matrix
    L = cholesky_factor(A, backend=backend)
    assert np.allclose(L @ L.T, A, atol=1e-8)
    assert np.allclose(np.triu(L, 1), 0.0)  # L really is lower triangular
    x = cholesky_solve_factorized(L, b, backend=backend)
    assert np.allclose(x, x_ref, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cholesky_factor_rejects_non_spd_and_asymmetric(backend):
    with pytest.raises(ValueError, match="symmetric"):
        cholesky_factor(np.array([[4.0, 100.0], [1.0, 3.0]]), backend=backend)
    with pytest.raises(ValueError, match="positive definite"):
        cholesky_factor(np.array([[1.0, 0.0], [0.0, -1.0]]), backend=backend)


def test_lu_factor_singular_raises():
    with pytest.raises(ValueError, match="singular"):
        lu_factor(np.ones((3, 3)))


def test_factor_input_validation():
    with pytest.raises(ValueError):
        lu_factor(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        lu_solve_factorized(np.eye(2), np.array([0, 0]), np.ones(3))
    with pytest.raises(ValueError, match="1-D vector or a 2-D"):
        lu_solve_factorized(np.eye(2), np.array([0, 0]), np.ones((2, 2, 2)))


def test_factor_backends_agree(rng):
    """The three backends must produce interchangeable factorisations."""
    A = rng.standard_normal((9, 9))
    A = np.tril(A) + np.tril(A, -1).T + 9 * np.eye(9)
    b = rng.standard_normal(9)
    results = {}
    for be in BACKENDS:
        LU, piv = lu_factor(A, backend=be)
        results[be] = lu_solve_factorized(LU, piv, b, backend=be)
        assert np.array_equal(piv, lu_factor(A, backend=BACKENDS[0])[1])
    ref = results[BACKENDS[0]]
    for be, x in results.items():
        assert np.allclose(x, ref, atol=1e-10), f"{be} disagrees"

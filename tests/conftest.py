"""Shared pytest fixtures."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20260101)


@pytest.fixture
def small_dense(rng):
    """A small dense nonsingular matrix."""
    A = rng.standard_normal((5, 5))
    # make it well conditioned
    A = A @ A.T + 5 * np.eye(5)
    x = rng.standard_normal(5)
    b = A @ x
    return A, b, x


@pytest.fixture
def spd_matrix(rng):
    """Symmetric positive-definite matrix (ideal for Cholesky)."""
    n = 6
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    d = np.linspace(1.0, 10.0, n)  # eigenvalues in [1,10]
    A = Q @ np.diag(d) @ Q.T
    x = rng.standard_normal(n)
    b = A @ x
    return A, b, x


@pytest.fixture
def diag_dominant(rng):
    """Strictly diagonally dominant matrix (iterative methods converge)."""
    n = 8
    A = rng.standard_normal((n, n)) * 0.1
    np.fill_diagonal(A, np.abs(A).sum(axis=1) + 1.0)
    x = rng.standard_normal(n)
    b = A @ x
    return A, b, x

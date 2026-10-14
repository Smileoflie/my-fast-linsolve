"""Performance benchmark: pure Python vs Numba vs Cython vs NumPy.

Usage
-----
    python benchmarks/benchmark.py            # default sizes
    python benchmarks/benchmark.py --n 1000   # custom size

The benchmark warms up Numba first (JIT compilation is not timed), then
times each backend on a well-conditioned dense system.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# make src importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from linsolve import gauss_solve, lu_solve, cholesky_solve  # noqa: E402
from linsolve._backend import ACTIVE_BACKEND  # noqa: E402


def make_system(n: int, spd: bool = False, seed: int = 0):
    rng = np.random.default_rng(seed)
    if spd:
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        A = Q @ np.diag(np.linspace(1.0, 10.0, n)) @ Q.T
    else:
        A = rng.standard_normal((n, n))
        A = A @ A.T + n * np.eye(n)  # well conditioned
    b = rng.standard_normal(n)
    return A, b


def timeit(fn, repeat: int = 3):
    # warmup
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="matrix size")
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    n = args.n

    print(f"Active backend: {ACTIVE_BACKEND.value}")
    print(f"Matrix size: {n} x {n}")
    print()

    # ---- direct: LU solve on dense ----------------------------------------
    A, b = make_system(n, spd=False)
    # warmup numba compilation
    lu_solve(A, b, backend="numba")
    try:
        from linsolve import _cython_core  # noqa: F401
        cy_avail = True
    except Exception:
        cy_avail = False

    rows = []
    t_py = timeit(lambda: lu_solve(A, b, backend="python"), repeat=1 if n > 300 else args.repeat)
    rows.append(("LU pure Python", t_py))
    t_nb = timeit(lambda: lu_solve(A, b, backend="numba"), repeat=args.repeat)
    rows.append(("LU Numba JIT", t_nb))
    if cy_avail:
        t_cy = timeit(lambda: lu_solve(A, b, backend="cython"), repeat=args.repeat)
        rows.append(("LU Cython", t_cy))
    t_np = timeit(lambda: np.linalg.solve(A, b), repeat=args.repeat)
    rows.append(("LU numpy.linalg.solve", t_np))

    # ---- Cholesky on SPD --------------------------------------------------
    Aspd, bspd = make_system(n, spd=True)
    cholesky_solve(Aspd, bspd, backend="numba")  # warmup
    t_py_c = timeit(lambda: cholesky_solve(Aspd, bspd, backend="python"),
                    repeat=1 if n > 300 else args.repeat)
    t_nb_c = timeit(lambda: cholesky_solve(Aspd, bspd, backend="numba"), repeat=args.repeat)
    rows.append(("Cholesky pure Python", t_py_c))
    rows.append(("Cholesky Numba JIT", t_nb_c))
    t_np_c = timeit(lambda: np.linalg.solve(Aspd, bspd), repeat=args.repeat)
    rows.append(("Cholesky numpy (ref)", t_np_c))

    # ---- print table ------------------------------------------------------
    print(f"{'method':<28}{'time (s)':>12}{'vs NumPy':>12}{'vs pure Py':>14}")
    print("-" * 66)
    baseline_py = dict(rows)["LU pure Python"]
    for name, t in rows:
        ratio_np = t_np / t if t > 0 else float("inf")
        ratio_py = baseline_py / t if t > 0 else float("inf")
        print(f"{name:<28}{t:>12.4f}{ratio_np:>11.2f}x{ratio_py:>13.1f}x")

    print()
    if t_nb > 0:
        print(f"Numba speedup over pure Python: {baseline_py / t_nb:,.0f}x")
    if cy_avail:
        print(f"Cython speedup over Numba:     {t_nb / t_cy:.2f}x")
    print(f"Numba LU as % of NumPy:        {t_np / t_nb * 100:.1f}%")


if __name__ == "__main__":
    main()

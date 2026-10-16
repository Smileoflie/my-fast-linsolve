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
import os
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
    p.add_argument("--skip-python", action="store_true",
                   help="skip the pure-Python baseline entirely")
    args = p.parse_args()
    n = args.n
    # The pure-Python backend is an interpreted O(n^3) loop: ~10 s at n=400 and
    # ~3 min at n=1000. Never let a default run sit there for minutes.
    py_limit = 400
    measure_py = (not args.skip_python) and n <= py_limit

    print(f"Active backend: {ACTIVE_BACKEND.value}")
    print(f"Matrix size: {n} x {n}")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        print(f"{var}={os.environ.get(var, '<unset>')}", end="  ")
    print("\n(hint: set those to 1 for a single-threaded apples-to-apples"
          " comparison against the Numba kernels)\n")

    try:
        from linsolve import _cython_core  # noqa: F401
        cy_avail = True
    except Exception:
        cy_avail = False

    # ---- direct: LU solve on dense ----------------------------------------
    A, b = make_system(n, spd=False)
    lu_solve(A, b, backend="numba")  # warmup JIT compilation (not timed)

    lu_rows = []
    t_py = None
    if measure_py:
        t_py = timeit(lambda: lu_solve(A, b, backend="python"), repeat=1)
        lu_rows.append(("pure Python", t_py))
    t_nb = timeit(lambda: lu_solve(A, b, backend="numba"), repeat=args.repeat)
    lu_rows.append(("Numba JIT", t_nb))
    t_cy = None
    if cy_avail:
        t_cy = timeit(lambda: lu_solve(A, b, backend="cython"), repeat=args.repeat)
        lu_rows.append(("Cython", t_cy))
    t_np = timeit(lambda: np.linalg.solve(A, b), repeat=args.repeat)
    lu_rows.append(("numpy.linalg.solve", t_np))

    # ---- Cholesky on SPD --------------------------------------------------
    Aspd, bspd = make_system(n, spd=True)
    cholesky_solve(Aspd, bspd, backend="numba")  # warmup

    ch_rows = []
    t_py_c = None
    if measure_py:
        t_py_c = timeit(lambda: cholesky_solve(Aspd, bspd, backend="python"), repeat=1)
        ch_rows.append(("pure Python", t_py_c))
    t_nb_c = timeit(lambda: cholesky_solve(Aspd, bspd, backend="numba"), repeat=args.repeat)
    ch_rows.append(("Numba JIT", t_nb_c))
    if cy_avail:
        ch_rows.append(("Cython",
                        timeit(lambda: cholesky_solve(Aspd, bspd, backend="cython"),
                               repeat=args.repeat)))
    t_np_c = timeit(lambda: np.linalg.solve(Aspd, bspd), repeat=args.repeat)
    ch_rows.append(("numpy.linalg.solve", t_np_c))

    # ---- print tables (each group uses its own baselines) -----------------
    def print_group(title, rows, py_base, np_base):
        print(f"[{title}]")
        print(f"{'method':<22}{'time (s)':>12}{'vs NumPy':>12}{'vs pure Py':>14}")
        print("-" * 60)
        for name, t in rows:
            ratio_np = np_base / t if t > 0 else float("inf")
            ratio_py = (py_base / t) if py_base else float("nan")
            py_str = f"{ratio_py:>13.1f}x" if py_base else f"{'n/a':>14}"
            print(f"{name:<22}{t:>12.4f}{ratio_np:>11.2f}x{py_str}")
        print()

    print_group("LU solve (dense)", lu_rows, t_py, t_np)
    print_group("Cholesky solve (SPD)", ch_rows, t_py_c, t_np_c)

    if not measure_py:
        print(f"(pure-Python baseline skipped: n={n} > {py_limit}; the interpreted "
              f"O(n^3) loop needs roughly {6e-7 * n ** 3:,.0f} s per call)\n")

    def vs_numpy(label, t, t_ref):
        r = t_ref / t
        who = f"{r:.2f}x faster" if r >= 1 else f"{1 / r:.2f}x slower"
        print(f"{label:<32}{who:>10} than NumPy")

    if t_py:
        print(f"{'Numba LU speedup over pure Python:':<32}{t_py / t_nb:,.0f}x")
    else:
        print(f"{'Numba LU speedup over pure Python:':<32}{'not measured':>10}")
    if cy_avail:
        print(f"{'Cython vs Numba (LU):':<32}{t_nb / t_cy:.2f}x")
    vs_numpy("Numba LU", t_nb, t_np)
    if cy_avail:
        vs_numpy("Cython LU", t_cy, t_np)
    vs_numpy("Numba Cholesky", t_nb_c, t_np_c)


if __name__ == "__main__":
    main()

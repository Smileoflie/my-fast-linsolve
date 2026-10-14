# linsolve（`fast-linsolve`）

面向科学计算的高性能线性方程组求解器库：封装 **3 种直接法 + 3 种迭代法**，
通过 **Numba JIT** 和 **Cython** 静态编译做深度性能优化，并在运行时自动回退
到纯 NumPy 实现。

## 功能特性

| 类别 | 算法 | 说明 |
|---|---|---|
| 直接法 | Gauss 消元 | 带部分主元选择（partial pivoting） |
| 直接法 | LU 分解 | PA = LU，Doolittle 格式，部分主元 |
| 直接法 | Cholesky 分解 | 适用于对称正定矩阵 |
| 迭代法 | Jacobi 迭代 | 经典定常迭代 |
| 迭代法 | Gauss-Seidel 迭代 | 原地更新 |
| 迭代法 | SOR 逐次超松弛 | ω ∈ (0, 2) |

三种后端，运行时自动按优先级选择：

1. `cython`  —— 编译后的 C 扩展（最快，构建时需要 C 工具链）
2. `numba`    —— JIT 编译的原生机器码
3. `python`   —— 纯 NumPy 循环参考实现（始终可用）

## 安装

```bash
pip install fast-linsolve
```

需要加速层时：

```bash
pip install fast-linsolve[numba]     # Numba JIT
pip install fast-linsolve[cython]    # Cython（需要 MSVC Build Tools 或 gcc）
```

## 快速上手

```python
import numpy as np
from linsolve import solve, gauss_solve, lu_solve, cholesky_solve
from linsolve import jacobi, gauss_seidel, sor

A = np.array([[4, 1, 2], [0, 5, 1], [1, 0, 3]], dtype=float)
b = np.array([7, 7, 4], dtype=float)

# 统一入口
x = solve(A, b, method="lu")
print(np.allclose(A @ x, b))   # True

# 直接法
x = gauss_solve(A, b)
x = lu_solve(A, b)

# 对称正定矩阵 -> Cholesky
Aspd = np.array([[4, 1, 1], [1, 3, 0], [1, 0, 2]], dtype=float)
x = cholesky_solve(Aspd, np.array([6, 7, 5], dtype=float))

# 迭代法返回 (x, info)
x, info = jacobi(A, b, tol=1e-8, max_iter=200)
print(info)   # {'iterations': ..., 'residual': ..., 'converged': True}

x, info = sor(A, b, omega=1.2, tol=1e-8)
```

强制指定后端：

```python
x = lu_solve(A, b, backend="numba")
x = lu_solve(A, b, backend="python")   # 慢速参考实现
```

## 性能数据

在稠密良态矩阵上实测（Windows，Python 3.14，Numba 单线程，
NumPy 走 LAPACK）：

| 规模 | 方法 | 纯 Python | Numba JIT | 加速比 | NumPy LAPACK |
|---|---|---|---|---|---|
| 300  | LU        | 4.84 s | 5.0 ms  | **960×**  | 24.6 ms |
| 300  | Cholesky  | 2.00 s | 0.9 ms  | **5,600×** | 9.4 ms  |
| 1000 | LU        | 180.4 s| 260 ms  | **694×**  | 67 ms   |
| 1000 | Cholesky  | 74.9 s | 34 ms   | **5,400×** | 42 ms   |

说明：

- LU 相比纯 Python 循环，Numba 加速比在 **100–800×** 区间；
  Cholesky 因为算术密度更低，Python 解释器开销占比更大，加速比更高。
- 在大矩阵稠密 LU 上，NumPy 自带的 LAPACK/BLAS（分块、SIMD、可能多线程）
  仍然更快；本项目的 Numba LU 在 n=1000 时约为 NumPy 速度的 26%。
  Cholesky 算术密度低，Numba 实现与 NumPy 持平甚至略快。
- 在支持的平台上，Cython 内核预计比 Numba 再快 **1.5–3×**
  （安装时有 C 工具链即自动编译）。

自己跑基准：

```bash
python benchmarks/benchmark.py --n 1000
```

## 开发

```bash
git clone https://github.com/Smileoflie/my-fast-linsolve.git
cd my-fast-linsolve
python -m pip install -e .[dev,numba]

# 跑测试
pytest

# 构建文档
sphinx-build -b html docs/source docs/build

# 打包并发布到 PyPI
python -m build
twine upload dist/*
```

## 项目结构

```
src/linsolve/
  __init__.py        对外 API
  direct.py          gauss / lu / cholesky 封装
  iterative.py       jacobi / gauss_seidel / sor 封装
  _raw.py            纯 Python 循环参考实现
  _numba.py          @njit JIT 实现
  _cython_core.pyx   Cython 静态类型内核
  _backend.py        运行时后端检测
tests/               pytest 测试（23 个用例）
benchmarks/          性能基准脚本
docs/source/         Sphinx 文档源
```

## 许可证

MIT

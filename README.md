# linsolve（`fast-linsolve`）

面向科学计算的高性能线性方程组求解器库：封装 **3 种直接法 + 3 种迭代法**，
通过 **Cython** 静态编译和 **Numba JIT** 做深度性能优化，并在运行时自动回退到
纯 NumPy 实现。

| 类别 | 算法 | 说明 |
|---|---|---|
| 直接法 | Gauss 消元 | 带部分主元选择（partial pivoting） |
| 直接法 | LU 分解 | `PA = LU`，Doolittle 格式，可复用分解结果 |
| 直接法 | Cholesky 分解 | 适用于对称正定矩阵，校验对称性 |
| 迭代法 | Jacobi 迭代 | 经典定常迭代 |
| 迭代法 | Gauss-Seidel 迭代 | 原地更新 |
| 迭代法 | SOR 逐次超松弛 | ω ∈ (0, 2) |

三种后端，运行时按优先级自动选择：

1. `cython` —— 预编译的 C 扩展（最快；官方 wheel 已内置，源码安装需要 C 编译器）
2. `numba` —— JIT 编译的原生机器码（`pip install fast-linsolve[numba]`）
3. `python` —— 纯 NumPy 循环参考实现（始终可用）

用 `linsolve.backend_info()` 可以查看当前实际生效的后端及原因。

## 安装

```bash
pip install fast-linsolve                 # wheel 内已含 Cython 扩展
pip install fast-linsolve[numba]          # 追加 Numba 后端
pip install fast-linsolve[cython]         # 从源码构建 Cython 扩展（需要 C 编译器）
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

# 迭代法返回 (x, info)，收敛判据为相对残差 ‖b−Ax‖ ≤ atol + rtol·‖b‖
x, info = jacobi(A, b, rtol=1e-8, max_iter=200)
print(info)   # {'iterations': ..., 'residual': <相对残差>, 'residual_abs': ..., 'converged': True}

x, info = sor(A, b, omega=1.2, rtol=1e-8)
```

### 一次分解、多次回代

`lu_factor` / `cholesky_factor` 把 O(n³) 的分解和 O(n²) 的回代拆开，多个右端项
（或随时间反复求解）时可以复用同一份分解——这正是 LU 相对 Gauss 消元的全部价值：

```python
from linsolve import lu_factor, lu_solve_factorized

LU, piv = lu_factor(A)                  # O(n^3)，只做一次
x1 = lu_solve_factorized(LU, piv, b)    # O(n^2)
X = lu_solve_factorized(LU, piv, B)     # B 为 (n, k)，一次解 k 个右端项

# 分解满足文档承诺的 P A = L U：
L = np.tril(LU, -1) + np.eye(len(A))
U = np.triu(LU)
```

实测（n=400，8 个右端项）：逐个调用 `lu_solve` 共 103 ms，分解一次后复用
13.5 ms，**7.6 倍**。

强制指定后端：

```python
x = lu_solve(A, b, backend="cython")
x = lu_solve(A, b, backend="python")   # 慢速参考实现
```

## 性能数据

Windows 11 / Python 3.13 / x86-64 / NumPy 1.26 (OpenBLAS) / Numba 0.68 / MinGW GCC，
`python -m benchmarks.benchmark`，每组取 3 次最小值：

**n = 400，多线程 BLAS（默认配置）**

| 方法 | 纯 Python | Numba | Cython | NumPy LAPACK | 相对 NumPy |
|---|---|---|---|---|---|
| LU 求解 | 6.77 s | 9.6 ms | **7.7 ms** | 122 ms | Cython **快 15.9×** |
| Cholesky 求解 | 2.64 s | **4.4 ms** | 9.7 ms | 76 ms | Numba **快 17.2×** |

**n = 1200，全部单线程（`OMP/OPENBLAS/NUMBA_NUM_THREADS=1`）**

| 方法 | Numba | Cython | NumPy LAPACK |
|---|---|---|---|
| LU 求解 | 447 ms | 442 ms | **27.7 ms**（快 15.4×） |
| Cholesky 求解 | 71 ms | 250 ms | **28.0 ms**（快 2.5×） |

这两张表怎么读（也是这个项目的核心结论）：

- 相对纯 Python 循环，编译后端快 **600–880×**。
- **中小规模（n ≲ 400）我们的内核比 NumPy 快 12–17 倍**：LAPACK 在这个规模下
  每次调用的固定开销和多线程启动开销占比过高，而我们省掉了这些。
- **大规模（n ≳ 1000）LAPACK 反超约 15 倍**：它用分块算法 + SIMD + 多线程，
  而我们的是教科书三重循环，没有分块、没有显式向量化、也不并行。
- 实测过 cache blocking：改成面板分解 + rank-B 更新后，n ≤ 1200 只快
  1.09–1.17×，n = 2000 反而慢 4%。**因此没有采用**——单纯分块填不平与 BLAS-3
  的差距，得配上 SIMD 和线程才能真正接近 LAPACK。
- Cython 相对 Numba 约 **1.0–1.24×**。这个数字强烈依赖优化级别：setuptools 给
  MinGW 默认传 `-O1` 时 Cython 只有 Numba 的 **0.73–0.89×**；本项目在 `setup.py`
  的 `OptionalBuildExt` 里显式加了 `-O3`。

跑基准（默认 n=500）：

```bash
python -m benchmarks.benchmark --n 1000
```

> 纯 Python 是解释器 O(n³) 循环，n=1200 单次约 1000 秒。基准脚本在 n > 400 时
> 自动跳过这一列，避免看起来像卡死。

## 数值约定

- **收敛判据**：`‖b − A x‖ ≤ atol + rtol · ‖b‖`（SciPy 约定，`rtol=1e-8`，
  `atol=0.0`）。`info["residual"]` 是相对残差，`info["residual_abs"]` 是绝对残差。
  旧参数 `tol` 仍可用，但按绝对容差处理并给出 `DeprecationWarning`。
- **奇异判据**：`eps · max|A| · n`（相对阈值），因此整体缩放不改变可解性。
- **Cholesky 会校验对称性**：非对称矩阵直接报错，而不是静默返回错误解。
- **迭代法会校验**：对角元非零、`b` 为一维、`x0` 长度正确。
- 所有函数都不修改传入的数组，一律返回新的 `float64` 数组。

## 开发

```bash
git clone https://github.com/Smileoflie/my-fast-linsolve.git
cd my-fast-linsolve
python -m pip install -e .[dev,numba,cython]

pytest                                     # 覆盖三个后端
python -m benchmarks.benchmark --n 1000
sphinx-build -b html docs/source docs/build
```

CI（`.github/workflows/ci.yml`）在 Linux/Windows/macOS × Python 3.9–3.13 上跑测试，
并断言编译后端确实生效（`LINSOLVE_REQUIRE_CYTHON=1`：扩展没编出来即失败，而不是
静默降级）。打 `v*` tag 时用 cibuildwheel 构建各平台 wheel 并发布到 PyPI。

## 项目结构

```
src/linsolve/
  __init__.py        对外 API
  direct.py          gauss / lu / cholesky 及分解复用封装
  iterative.py       jacobi / gauss_seidel / sor 封装
  _kernels.py        六种算法的唯一实现源，Numba 直接编译同一份源码
  _cython_core.pyx   Cython 静态类型内核（与 _kernels.py 一一对应）
  _backend.py        后端检测与调度
benchmarks/          性能基准脚本
docs/source/         Sphinx 文档源
.github/workflows/   CI 与自动发版
```

设计上只有**两份**算法实现（Python 与 Numba 共用同一份源码，Cython 一份），
而不是历史上的三份各自维护——后者曾导致 `max_iter=0` 在不同后端行为不一致。
详见 [CODE_REVIEW.md](CODE_REVIEW.md)。

## 变更记录

见 [CHANGELOG.md](CHANGELOG.md)。

## 许可证

MIT

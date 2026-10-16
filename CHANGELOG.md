# Changelog

## 0.3.0 — 分解复用、单源内核、可发布的编译 wheel

### 新增

- **分解复用 API**：`lu_factor(A) -> (LU, piv)` / `lu_solve_factorized(LU, piv, b)`、
  `cholesky_factor(A) -> L` / `cholesky_solve_factorized(L, b)`。分解只做一次，
  多个右端项复用；`b` 支持 `(n,)` 或 `(n, k)` 批量右端项。
  实测 n=400、8 个右端项：103 ms → 13.5 ms（**7.6×**）。
  `lu_factor` 的返回契约 `P A = L U` 有专门的测试直接验证。
- `linsolve.backend_info()` 已在 0.2.0 加入，本版补充到 README 与文档。

### 架构

- **`_raw.py` / `_numba.py` 合并为单一 `_kernels.py`**：六种算法只写一遍，
  Numba 通过 `_jit()` 编译**同一份源码**（`kernels.nb.*`）。历史上的三份重复实现
  正是 `max_iter=0` 行为分叉的根源。算法实现从三份降为两份
  （`_kernels.py` + `_cython_core.pyx`）。
- 后端调度收敛为 `_backend.kernels_for()`：三个后端暴露同名内核，封装层只剩一次
  查表，不再逐个 if/elif 分发。

### 构建与发布

- **wheel 现在真正带编译内核**：此前 `build-system.requires` 不含 numpy/Cython，
  构建时静默跳过扩展，发出去的是不含任何编译内核的 `py3-none-any`。
  现将 numpy/Cython 加入构建依赖，并新增 `LINSOLVE_REQUIRE_CYTHON=1`：
  CI 与 cibuildwheel 下扩展未产出即失败。
- `setup.py` 的 `OptionalBuildExt`：gcc 系工具链显式加 `-O3`
  （setuptools 默认传 `-O1`，此时 Cython 只有 Numba 的 0.73–0.89×；加 `-O3` 后为
  1.0–1.24×）；无编译器时打印明确警告并降级，而不是静默产出纯 Python 包。
- 新增 `.github/workflows/ci.yml`：Linux/Windows/macOS × Python 3.9–3.13 测试矩阵
  （并断言编译后端生效）、cibuildwheel 平台 wheel、sdist，`v*` tag 自动发 PyPI。
- `benchmarks/benchmark.py`：LU 与 Cholesky 分行统计各自的基线；新增 Cython 行；
  n > 400 时自动跳过纯 Python 基线（解释器 O(n³) 在 n=1200 需约 1000 秒/次，
  此前会让基准看起来像卡死）。

### 文档

- README 性能章节改为**实测数据 + 硬件声明 + 方法说明**，删除了此前无法复现的
  固定百分比（原"1000 阶达 NumPy 62%""Cython 比 Numba 快 1.5–3×"）。当前结论：
  n ≲ 400 时编译内核比 NumPy 快 12–17×，n ≳ 1000 时 LAPACK 反超约 15×；
  cache blocking 实测仅 1.09–1.17×，故未采用。
- 新增 `CODE_REVIEW.md`：项目自查报告，含每条问题的复现实验与修复状态。

### 兼容性

- 无破坏性变更（0.2.0 的 `rtol/atol` 与关键字参数约定保持不变）。
- 新增的 `lu_solve`/`cholesky_solve` 内部改为"分解 + 回代"两步，数值路径等价。

## 0.2.0 — 数值健壮性修复

本轮修复来自一次完整的代码审计（见 `CODE_REVIEW.md`），全部为行为级修复。

### 修复（行为变更）

- **Cholesky 增加对称性校验**：非对称矩阵此前会静默返回错误解，现在抛出
  `ValueError`（`cholesky_solve`）。
- **奇异判据改为相对阈值**：由写死的绝对值 `1e-15` 改为
  `eps * max|A| * n`。良态但整体缩放的矩阵（如 `1e-16 * A`）不再被误判为奇异
  （`gauss_solve` / `lu_solve`，三后端一致）。
- **迭代法收敛判据改为相对残差**（SciPy 约定）：
  `‖b − A x‖ ≤ atol + rtol * ‖b‖`。新增 `rtol`（默认 `1e-8`）与 `atol`
  （默认 `0.0`）参数；原 `tol` 保留为绝对容差的兼容别名，使用时会发出
  `DeprecationWarning`。`info["residual"]` 现在报告**相对**残差，
  新增 `info["residual_abs"]` 报告绝对残差。
- **迭代法前置校验**：对角元为 0 时抛出 `ValueError`（此前静默产出 inf/nan）；
  `b` 必须为 1-D；`x0` 长度必须与 `n` 一致。
- **`max_iter=0` 修复**：不再抛 `UnboundLocalError`；所有后端统一为
  "先检查初始残差"，精确的 `x0` 会以 0 次迭代收敛。
- **`solve()`**：对直接法传入多余关键字参数（如 `tol`）现在抛出 `TypeError`，
  不再静默丢弃。

### 工程

- 新增 `linsolve.backend_info()`：报告各后端可用性及不可用原因，便于排查
  后端的静默降级。
- 打包：`build-system` 要求提升为 `setuptools>=77`（`license` 的 PEP 639 语法
  实际需要它，此前 `setuptools<77` 源码构建会失败）；版本号改为
  `dynamic`，以 `linsolve.__version__` 为唯一来源。
- benchmark 表格修复：LU 与 Cholesky 两组各自使用自己的 NumPy / 纯 Python
  基线（此前 Cholesky 行的 "vs NumPy"/"vs pure Py" 错用了 LU 的基线），并打印
  BLAS 线程数环境变量提示。
- 测试：23 → 42 个用例，新增非对称/缩放/零对角/`max_iter=0`/尺度不变性等回归测试。

### 兼容性注意

- 迭代法签名变为关键字参数：`rtol`/`atol`/`max_iter`/`backend` 现在仅能以
  关键字传入；`tol` 仍可用但已弃用。
- `info["residual"]` 语义由绝对残差变为相对残差。

## 0.1.1

- 首个发布版本：3 种直接法 + 3 种迭代法，Cython/Numba/纯 Python 三后端自动回退。

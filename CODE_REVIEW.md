# fast-linsolve 代码审查报告

> 审查范围：`src/linsolve/` 全部 6 个模块、测试、benchmark、打包配置。
> 所有结论都在本机（Windows / Python 3.12 / NumPy 1.26.4 / Numba 0.60.0 / 16 线程 BLAS）
> 重新跑过复现实验，证据附在每条问题下。
> 已有基线：`pytest` 23 个用例全部通过 ✓

> **修复状态**
>
> - **0.2.0**：§1 全部 7 条（P0 正确性/健壮性）+ 工程项 §4.3 setuptools、§4.4 版本号、
>   §4.5 benchmark 表格、§4.6 回归测试、§4.7 移除 INTERVIEW_PREP.md、§4.8
>   `resolve()`/`backend_info()` —— 测试 23 → 42 全绿。
> - **0.3.0**：§2.1 分解复用 API、§4.1 三份实现合并为两份（`_kernels.py` 单源）、
>   §4.2 编译 wheel 与 cibuildwheel CI —— 测试 42 → 71 全绿。
> - **明确不修**：§3.3 / §3.4 的 cache blocking 已实测（1.09–1.17×，n=2000 反而
>   慢 4%），结论是不值得保留，理由与数据见 README。
> - 剩余可选增强见 §6。

---

## 0. 优先级速览

| 级别 | 问题 | 证据 | 位置 |
|---|---|---|---|
| **P0** | Cholesky 不校验对称性，静默返回错误解 | 残差 27.0（应为 0） | `_raw/_numba/_cython` cholesky |
| **P0** | 奇异判据用写死的绝对阈值 `1e-15`，良态矩阵被误判 | `1e-16*A` 抛"矩阵奇异" | gauss/lu 三处 |
| **P0** | 收敛判据是**绝对**残差，与 `b` 量级耦合 | `|b|=1e-6` 时 4 步"收敛"，相对误差 6.9e-3 | `_iterate` / 三个 nb / cy |
| **P1** | LU/Cholesky 分完就丢，无法复用分解 | 5 个右端项重复分解 33 ms vs 1.5 ms | `direct.py` |
| **P1** | 三后端共 644 行近重复代码，已经出现行为分叉 | `max_iter=0` 时 python 报错 / numba 正常 | `_raw/_numba/_cython` |
| **P1** | 迭代法每次迭代重算残差，实测吃掉 31% 时间 | n=500、200 次迭代：15.8 ms 中 4.9 ms | 三个后端 |
| **P1** | 发布的 wheel 是 `py3-none-any`，**没有 Cython 扩展** | wheel 内无 `.pyd/.so` | `dist/` |
| **P2** | `license = "MIT"` 需 setuptools≥77，但声明 ≥68 | setuptools 75.1.0 解析直接失败 | `pyproject.toml` |
| **P2** | benchmark 表格把 LU 的 baseline 套用到 Cholesky 行 | "Cholesky 2.4x vs pure Py" | `benchmark.py` |
| **P2** | 对角元为 0 时迭代法静默产出 inf/nan | `x=[inf inf]`，无异常 | 迭代法三处 |
| **P2** | `INTERVIEW_PREP.md` 被提交进仓库 | 仓库含 GitHub 地址 | 仓库根目录 |

---

## 1. P0 · 正确性与数值健壮性

### 1.1 Cholesky 不校验对称性 → 静默给错答案（最严重）

只读了 `A` 的下三角（`A[i, j]`，`j <= i`），上三角被完全忽略。喂一个非对称矩阵不会报错，
而是返回一个**看起来正常但完全错误**的解：

```python
A = np.array([[4., 100.], [1., 3.]])     # 非对称
x = cholesky_solve(A, b)
# 实际得到 x = [0.1818, 0.2727]   残差 ‖Ax-b‖ = 27.0
# 真解      x = [1.1023, -0.0341]  残差 0
```

这正是 Cholesky 最容易踩的坑——比正态矩阵算错更危险，因为不报错。

**建议**（三处 kernel + 三个 wrapper 都要加）：

```python
def _check_symmetric(A, tol=1e-10):
    scale = np.abs(A).max()
    if not np.allclose(A, A.T, rtol=tol, atol=tol * max(scale, 1.0)):
        raise ValueError("A must be symmetric; use lu_solve/gauss_solve otherwise.")
```

如果担心 `O(n²)` 校验开销，可以加 `check_symmetric=True` 开关，并在文档里说清
"关掉即等价于只使用下三角"。

### 1.2 奇异判据是绝对阈值，不是相对阈值

`if abs(M[pivot, k]) < 1e-15` —— 这个 1e-15 是写死的绝对值，与矩阵量级无关：

```python
A = 1e-16 * np.array([[2., 1.], [1., 3.]])   # 条件数仍是 3.6，完全良态
gauss_solve(A, b)   # ValueError: Matrix is singular to working precision.
```

数值上正确答案应该被正常算出来。反过来，量级很大的矩阵（比如 1e20 量级的有限元刚度矩阵）
时这个阈值又形同虚设。

**建议**（三后端 gauss/lu 共 6 处）：先扫一遍求 `amax`，用相对判据

```python
eps = np.finfo(np.float64).eps
amax = max(abs(M[i, j]) for i in range(n) for j in range(n))
pivot_tol = eps * amax * n          # 或 eps * amax
...
if best <= pivot_tol:
    raise ValueError("Matrix is singular to working precision.")
```

对 O(1) 矩阵来说新阈值与旧的 `1e-15` 量级相近（eps·n·amax ≈ 2.6e-15，n=3），
所以不会破坏现有测试，但缩放不变性就正确了。

### 1.3 收敛判据是绝对残差 → 与 `b` 的量级耦合

`‖b - Ax‖₂ < tol` 里的 `tol` 是**绝对**量，于是同样的 `tol=1e-8` 在不同缩放下含义完全不同。
实测（`A=[[4,1],[1,3]]`，`tol=1e-8`）：

| ‖b‖ | 迭代数 | 报告"收敛"时的绝对残差 | **真实相对残差** |
|---|---|---|---|
| 1.4e0 | 16 | 3.3e-9 | 2.3e-9 |
| 1.4e6 | 27 | 3.9e-9 | 2.8e-15 |
| **1.4e-6** | **4** | 9.8e-9 | **6.9e-3** ← 解只对 2 位有效数字 |

也就是说第 3 行这种情况，函数报告"收敛"，但解是错的。这是迭代法最典型的接口缺陷。

**建议**：改成 SciPy 风格的双参数判据，并把 `info["residual"]` 换成相对量

```python
def _done(rnorm, bnorm, rtol, atol):
    return rnorm <= atol + rtol * bnorm
```

签名可以平滑过渡（`tol` 保留为 `atol` 的别名并给 DeprecationWarning）；
因为默认语义变了，建议把版本号推到 `0.2.0` 并在 CHANGELOG 里写明。

### 1.4 `max_iter=0` 后端行为不一致（分叉 bug 的实证）

```python
jacobi(A, b, max_iter=0, backend="python")  # UnboundLocalError: 'rnorm'
jacobi(A, b, max_iter=0, backend="numba")   # 正常返回 {'iterations': 0, ...}
```

原因：`_raw._iterate` 的 `rnorm` 只在循环体内赋值，循环不执行就未绑定。
这不是"边缘 case 无所谓"——它是三份实现各自维护、已经产生语义偏差的直接证据
（详见 4.1）。**建议**：进循环前先算一次初始残差，顺便也让返回值语义更清晰。

### 1.5 对角元为 0 时静默产出 inf/nan

```python
A = np.array([[0., 1.], [1., 0.]])
jacobi(A, b, max_iter=3)   # x = [inf inf]，info['residual'] = nan，无异常
```

伴随 `RuntimeWarning: divide by zero`。**建议**在 `_as_float` 里前置校验
（`np.any(np.diag(A) == 0)` → raise），并在迭代结束时检查 `np.isfinite`，否则
用户拿到 nan 会一路传播到自己的结果里。

### 1.6 `b` 为二维时报错信息是 Numba 内部栈

```python
lu_solve(A, np.ones((2, 1)))
# 抛 numba.core.errors.TypingError，附 16 条候选实现的长栈
```

用户完全看不懂，而且 python/numba 后端报错类型还不一样。**建议**在 `_as_float` 里
显式 `if b.ndim != 1: raise ValueError(...)`（多右端项支持见 2.1）。

### 1.7 `solve()` 会静默丢弃直接法的多余参数

```python
solve(A, b, method="lu", tol=1e-10)   # 不报错，也不生效
```

直接法分支是 `_DIRECT[m](A, b, backend=backend)`，`**kwargs` 被吞掉了。用户少打一个
`method` 就会以为 `tol` 生效了。**建议**：直接法分支显式拒绝额外 kwargs
（`if kwargs: raise TypeError(...)`），或者干脆把 `solve` 拆成
`solve_direct` / `solve_iterative`。

---

## 2. P1 · API 设计（"封装"最该补的一块）

### 2.1 LU 分完就丢，丢了 LU 存在的意义

LU 相对 Gauss 消元的**唯一优势**就是"一次分解、多次回代"（分解 O(n³)，每次回代 O(n²)）。
但现在的接口是 `lu_solve(A, b) -> x`，`L/U/P` 在函数返回瞬间被丢掉。
实测 n=300、5 个右端项：

```
分别调用 lu_solve(A, B[:,k]) × 5 : 33.0 ms   ← 分解被重做了 5 次
np.linalg.solve(A, B) 一次性      :  1.5 ms   ← 分解 1 次 + 5 次回代
```

22 倍差距全在这里。**建议**拆成三段式（Cholesky 同理）：

```python
lu_factor(A, backend="auto")        -> (LU, piv)      # LU 打包存，piv 是行交换记录
lu_solve_factorized(LU, piv, b)     -> x
cho_factor(A)                       -> L
cho_solve_factorized(L, b)          -> x
# 保留 lu_solve(A, b) 作为便捷封装 = factor + solve_factorized
```

再顺手支持多右端项 `b` 为 `(n, k)`（LAPACK 的 `dgetrs` 就是这语义），这样上面那
22 倍才真正拿到手。这是个很自然的 0.2.0 特性，也是面试里讲"我理解 LU 的意义"的
硬证据。

### 2.2 直接法和迭代法返回值形态不一致

`solve()` 的返回类型随 `method` 变化（直接法 → `ndarray`，迭代法 → `(x, info)`），
调用方没法写统一代码。**建议**二选一：全部返回 `(x, info)`，或给直接法也补一个
`info`（含 `pivots` / `residual` / `condition_estimate`），并保留
`return_info=False` 的兼容开关。

### 2.3 其他接口小项

- 缺 `atol/rtol`（见 1.3）、`verbose`、非收敛时的 `warnings.warn`（现在只靠返回值，容易漏检）。
- `x0` 长度不校验，错误长度会抛出 numpy 内部的 `cannot assign slice from input of
  different size`，建议前置校验。
- 建议导出 `py.typed` 并补 `linsolve/__init__.pyi`，现在 `backend: str` 这类参数
  用字面量联合类型（`Literal["auto","cython","numba","python"]`）会更好用。

---

## 3. P1 · 性能（全部为实测数据）

### 3.1 迭代法每次迭代都重算一遍残差 —— 实测占 31% 时间

n=500、跑满 200 次迭代：

```
Jacobi 总耗时           = 15.8 ms
其中残差检查部分（200 次）=  4.9 ms   (31%)
```

也就是说每次收敛检查的 O(n²) 扫描，直接把迭代法的成本翻倍。
**建议**（按性价比排序）：

1. **稠密场景**：不要每次都检查，改成每 `m` 次迭代检查一次（`m=5~10`），
   或者用"更新量范数变小了再算精确残差"的两阶段判据。这一条就能回收约 25% 的时间，
   改动量极小，接口不用变。
2. **稀疏场景**（未来支持 CSR 时）：维护残差向量才是真正划算的做法——
   更新 `x_i` 时 `r_i = s_i - A_ii·x_i^old` 是免费得到的，之后 `r_k -= A_kj·Δx_j`
   在稀疏下是 O(nnz)，比重新走一遍 O(n²) 便宜一个数量级。

> 注意：稠密矩阵下"每步维护残差向量"的代价同样是 O(n²/分量)，并不省——所以
> 稠密场景请走第 1 条。这一点容易被想当然搞错，值得在报告/README 里写清楚。

### 3.2 与 NumPy 的对比不公平，README 的百分比建议改写

`np.linalg.solve` 走的是**多线程** OpenBLAS（本机 16 线程），而 Numba 是单线程。
本机实测 n=300 LU：

```
LU pure Python     2.988 s
LU Numba JIT       5.3 ms     (561× vs 纯 Python)
LU numpy.linalg    0.6 ms     ← Numba 只有它的 11.3%
```

README 里写的 26%（n=1000）和这个 11.3% 都能自圆其说，但既然你已经有
`INTERVIEW_PREP.md` 里那份"数字对不上怎么解释"的焦虑，最干净的解法是**让数字
可复现**：

- benchmark 里设 `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` 做单线程对单线程；
- 同时打印环境信息（CPU 型号、核数、BLAS 后端 `np.show_config()`、NumPy/Numba 版本）；
- 把 README 的固定百分比改成"实测表格 + 硬件声明"，或者直接写区间。

这样"62% 还是 26%"这种问题就不存在了——你手里有可复现的脚本和声明了硬件的表格。

### 3.3 朴素 `parallel=True` 会让 LU 变慢（有数据，别踩）

在很多"还能怎么优化"的答案里会写"Numba 可以 `parallel=True` 多线程"。实测并不成立：

```
n=300 LU   串行 @njit          :  7.23 ms
n=300 LU   @njit(parallel=True): 14.76 ms   ← 慢了一倍
numpy.linalg.solve             :  0.89 ms
```

原因：`prange` 加在长度随 `k` 递减的 i 循环上，每轮外层都付出线程调度开销，
而并行收益随 `k` 增大迅速衰减。**结论**：要真正吃多核，必须先做**分块（blocking）**，
再在块内更新上并行。这个"我试过，反而更慢，因为……"的结论，比空谈"可以加
parallel"值钱得多。

### 3.4 缺 cache blocking —— 与 LAPACK 差距的主因

教科书三重循环天然不 cache-friendly（列主序下 `M[i,j]` 沿 j 走是跨行访问）。
LAPACK 快在分块 + SIMD + 多线程。如果只想挑一件事做，做分块：

```python
B = 64   # 块大小
for k in range(0, n, B):
    for kk in range(k, min(k+B, n)):
        # 选主元 + 更新 [k+B:, kk] 列
        # 块内剩余部分单独走小循环（j 在 kk+1..k+B 内）
    # 块外尾部的 rank-B 更新用 j 分块展开，提高 cache 命中
```

预期能把 n=1000 的 LU 从"NumPy 的 11%"拉到 40–60% 区间（具体数字跑出来再说，
不要在 README 里先写死）。

> **实测结论（0.3.0）**：不成立。面板分解 + rank-B 更新在 n=300/600/1200 上只快
> 1.09/1.11/1.17×，n=2000 反而慢 4%（见 README 的性能章节数据）。原因是内层 j 循环
> 本来就是连续访存、LLVM 已经做了向量化，分块减少的 cache 流量很有限；真正的差距
> 在 BLAS-3（外积形式 + SIMD + 多线程），单纯分块补不上。**因此该优化未采用。**

### 3.5 参考实现的小瑕疵

`_raw.jacobi` 每次迭代都 `np.zeros` 新建数组（numba 版已经提出去了），
参考实现本身也可以更干净；`_raw.cholesky_solve` 热循环里用 `np.sqrt` 标量调用，
改用 `math.sqrt` 更合适（虽然它本来就要慢）。

---

## 4. P1/P2 · 工程与架构

### 4.1 三份实现共 644 行重复代码，且已经分叉

```
_raw.py            209 行
_numba.py          212 行   ← 与 _raw 算法逐行相同，只是加了装饰器
_cython_core.pyx   223 行   ← 同上，改成静态类型
```

同一个算法要改三遍。1.4 那条 `max_iter=0` 的 bug 就是分叉的实证
（`_raw` 错、`_numba` 对）。**建议两条路**：

**（A）把 `_numba.py` 整个删掉（推荐，我验证过可行）**
Numba 不支持闭包捕获 `n`，这才是当初要写第二份源码的原因。把 `_raw._iterate`
里的 `update(A, b, x)` 闭包改成整数标志位 `method`，**同一份源码**就能被 `@njit`
直接编译。实测：改完之后 njit 版本与现有 `jacobi_nb` 结果逐位一致（迭代数 13 = 13）。

```python
@njit(cache=True, fastmath=True)
def _iterate(A, b, x0, tol, max_iter, method):   # method: 0=Jacobi, 1=GS, 2=SOR
    ...

_iterate_py   = _iterate            # 纯 Python 直接调
_iterate_njit = _njit(_iterate)     # 有 numba 时用 njit 包装
```

维护量 3 份 → 2 份，且不会再出现 1.4 那种分叉。
（代价：纯 Python 参考实现里多一个 `if method ==` 分支，这点开销无所谓，
它本来就是慢速基线。）

**（B）更彻底**：Cython 作为唯一的"快"实现，`_raw.py` 只作参考实现，
不再试图让三份都对齐。

**Cython 侧的额外改进**：`.pyx` 里参数还在用旧缓冲区语法
`np.ndarray[D, ndim=2, mode="c"]`，建议换成内存视图 `double[:, ::1]` ——
少一层 numpy 数组对象封装，且新语法是官方推荐方向（旧语法已标记不推荐）。
`_resid` 的 `except? -1.0` 也建议改掉：-1.0 是合法返回值，用 `except?` 语义上不干净，
直接 `except *` 或去掉异常规格。

### 4.2 发布的 wheel 里没有 Cython 扩展（这条最亏）

```
$ unzip -l dist/fast_linsolve-0.1.1-py3-none-any.whl
   linsolve/_cython_core.pyx   ← 只带了源码，没有编译产物
   含编译扩展(.pyd/.so): False
```

`py3-none-any` 意味着**所有从 PyPI 安装的用户永远跑不到 Cython 后端**，只能
numba 或纯 Python。你在 README 和面试准备里反复讲的"Cython 再快 1.5–3×"，
用户实际一次都体会不到——这是整份代码库里投入产出比最失衡的一点。

**建议**：

1. 加 GitHub Actions，用 `cibuildwheel` 出 Windows/macOS/Linux × py3.9–3.13 的
   **平台 wheel**（`cp39-win_amd64` 这种，名字里带平台和 ABI 标签）；
2. 同时保留一个 `py3-none-any` 的纯 Python 兜底 wheel 给没有匹配平台的人；
3. 顺带在 CI 里跑 pytest，不要再手动"本机没装 MSVC 所以没编出来"。

### 4.3 打包元数据：`license` 字段需要 setuptools ≥ 77，但声明的是 ≥ 68

`pyproject.toml` 用了 PEP 639 的新语法：

```toml
license = "MIT"
license-files = ["LICENSE"]
```

PEP 639 支持是 **setuptools 77.0.0** 才加的。实测用 setuptools 75.1.0 解析直接失败：

```
ValueError: invalid pyproject.toml config: `project.license`.
configuration error: `project.license` must be valid exactly by one definition (2 matches found)
```

而 build-system 只声明了 `requires = ["setuptools>=68", "wheel"]`。
pip 默认开构建隔离会拉最新 setuptools，所以**通常**能过，但：

- `pip install --no-build-isolation .` 直接构建失败；
- 系统 setuptools < 77 的环境（很多发行版自带）失败；
- 用户从 sdist 构建时踩坑。

**修**：`requires = ["setuptools>=77"]`，或退回 `license = { text = "MIT" }`。
（你现有 `dist/` 里那个 wheel 的 `License-Expression: MIT` 说明它是用新 setuptools 构建的，
所以问题只在源码安装路径上——但这类问题最难排查。）

### 4.4 版本号三处硬编码

`pyproject.toml`、`src/linsolve/__init__.py`、README 里的 `0.1.1` 各写一遍，
发版时容易漏。**修**：

```toml
dynamic = ["version"]
[tool.setuptools.dynamic]
version = { attr = "linsolve.__version__" }
```

### 4.5 benchmark 脚本的表格算错了

`ratio_np` 和 `baseline_py` 都取自 LU 的 NumPy / 纯 Python 时间，却套用到了
Cholesky 的行上。实跑输出：

```
Cholesky pure Python    1.2669    0.00x     2.4x    ← 2.4 = LU的纯Python时间 / Cholesky的纯Python时间
Cholesky numpy (ref)    0.0004    1.47x  7246.2x    ← 1.47 的分母是 LU 的 numpy 时间
```

跨方法比较没有意义，而 Cholesky 行的 "vs NumPy" 分母（0.0006）也不是它真正的
参照（0.0004）。**修**：把基准按方法分组，每组各自算 baseline；表格加一列
"同组内最快"或直接输出 JSON 便于回归对比。

### 4.6 测试覆盖的缺口

现有 23 个用例都过 ✓，但恰好漏掉了本报告里所有真实 bug：

- ❌ 非对称矩阵喂给 `cholesky_solve`（1.1，会静默给错答案——**最该补的一条**）
- ❌ 缩放不变性：`solve(1e-16*A, 1e-16*b)` 应等于 `solve(A, b)`（1.2）
- ❌ 相对残差：带尺度变化的 `b` 下"收敛"的解是否真的可用（1.3）
- ❌ `max_iter=0`（1.4）、对角元为 0（1.5）、`b` 为二维（1.6）、`x0` 长度错（2.3）
- ❌ 病态矩阵（Hilbert 矩阵、条件数 1e12 级别）的精度测试
- ❌ 后端**两两互比**：现在每个后端都各自与 NumPy 比，没有交叉验证。
  建议上 `hypothesis` 做 property-based testing（随机 n、随机矩阵、随机多个后端比对）
- ❌ 性能回归测试：benchmark 只打印，不设阈值。可以在 CI 里对 `pure python` 的
  相对耗时设一个宽松上限（比如 300% 抖动带），防止某次重构悄悄把算法退化成 O(n³) 常数翻倍

### 4.7 `INTERVIEW_PREP.md` 建议从仓库里移除

README 里公开了 GitHub 地址，而 `INTERVIEW_PREP.md` 里写着
"这个数字如果被追问怎么测的，你要准备说法"、"定性描述，别编实验细节"、
以及一张"简历写的 vs 实测"的对照表（第 175–180 行）。

如果这个仓库是公开的、或简历里附了链接、或面试官顺手点开了 GitHub：
这个文件提供的信息量远大于收益。它不是在证明你懂数值计算，而是在记录
"我知道简历数字对不上，并准备了说辞"。

**建议**：`git rm --cached INTERVIEW_PREP.md` 并加入 `.gitignore`，本地自己留着
（内容本身对你是有用的）。真想留一份进仓库，就把它改写成
`docs/DESIGN_NOTES.md`：只谈设计取舍与优化思路，去掉简历数字那一节。

### 4.8 杂项

- `_backend.resolve()` 里 `raise ValueError(...)` 建议加 `from None`，
  否则 `except ValueError: raise ValueError(...)` 会给用户打印
  "During handling of the above exception, another exception occurred" 的嵌套噪音。
- `detect_backend()` 用 `except Exception: pass` 吞掉一切。用户装了 Cython 但 ABI
  不匹配时会**毫无提示**地降级到 Numba。至少在降级时 `warnings.warn` 一句并说明原因。
- `detect_backend()` 在 import 时求值一次，进程内后装的 numba 不会生效（可接受，
  但建议文档里写明，并提供 `refresh_backend()`）。
- 缺 `CHANGELOG.md`、`docs/` 里缺"后端选择与性能特征"一章（README 有，但 Sphinx 文档里没有）。
- `egg-info/` 和 `dist/` 已在 `.gitignore` 里但文件还在工作区，发布前清理一下。

---

## 5. 建议的执行顺序

**第一晚（1–2 小时，收益最大）**
1. 1.1 Cholesky 对称性校验（3 处 wrapper 即可，不用动 kernel）
2. 1.2 相对奇异判据
3. 1.4 `max_iter=0` 修复 + 1.5 对角元校验
4. 补 4.6 里对应的 4 个测试

**第二晚（半天）**
5. 1.3 收敛判据改 `rtol/atol`（顺带 0.2.0 版本号 + CHANGELOG）
6. 4.3 `setuptools>=77` + 4.4 动态版本号 + 4.5 benchmark 修表
7. 4.7 把 `INTERVIEW_PREP.md` 移出仓库

**然后（真正加分，需要 1–2 天）**
8. 2.1 `lu_factor` / `lu_solve_factorized` + 多右端项（把 22 倍差距拿到手）
9. 4.1(A) 消除 `_numba.py`，合并成两份源码
10. 4.2 GitHub Actions + cibuildwheel 预编译 wheel

**有余力再做**
11. 3.4 cache blocking（配 3.3 里"为什么朴素 parallel 反而慢"的实验）
12. 稀疏矩阵 CSR 支持（这一步才真正打开迭代法的价值空间）

---

## 6. 后续工作（尚未处理）

按性价比排序，均为可选增强：

1. **稀疏矩阵（CSR）支持** —— 这才是迭代法真正的用武之地；配合稀疏格式，迭代法的
   存储从 O(n²) 降到 O(nnz)，收敛检查也可以做成增量式。
2. **SIMD / 线程** —— 分块实测只快 1.09–1.17×（见 §3.4 与 README），真正拉开与
   LAPACK 差距的是向量化与多线程；朴素 `parallel=True` 反而更慢（§3.3）。
3. **`solve()` 返回值统一** —— 直接法返回 `ndarray`、迭代法返回 `(x, info)`，
   调用方无法写统一代码（§2.2）。
4. **收敛检查降频** —— 迭代法每次迭代重算残差占 31% 耗时，改成每 m 次检查一次可
   回收大部分（§3.1）。
5. **`_cython_core.pyx` 迁移到内存视图** —— 目前仍用 `np.ndarray[...] mode="c"`
   旧缓冲区语法，改用 `double[:, ::1]` 更快，也是官方推荐方向（§4.1）。
6. **`py.typed` / 类型标注** —— `backend: str` 可改为
   `Literal["auto","cython","numba","python"]`（§2.3）。

## 7. 本次审查的方法

本报告的所有结论都可以在本机重新跑出来：

- `pytest` —— 回归测试（含新增的对称性、缩放不变性、尺度不变性、边界条件用例）
- `python -m benchmarks.benchmark --n 400` —— 性能对比（分组基线、跳过慢速纯 Python 列）
- 逐条复现脚本见各节的"证据"引用（Cholesky 非对称、`1e-16*A` 误判奇异、
  `|b|=1e-6` 假收敛、`max_iter=0` 后端不一致、零对角产出 inf/nan）

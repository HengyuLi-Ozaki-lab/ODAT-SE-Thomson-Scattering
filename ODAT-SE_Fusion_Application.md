# ODAT-SE 在核聚变等离子体诊断中的应用：基于贝叶斯推断的Thomson散射光谱逆问题求解

---

## 目录

1. [研究背景与动机](#1-研究背景与动机)
2. [ODAT-SE 软件包功能分析](#2-odat-se-软件包功能分析)
3. [ICDDPS 会议主题契合度分析](#3-icddps-会议主题契合度分析)
4. [针对问题：聚变等离子体Thomson散射诊断的逆问题](#4-针对问题聚变等离子体thomson散射诊断的逆问题)
5. [选择 ODAT-SE 的理由](#5-选择-odat-se-的理由)
6. [理论框架](#6-理论框架)
7. [具体实现方案](#7-具体实现方案)
8. [实例演示：合成数据验证](#8-实例演示合成数据验证)
9. [预期成果与意义](#9-预期成果与意义)
10. [总结与展望](#10-总结与展望)
11. [参考文献](#11-参考文献)

---

## 1. 研究背景与动机

### 1.1 核聚变研究中的数据驱动方法

核聚变研究正步入一个以数据密集型科学为特征的新阶段。从 ITER 到各类仿星器与托卡马克装置，实验产生的诊断数据体量呈指数增长，传统的基于手动拟合和简单最小二乘的数据分析方法已难以满足需求。近年来，**数据驱动的等离子体科学 (Data-Driven Plasma Science)** 已发展为一个独立的研究方向，其核心目标包括：

- 从复杂的实验诊断数据中精确推断等离子体参数
- 对推断结果进行系统的不确定性量化 (Uncertainty Quantification, UQ)
- 基于贝叶斯推断的模型选择与验证
- 加速数值模拟与实验优化的闭环反馈

这些方向与即将于2026年8月在德国基尔大学举办的 **第七届国际数据驱动等离子体科学会议 (ICDDPS-7)** 的核心主题高度一致。

### 1.2 逆问题在聚变诊断中的核心地位

聚变等离子体诊断的本质是**逆问题 (inverse problem)**：我们无法直接测量等离子体内部的温度、密度等物理量，而是通过测量辐射光谱、散射信号、磁场分布等"可观测量"，反推出等离子体的真实物理状态。这类逆问题通常具有以下特征：

- **非线性**：正向模型（物理量→信号）通常是强非线性的
- **病态性 (ill-posedness)**：不同参数组合可能产生相似信号
- **噪声影响**：实验数据不可避免地受到光子噪声、电子噪声等干扰
- **多模态性**：损失函数可能存在多个局域最小值

因此，需要一个通用的、模块化的逆问题求解框架来系统性地处理这些挑战。

### 1.3 ODAT-SE 与聚变研究的已有联系

值得注意的是，ODAT-SE (原名 2DMAT) 已被日本磁约束聚变研究者直接应用于 **Thomson 散射诊断的数据分析**。在 Compact Helical Device (CHD) 的 Thomson 散射系统概念设计中，研究者使用了 ODAT-SE 中实现的 Population Annealing Monte Carlo (PAMC) 采样方法进行贝叶斯推断，从合成光谱中反演电子温度和密度参数 (Morishita et al., arXiv:2511.06330, 2025)。此外，在非相干 Thomson 散射蒙特卡洛模拟方法的开发中 (arXiv:2508.20627, 2025)，ODAT-SE 同样被用作参数反演的核心工具。这些先驱性工作证明了将 ODAT-SE 拓展到更广泛的核聚变诊断应用中的可行性和重要价值。

---

## 2. ODAT-SE 软件包功能分析

### 2.1 概述

**ODAT-SE (Open Data Analysis Tool for Science and Engineering)** 是由东京大学物性研究所开发的开源数据分析框架（GitHub: [issp-center-dev/ODAT-SE](https://github.com/issp-center-dev/ODAT-SE)，文档: [ODAT-SE Manual](https://issp-center-dev.github.io/ODAT-SE/manual/main/en/index.html)）。其核心设计理念是：**将搜索/优化算法与正向问题求解器 (Direct Problem Solver) 解耦，形成模块化的逆问题求解平台**。

> 引用：Y. Motoyama, K. Yoshimi, I. Mochizuki, H. Iwamoto, H. Ichinose, and T. Hoshi,
> "Data-analysis software framework 2DMAT and its application to experimental measurements for two-dimensional material structures,"
> *Computer Physics Communications*, **280**, 108465 (2022). [DOI: 10.1016/j.cpc.2022.108465](https://doi.org/10.1016/j.cpc.2022.108465)

### 2.2 核心架构

ODAT-SE 的架构可概括为以下三层模块：

```
┌────────────────────────────────────────────────────┐
│                   ODAT-SE 框架                      │
├─────────────────────┬──────────────────────────────┤
│  搜索算法层 (Algorithm) │  正向求解器层 (Solver)      │
│                     │                              │
│  • Nelder-Mead 优化  │  • 用户自定义 Solver         │
│  • Grid Search      │  • TRHEPD/RHEED 求解器       │
│  • Bayesian 优化    │  • SXRD 求解器              │
│  • Replica Exchange │  • LEED 求解器              │
│    Monte Carlo      │  • (拓展: Thomson散射求解器)  │
│  • Population       │                              │
│    Annealing MC     │                              │
├─────────────────────┴──────────────────────────────┤
│              损失函数 / 目标函数                      │
│     L(θ) = Σ [Y_exp(λ) - Y_model(λ; θ)]²          │
└────────────────────────────────────────────────────┘
```

### 2.3 五种内置搜索算法

| 算法 | 类型 | 并行化 | 特点 | 适用场景 |
|------|------|--------|------|----------|
| **Nelder-Mead** (`minsearch`) | 局部优化 | 否 | 无需梯度，快速收敛 | 初步参数估计，低维问题 |
| **Grid Search** (`mapper`) | 全局探索 | 是 | 穷举，全面覆盖 | 低维参数空间可视化 |
| **Bayesian Optimization** (`bayes`) | 全局优化 | 部分 | 高斯过程代理模型 | 高计算代价正向模型 |
| **Replica Exchange MC** (`exchange`) | 全局采样 | 是 | 并行回火，克服能垒 | 多模态后验分布 |
| **Population Annealing MC** (`pamc`) | 全局采样 | 是 | 种群退火，计算自由能 | 贝叶斯推断，模型选择 |

### 2.4 模块化设计的关键优势

ODAT-SE 自 v3.0 起采用了完全模块化的架构。用户自定义正向求解器只需要实现一个 Python 类，该类接收参数向量并返回损失函数值。这一设计使得：

- 任何物理领域的正向模型都可以无缝接入
- 搜索算法与物理问题完全解耦
- 同一物理问题可以方便地比较不同算法的性能
- 支持超算级别的 MPI 并行计算

---

## 3. ICDDPS 会议主题契合度分析

### 3.1 ICDDPS-7 概况

**第七届国际数据驱动等离子体科学会议 (ICDDPS-7)** 将于 **2026年8月3-7日** 在德国基尔大学举行 ([官网](https://www.icddps.org/))。会议旨在汇聚全球利用 AI/ML 和数据科学方法推进等离子体科学的研究者，涵盖核聚变与等离子体加工等广泛主题。

会议特邀报告涵盖：
- **Jonathan Citrin** (Google DeepMind): TORAX—基于 JAX 的可微分托卡马克输运模拟器
- **Lu Lu** (Yale University): 函数空间上的算子学习与扩散模型
- **Siddhartha Mishra** (ETH Zurich): 物理中的数据驱动 AI 模拟

### 3.2 本提案与会议主题的契合点

根据 ICDDPS 历届会议的主题范围（参考 ICDDPS-4/5 的 Special Topic in *Physics of Plasmas*），本提案的契合点包括：

| ICDDPS 主题方向 | 本提案对应内容 |
|-----------------|--------------|
| 数据驱动的等离子体诊断 | Thomson 散射光谱的贝叶斯反演 |
| 机器学习/贝叶斯方法在聚变中的应用 | PAMC 采样、贝叶斯模型选择 |
| 开源工具与可复现研究 | ODAT-SE 开源框架的跨领域拓展 |
| 不确定性量化 | 后验分布与参数不确定性分析 |
| 代理模型与优化 | 正向模型的模块化与算法比较 |
| 非Maxwell分布函数的推断 | 非平衡态电子速度分布函数识别 |

---

## 4. 针对问题：聚变等离子体Thomson散射诊断的逆问题

### 4.1 Thomson散射的物理原理

Thomson 散射是磁约束聚变等离子体中测量电子温度 $T_e$ 和电子密度 $n_e$ 最可靠的诊断方法之一。其基本过程是：高功率脉冲激光注入等离子体后，自由电子对入射光子产生弹性散射。散射光谱的**多普勒展宽**反映了电子速度分布函数 (EVDF)，从而包含了 $T_e$ 和 $n_e$ 的信息。

对于**非相干 Thomson 散射**（当散射参数 $\alpha = 1/(k \lambda_D) \ll 1$），散射功率谱密度为：

$$
\frac{dP_s}{d\Omega d\lambda} \propto n_e \cdot r_e^2 \cdot S(\mathbf{k}, \omega)
$$

其中 $S(\mathbf{k}, \omega)$ 是电子的动态结构因子 (dynamic form factor)。对于各向同性的 Maxwell 分布：

$$
S(\mathbf{k}, \omega) \propto \frac{1}{k v_{th,e}} \exp\left(-\frac{m_e c^2}{2 k_B T_e} \cdot \frac{(\lambda - \lambda_0)^2}{\lambda_0^2}\right)
$$

其中 $v_{th,e} = \sqrt{k_B T_e / m_e}$ 是电子热速度，$\lambda_0$ 是入射激光波长。在相对论性温度下（$T_e > 1$ keV），需要使用 Selden 函数进行修正。

### 4.2 逆问题的数学表述

设多色仪系统有 $N_{ch}$ 个光谱通道，每个通道测量的信号为 $S_i^{exp}$（$i = 1, ..., N_{ch}$），正向模型预测的信号为 $S_i^{model}(\theta)$，其中参数向量 $\theta = (T_e, n_e, ...)$。逆问题可以表述为最小化加权残差：

$$
\chi^2(\theta) = \sum_{i=1}^{N_{ch}} \frac{[S_i^{exp} - S_i^{model}(\theta)]^2}{\sigma_i^2}
$$

其中 $\sigma_i$ 是第 $i$ 个通道的测量不确定性。在贝叶斯框架中，我们寻求后验分布：

$$
P(\theta | D) \propto P(D | \theta) \cdot P(\theta) = \mathcal{L}(\theta) \cdot \pi(\theta)
$$

其中 $\mathcal{L}(\theta) = \exp(-\chi^2 / 2)$ 是似然函数，$\pi(\theta)$ 是先验分布。

### 4.3 本实例选择的具体问题

我们选择以下具有代表性的问题进行 ODAT-SE 演示：

**从多色仪 Thomson 散射信号中，利用 ODAT-SE 的多种搜索/采样算法反演电子温度和密度，并进行算法比较与不确定性量化。**

具体包括：
1. 构建 Thomson 散射正向模型作为 ODAT-SE 的自定义 Solver
2. 生成包含噪声的合成诊断数据
3. 使用 ODAT-SE 的 5 种算法分别进行参数反演
4. 比较各算法的收敛速度、精度和计算效率
5. 利用 PAMC 方法获取完整的后验分布
6. 演示贝叶斯模型选择（Maxwell vs. non-Maxwell EVDF）

---

## 5. 选择 ODAT-SE 的理由

### 5.1 与现有工具的比较

| 特性 | ODAT-SE | 自编 Python 脚本 | Optuna/scikit-optimize | MINERVA (聚变) |
|------|---------|----------------|----------------------|----------------|
| 模块化正向求解器 | ✅ 原生支持 | ❌ 需手动集成 | ❌ 不支持 | ✅ 专用 |
| 多算法比较 | ✅ 5种内置 | ❌ 需逐一实现 | ⚠️ 部分支持 | ❌ 固定方法 |
| MPI 并行计算 | ✅ 原生支持 | ❌ 需自行实现 | ❌ 有限 | ✅ 支持 |
| 贝叶斯采样 | ✅ REMC + PAMC | ❌ 需集成 emcee 等 | ❌ 仅优化 | ✅ Nested Sampling |
| 自由能/模型证据 | ✅ PAMC 内置 | ❌ 需额外计算 | ❌ 不支持 | ⚠️ 部分支持 |
| 开源可复现 | ✅ MPL-2.0 | 取决于作者 | ✅ | ⚠️ 部分开源 |
| 跨领域通用性 | ✅ 设计目标 | ❌ 问题特定 | ✅ 通用 | ❌ 聚变专用 |
| 已有聚变应用先例 | ✅ CHD/LHD | - | - | ✅ ASDEX-U |

### 5.2 核心优势总结

1. **已验证的聚变应用**：ODAT-SE 已在日本 CHD 项目的 Thomson 散射分析中得到成功应用，由 JST Moonshot 项目支持
2. **算法多样性**：在同一框架下对比 5 种算法，为核聚变诊断社区提供 benchmark
3. **贝叶斯模型选择**：PAMC 方法可直接计算自由能（模型证据），支持物理模型的定量比较——这在区分 Maxwell 与 non-Maxwell EVDF 时至关重要
4. **超算级别可扩展性**：PAMC 和 REMC 的并行实现已在 Fugaku 等超级计算机上验证
5. **跨领域方法论迁移**：展示材料科学工具在聚变领域的成功迁移，符合 ICDDPS 促进学科交叉的宗旨

---

## 6. 理论框架

### 6.1 正向模型：Thomson 散射光谱计算

对于具有 $N_{ch}$ 个光谱通道的多色仪系统，每个通道 $i$ 的期望光电子数为：

$$
S_i(\theta) = n_e \cdot C_{cal} \cdot \int_{\lambda_{i,min}}^{\lambda_{i,max}} T_i(\lambda) \cdot \frac{d\sigma_{TS}}{d\Omega d\lambda}(\lambda; T_e) \, d\lambda
$$

其中：
- $T_i(\lambda)$ 是第 $i$ 个通道的滤波器透射函数
- $\frac{d\sigma_{TS}}{d\Omega d\lambda}$ 是微分 Thomson 散射截面
- $C_{cal}$ 是系统标定常数（由 Raman 散射标定确定）
- $\theta = (T_e, n_e)$ 是待反演参数

对于非相对论性情况（$T_e \lesssim 1$ keV），散射截面的谱形为高斯函数。对于相对论性温度，采用 Selden 近似或完整的 Naito-Kato-Nurunaga 函数。

### 6.2 噪声模型

Thomson 散射信号的主要噪声源包括：

- **光子噪声** (Poisson statistics): $\sigma_{photon,i} = \sqrt{S_i}$
- **电子学噪声** (Gaussian): $\sigma_{elec,i}$ = 常数
- **杂散光** (stray light): $S_{stray,i}$

总不确定性：$\sigma_i = \sqrt{S_i + S_{stray,i} + \sigma_{elec,i}^2}$

### 6.3 ODAT-SE 中的贝叶斯推断

将上述正向模型封装为 ODAT-SE 的 Solver 后，损失函数定义为：

$$
f(\theta) = \sum_{i=1}^{N_{ch}} \frac{[S_i^{obs} - S_i(\theta)]^2}{\sigma_i^2}
$$

在 ODAT-SE 的 PAMC 算法中，这被解释为"能量函数" $E(\theta) = f(\theta)/2$，温度参数 $\beta$ 对应贝叶斯推断中的似然精度。PAMC 通过逆温度序列 $\beta_0 < \beta_1 < ... < \beta_M = 1$ 逐步从先验分布冷却到后验分布，同时计算配分函数之比，得到**自由能**（即负对数模型证据的估计）：

$$
\ln Z = \ln \int \exp(-\beta E(\theta)) \pi(\theta) d\theta
$$

这使得不同 EVDF 模型之间的贝叶斯模型选择成为可能。

---

## 7. 具体实现方案

### 7.1 实现流程总览

```
Phase 1: 正向模型构建
    ├── 实现 Thomson 散射谱计算函数
    ├── 实现多色仪响应模型
    └── 封装为 ODAT-SE Solver 类

Phase 2: 合成数据生成
    ├── 设定真实参数 (T_e, n_e)
    ├── 计算理论信号
    └── 添加符合统计特性的噪声

Phase 3: ODAT-SE 参数反演
    ├── (a) Nelder-Mead 快速优化
    ├── (b) Grid Search 参数空间可视化
    ├── (c) Bayesian Optimization
    ├── (d) Replica Exchange Monte Carlo
    └── (e) Population Annealing Monte Carlo

Phase 4: 分析与比较
    ├── 反演精度比较
    ├── 计算效率比较
    ├── 后验分布分析
    └── 贝叶斯模型选择演示

Phase 5: 成果展示
    ├── 算法 benchmark 图表
    ├── 后验分布可视化
    └── 模型选择结果
```

### 7.2 自定义 Solver 实现

根据 ODAT-SE 的模板设计 ([ODAT-SE-template](https://github.com/issp-center-dev/ODAT-SE-template))，自定义 Solver 需要继承 `odatse.solver.SolverBase` 并实现 `evaluate` 方法。以下是核心代码框架：

```python
# ts_solver.py - Thomson Scattering Solver for ODAT-SE

import numpy as np
import odatse

class ThomsonScatteringSolver(odatse.solver.SolverBase):
    """
    ODAT-SE Solver for Thomson scattering spectral fitting.
    
    Forward model: Calculates expected photoelectron signals 
    in polychromator channels given (T_e, n_e).
    """
    
    def __init__(self, info):
        super().__init__(info)
        # 加载多色仪配置
        self.n_channels = info.solver.get("n_channels", 5)
        self.lambda_laser = info.solver.get("lambda_laser", 1064.0)  # nm
        self.filter_functions = self._load_filters(info)
        self.observed_data = self._load_data(info)
        self.sigma = self._load_uncertainties(info)
        
    def evaluate(self, x, args=(), nprocs=1, nthreads=1):
        """
        Evaluate the loss function for given parameters.
        
        Parameters
        ----------
        x : np.ndarray
            Parameter vector [T_e (eV), n_e (1e19 m^-3)]
            
        Returns
        -------
        float
            Chi-squared loss value
        """
        Te = x[0]   # Electron temperature in eV
        ne = x[1]   # Electron density in 1e19 m^-3
        
        # 计算理论 Thomson 散射光谱
        model_signals = self._compute_ts_spectrum(Te, ne)
        
        # 计算 chi-squared
        residuals = (self.observed_data - model_signals) / self.sigma
        chi2 = np.sum(residuals**2)
        
        return chi2
    
    def _compute_ts_spectrum(self, Te, ne):
        """
        Compute Thomson scattering signals in each 
        polychromator channel.
        """
        signals = np.zeros(self.n_channels)
        
        # 波长网格
        dlambda_max = 5.0 * self.lambda_laser * np.sqrt(
            2.0 * Te / (511.0e3)  # Te in eV, m_e c^2 = 511 keV
        )
        wavelengths = np.linspace(
            self.lambda_laser - dlambda_max,
            self.lambda_laser + dlambda_max,
            1000
        )
        
        # Thomson 散射谱形 (非相对论近似)
        delta_lambda = wavelengths - self.lambda_laser
        sigma_lambda = self.lambda_laser * np.sqrt(
            2.0 * Te / (511.0e3)
        )
        spectrum = np.exp(-0.5 * (delta_lambda / sigma_lambda)**2) \
                   / (sigma_lambda * np.sqrt(2.0 * np.pi))
        
        # 对每个通道积分
        for i in range(self.n_channels):
            filter_response = self.filter_functions[i](wavelengths)
            signals[i] = ne * np.trapz(
                spectrum * filter_response, wavelengths
            )
        
        return signals
```

### 7.3 ODAT-SE 配置文件

以下是 TOML 格式的输入配置文件示例（用于 PAMC 算法）：

```toml
[base]
dimension = 2
output_dir = "output_pamc"

[solver]
name = "thomson_scattering"
n_channels = 5
lambda_laser = 1064.0
reference_file = "synthetic_data.dat"
uncertainty_file = "uncertainties.dat"

[solver.param]
string_list = ["Te", "ne"]
min_list = [10.0, 0.1]     # T_e: 10 eV,    n_e: 0.1×10^19 m^-3
max_list = [10000.0, 10.0]  # T_e: 10000 eV, n_e: 10×10^19 m^-3
unit_list = ["eV", "1e19m-3"]

[algorithm]
name = "pamc"
seed = 12345

[algorithm.pamc]
numsteps_annealing = 10
bmin = 0.01
bmax = 100.0
Tnum = 21           # 逆温度分点数
nreplica_per_proc = 20

[runner]
[runner.log]
interval = 10
```

### 7.4 运行与分析脚本

```python
# run_benchmark.py - 运行所有5种算法的benchmark

import odatse
import numpy as np
import matplotlib.pyplot as plt
from ts_solver import ThomsonScatteringSolver

# ---- 合成数据生成 ----
def generate_synthetic_data(Te_true, ne_true, n_channels=5, SNR=20):
    """生成合成Thomson散射数据"""
    solver = ThomsonScatteringSolver(config)
    true_signals = solver._compute_ts_spectrum(Te_true, ne_true)
    noise = true_signals / SNR * np.random.randn(n_channels)
    observed = true_signals + noise
    sigma = true_signals / SNR
    return observed, sigma

# ---- 算法比较 ----
algorithms = ["minsearch", "mapper", "bayes", "exchange", "pamc"]
results = {}

for algo in algorithms:
    config = odatse.Info.from_file(f"input_{algo}.toml")
    solver = ThomsonScatteringSolver(config)
    runner = odatse.Runner(solver, config)
    alg = odatse.Algorithm.from_name(algo, config, runner)
    alg.main()
    results[algo] = alg.result

# ---- 后验分布可视化 (PAMC结果) ----
samples = np.loadtxt("output_pamc/result.txt")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(samples[:, 0], bins=50, density=True)
axes[0].axvline(Te_true, color='r', linestyle='--')
axes[0].set_xlabel("T_e (eV)")
axes[1].hist(samples[:, 1], bins=50, density=True)
axes[1].axvline(ne_true, color='r', linestyle='--')
axes[1].set_xlabel("n_e (10^19 m^-3)")
plt.savefig("posterior_distribution.png")
```

### 7.5 贝叶斯模型选择实现

```python
# model_selection.py - Maxwell vs. non-Maxwell EVDF 比较

import numpy as np

class MaxwellSolver(ThomsonScatteringSolver):
    """Maxwell分布模型: 参数 θ = (T_e, n_e)"""
    def _evdf(self, v, params):
        Te = params[0]
        vth = np.sqrt(2 * Te * 1.602e-19 / 9.109e-31)
        return np.exp(-v**2 / vth**2) / (np.pi**1.5 * vth**3)

class KappaSolver(ThomsonScatteringSolver):
    """Kappa分布模型: 参数 θ = (T_e, n_e, κ)
    
    适用于含有高能尾部的非平衡态等离子体
    """
    def _evdf(self, v, params):
        Te, _, kappa = params[:3]
        vth = np.sqrt(2 * Te * 1.602e-19 / 9.109e-31)
        from scipy.special import gamma
        A = (1.0 / (np.pi * kappa * vth**2)**1.5) \
            * gamma(kappa + 1) / gamma(kappa - 0.5)
        return A * (1 + v**2 / (kappa * vth**2))**(-(kappa + 1))

# 分别对两个模型运行PAMC
# PAMC 输出包含自由能估计 F = -ln(Z)
# 贝叶斯因子: ln(B_12) = F_2 - F_1
# |ln(B)| > 5: 强证据; |ln(B)| > 10: 决定性证据
```

---

## 8. 实例演示：合成数据验证

### 8.1 实验设置

我们模拟一个典型的托卡马克 Thomson 散射系统：

| 参数 | 值 |
|------|-----|
| 激光波长 | 1064 nm (Nd:YAG) |
| 多色仪通道数 | 5 |
| 真实电子温度 | $T_e = 500$ eV |
| 真实电子密度 | $n_e = 3 \times 10^{19}$ m$^{-3}$ |
| 信噪比 (SNR) | 20 (典型值) |
| 参数搜索范围 | $T_e \in [10, 5000]$ eV, $n_e \in [0.1, 10] \times 10^{19}$ m$^{-3}$ |

### 8.2 预期结果展示

**结果 1: Grid Search 损失函数地形图**

Grid Search 将在 $T_e$-$n_e$ 平面上给出完整的 $\chi^2$ 地形图，可视化参数相关性和多模态结构。

**结果 2: 算法收敛比较表**

| 算法 | 反演 $T_e$ (eV) | 反演 $n_e$ | 函数评估次数 | 计算时间 |
|------|----------------|-----------|-------------|---------|
| Nelder-Mead | ~500 ± δ | ~3.0 ± δ | ~100 | 最短 |
| Grid Search | 500 (格点) | 3.0 (格点) | 10000+ | 中等 |
| Bayesian Opt. | ~500 ± δ | ~3.0 ± δ | ~50 | 中等 |
| REMC | 后验分布 | 后验分布 | ~5000 | 较长 |
| PAMC | 后验分布 + $F$ | 后验分布 + $F$ | ~10000 | 最长 |

**结果 3: PAMC 后验分布**

PAMC 提供完整的后验概率密度函数，包括 $T_e$ 和 $n_e$ 的边缘分布以及二者的联合分布（显示参数相关性）。

**结果 4: 模型选择**

当真实 EVDF 为 Maxwell 分布时，PAMC 计算的 Bayes 因子应给出 $\ln(B_{Maxwell/Kappa}) > 5$，明确支持 Maxwell 模型。

---

## 9. 预期成果与意义

### 9.1 科学意义

1. **方法论验证**：首次在同一开源框架下系统比较 5 种逆问题求解算法在聚变 Thomson 散射诊断中的性能
2. **跨领域迁移**：展示材料科学数据分析工具在聚变领域的成功迁移，促进领域间的方法论交流
3. **贝叶斯模型选择**：为聚变等离子体中非平衡态 EVDF 的识别提供定量工具，这对于理解粒子加热、电流驱动等物理过程至关重要

### 9.2 技术贡献

1. **可复用的 Thomson 散射 Solver 模块**：作为 ODAT-SE 生态系统的组件，供聚变社区直接使用
2. **算法选择指南**：为不同计算预算和精度需求下选择最优算法提供参考
3. **从合成数据到实验数据的过渡路径**：框架设计确保可以无缝接入真实实验数据

### 9.3 对 ICDDPS-7 的贡献

- 展示开源数据分析工具在聚变诊断中的新应用范式
- 贡献可复现的 benchmark 案例，供社区验证和扩展
- 促进材料科学与聚变科学之间的数据分析方法论交流

---

## 10. 总结与展望

本提案设计了一个基于 **ODAT-SE** 开源框架的核聚变等离子体 **Thomson 散射诊断逆问题求解实例**。该实例将 ODAT-SE 的模块化逆问题求解能力与核聚变 Thomson 散射诊断的实际需求结合，通过：

1. 构建 Thomson 散射正向模型作为 ODAT-SE 自定义 Solver
2. 系统比较 5 种搜索/采样算法在参数反演中的性能
3. 利用 PAMC 方法进行贝叶斯推断和模型选择

展示了 ODAT-SE 在核聚变领域的应用潜力。该实例具有**理论基础扎实、实现路径清晰、快速可实现（基于合成数据无需实验设备）、应用意义明确**的特点，完全契合 ICDDPS-7 会议关于数据驱动等离子体科学的主题。

**未来展望**：

- 接入真实 Thomson 散射实验数据（如 LHD、ASDEX-U 等装置）
- 扩展到其他聚变诊断（电子回旋辐射 ECE、软X射线断层重建等）
- 结合神经网络代理模型加速正向计算
- 与 TORAX 等可微分模拟器联合构建集成数据分析 (IDA) 流程

---

## 11. 参考文献

1. Y. Motoyama, K. Yoshimi, I. Mochizuki, H. Iwamoto, H. Ichinose, and T. Hoshi, "Data-analysis software framework 2DMAT and its application to experimental measurements for two-dimensional material structures," *Computer Physics Communications*, **280**, 108465 (2022). [DOI: 10.1016/j.cpc.2022.108465](https://doi.org/10.1016/j.cpc.2022.108465)

2. K. Saito et al., "Conceptual design of Thomson scattering system with high wavelength resolution in magnetically confined plasmas for electron phase-space measurements," arXiv:2511.06330 (2025). [Link](https://arxiv.org/abs/2511.06330)

3. Y. Morishita et al., "Monte Carlo simulation method for incoherent Thomson scattering spectra from arbitrary electron distribution functions," arXiv:2508.20627 (2025). [Link](https://arxiv.org/abs/2508.20627)

4. R. Fischer, A. Dinklage, and E. Pasch, "Bayesian modelling of fusion diagnostics," *Plasma Physics and Controlled Fusion*, **45**, 1095-1111 (2003).

5. A. Pavone, A. Merlo, S. Kwak and J. Svensson, "Machine learning and Bayesian inference in nuclear fusion research: an overview," *Plasma Physics and Controlled Fusion*, **65**, 053001 (2023). [DOI: 10.1088/1361-6587/acc60f](https://doi.org/10.1088/1361-6587/acc60f)

6. J. B. Greenwald et al., "Bayesian plasma model selection for Thomson scattering," *Review of Scientific Instruments*, **95**, 043004 (2024). [DOI: 10.1063/5.0158749](https://doi.org/10.1063/5.0158749)

7. ODAT-SE GitHub Repository: [https://github.com/issp-center-dev/ODAT-SE](https://github.com/issp-center-dev/ODAT-SE)

8. ODAT-SE Documentation: [https://issp-center-dev.github.io/ODAT-SE/manual/main/en/index.html](https://issp-center-dev.github.io/ODAT-SE/manual/main/en/index.html)

9. ICDDPS-7 Conference: [https://www.icddps.org/](https://www.icddps.org/)

10. Y. Morishita, S. Murakami, N. Kenmochi et al., "Bayesian inference of electron temperature and density from Thomson scattering in LHD," *Scientific Reports*, **14**, 137 (2024).

---

*本文档最后更新：2026年4月*

*作者注：本提案中的代码框架为概念性示例，完整可运行的实现需根据 ODAT-SE v3.x 的最新 API 进行调整。建议参考 [ODAT-SE-template](https://github.com/issp-center-dev/ODAT-SE-template) 获取最新的 Solver 开发模板。*

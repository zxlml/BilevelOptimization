<div align="center">

[English](./README.md) | **简体中文**

</div>

<h1 align="center">BilevelOptimization：随机双层优化的稳定性与泛化精细化分析</h1>

<p align="center">
  <a href="https://www.ijcai.org/proceedings/2024/609">
    <img src="https://img.shields.io/badge/IJCAI_2024-Paper-red" alt="IJCAI 2024">
  </a>
  <a href="https://github.com/zxlml/BilevelOptimization">
    <img src="https://img.shields.io/github/stars/zxlml/BilevelOptimization?style=social" alt="GitHub Stars">
  </a>
  <img src="https://img.shields.io/github/last-commit/zxlml/BilevelOptimization?color=blue" alt="Last Commit">
  <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs Welcome">
</p>

<h5 align="center"> ⭐ 如果您喜欢我们的项目，请在 GitHub 上给我们点一个 Star 以获取最新更新！ ⭐ </h5>

---

**BilevelOptimization** 是 IJCAI 2024 论文《*Fine-grained Analysis of Stability and Generalization for Stochastic Bilevel Optimization*》的官方实现。本项目提供一阶随机双层优化（SBO）求解器，以及一套将算法的**平均参数稳定性（on-average argument stability）**与**泛化 GAP** 联系起来的实验验证框架，在"数据重加权式超参数优化"任务上完整复现了论文的关键结论。

* 🧮 **论文三类算法完整实现**
  <br> SSGD（Algorithm 1）、带 warm-start 的 TSGD（Algorithm 2）、带重初始化的 UD（Algorithm 3），统一遵循论文口径（K = 外层迭代数，T = 内层迭代数）。

* 📈 **泛化 GAP 评估**
  <br> 全量验证/测试误差及其偏离度（Eq. 2），支持在不同样本量、迭代次数、学习率下的全程跟踪。

* 🔬 **稳定性实测**
  <br> 按 Definition 5 实测 l2 平均参数稳定性：替换单个验证样本后成对重跑并度量参数漂移，验证 O(K/m1) 理论界（Table 1）。

* ⚙️ **与论文一致的实验流水线**
  <br> 50% 标签损坏的 MNIST 重加权任务、784/256/10 MLP、batch size 8、步长 0.01（内层）/ 5（外层）、多 seed 平均并输出 CSV 与曲线图。

<span id='news'/>

## 📢 动态

- **[2026-09-24]**: 🚀 代码大规模重构发布：对齐论文的 K/T 迭代口径、SSGD/TSGD/UD 三算法实现、修正泛化 GAP 估计、新增稳定性实测、多 seed 实验驱动器，以及 10 个单元测试用例。
- **[2024-08]**: 🎉🎉🎉 论文被 **IJCAI 2024** 接收（论文集第 5508–5516 页）。
- **[2024]**: 🚀 初始代码发布（超参数优化-数据重加权与元学习任务）。

<span id='contents'/>

## 📑 目录

* <a href='#algorithms'>🧮 算法</a>
* <a href='#installation'>🔧 安装</a>
* <a href='#quickstart'>⚡ 快速开始</a>
* <a href='#structure'>📁 项目结构</a>
* <a href='#results'>📊 实验结果</a>
* <a href='#tests'>🧪 单元测试</a>
* <a href='#todo'>☑️ 待办清单</a>
* <a href='#citation'>📚 引用</a>
* <a href='#acknowledgement'>🔗 致谢</a>

<span id='algorithms'/>

## 🧮 算法

论文研究如下随机双层优化问题：

```
min_{x}  R(x) = F(x, y*(x)) = E_{ξ}[f(x, y*(x); ξ)]
s.t.     y*(x) = argmin_{y} G(x, y) = E_{ζ}[g(x, y; ζ)]
```

并针对一阶梯度方法给出稳定性与泛化分析。项目在 `core/ablo.py` 中提供三种求解器：

| 算法 | 论文 | 关键性质 | 配置 |
| :--- | :--- | :--- | :--- |
| **SSGD** | Algorithm 1 | 单时间尺度，每轮"一次内层 + 一次外层"更新（`T = 1`） | `ho_algo='ssgd'` / `T=1` |
| **TSGD** | Algorithm 2 | 双时间尺度：每轮外层前做 `T` 步内层更新，**warm-start** `y0_{k+1} = y^T_k` | `ho_algo='tsgd'` |
| **UD** | Algorithm 3（Bao et al., 2021） | 双时间尺度，但每轮外层前**重初始化** `y0_k = y0` | `ho_algo='ud'` |

泛化 GAP（Eq. 2）由全量验证误差与测试误差之差估计；l2 平均参数稳定性（Definition 5）通过成对运行度量——即把某个验证样本替换为独立抽样后重跑算法，计算参数漂移。

<span id='installation'/>

## 🔧 安装

```bash
# 克隆仓库
git clone https://github.com/zxlml/BilevelOptimization.git
cd BilevelOptimization

# 创建虚拟环境
conda create -n sbo python=3.10 -y
conda activate sbo

# 安装依赖
pip install torch torchvision matplotlib pytest
```

MNIST 数据集会在首次运行时由 `torchvision` 自动下载（缓存于 `workspace/datasets/mnist`）。

<span id='quickstart'/>

## ⚡ 快速开始

所有实验由 `Hyperparameter_Optimization.py` 驱动（默认即为论文设置）：

```bash
# 1. 快速端到端自检（几秒内完成）
python Hyperparameter_Optimization.py --mode smoke

# 2. Fig.1 复现：变化内层迭代 T（1, 32），K = 5000，3 个 seed
python Hyperparameter_Optimization.py --mode fig1 --K_max 5000 --seeds 42 52 82

# 3. Fig.2 复现：SSGD（T = 1），变化验证集大小 m1（500, 2000），5 个 seed
python Hyperparameter_Optimization.py --mode fig2 --K_max 5000 --seeds 42 52 62 72 82

# 4. 学习率敏感性：SSGD 上 lr_l × lr_h 网格
python Hyperparameter_Optimization.py --mode lr --K_max 3000 --seeds 42 52 62

# 5. 实测平均参数稳定性（Definition 5）
python Hyperparameter_Optimization.py --mode stability --K_max 2000 --seeds 42

# 单次自定义运行（也支持 --mode all）
python core/run.py  # 完整参数见 Hyperparameter_Optimization.default_config()
```

每种模式都会在 `workspace/exp/<mode>_<timestamp>/` 下输出多 seed 平均曲线（`*_summary.csv`）与逐指标曲线图；验证/测试指标每 `--eval_every` 轮在**全量数据集**上评估一次。

<span id='structure'/>

## 📁 项目结构

```
BilevelOptimization/
├── Hyperparameter_Optimization.py   # 实验驱动器（smoke / fig1 / fig2 / lr / stability）
├── Meta_Learning.py                 # Omniglot 单样本学习（l2l，SSGD 与 TSGD 的适应步数）
├── core/
│   ├── ablo.py                      # SSGD / TSGD / UD 求解器 + 稳定性实测
│   ├── datasets.py                  # Corrupted MNIST（50% 标签损坏）与 Omniglot 生成器
│   ├── loss.py                      # ReweightingMLP / MLPFeatureLearning 双层损失
│   ├── mlp.py                       # MLP 参数构造与前向
│   ├── random_search.py             # 随机搜索基线
│   ├── run.py                       # 单次运行入口（日志、记录保存、绘图）
│   └── utils.py                     # 种子 / loader / CSV 导出 / 记录聚合工具
├── func/                            # 自动微分辅助（展开式微分、张量运算）
├── tests/                           # 单元测试（pytest）
├── results/                         # 论文一致设置的复现曲线
└── workspace/                       # （已 git-ignore）数据集、日志、运行输出
```

<span id='results'/>

## 📊 实验结果

复现曲线（多 seed 平均，符合论文第 5 节设置）见 [`results/`](./results/) 目录。原始汇总 CSV：[`results/fig1_summary.csv`](./results/fig1_summary.csv)、[`fig2_summary.csv`](./results/fig2_summary.csv)、[`lr_summary.csv`](./results/lr_summary.csv)、[`stability.csv`](./results/stability.csv)。

**Fig.1 复现 —— 变化 `T` 与 `K`（TSGD，3 个 seed，`K_max = 5000`）**

| 配置 | val @K=1000 | val @K=5000 | test @K=5000 | GAP @K=5000 |
| :--- | :--- | :--- | :--- | :--- |
| TSGD, T=1  | 1.559 | **1.138** | 1.144 | +0.006 |
| TSGD, T=32 | **1.188**（最小） | 2.037（过拟合） | 2.167 | **+0.130** |

→ 过大的 `K` 与 `T` 会因过拟合损害泛化能力；`T = 32` 时测试误差在 `K ≈ 1000` 后开始上升（论文：`K > 3000` 且 `T = 32` 出现过拟合）。

**Fig.2 复现 —— SSGD（`T = 1`）变化验证集大小 `m1`（5 个 seed，`K_max = 5000`）**

| 配置 | 验证误差 | 测试误差 | GAP | 0-1 测试误差 |
| :--- | :--- | :--- | :--- | :--- |
| m1 = 500  | 1.067 | 1.118 | **+0.051**（随 K 增长） | 0.222 |
| m1 = 2000 | 1.114 | **1.099** | **-0.015**（保持稳定） | 0.200 |

→ 增大验证集可改善泛化：小 `m1` 的 GAP 随 `K` 稳步增长，而大 `m1` 的 GAP 保持平稳（对应论文 Fig. 2）。

**学习率敏感性（SSGD，`K = 3000`，3 个 seed）**

| lr_l（内层） | 0-1 测试误差（lr_h = 0.5） | 0-1 测试误差（lr_h = 5.0） |
| :--- | :--- | :--- |
| 0.001（欠拟合） | 0.374 | 0.376 |
| 0.01（论文值）     | **0.181** | **0.182** |
| 0.1（过大）  | 0.338 | 0.317 |

→ 合适的学习率至关重要；论文设置 `lr_l = 0.01, lr_h = 5` 位于稳定最优区间。

**实测 l2 平均参数稳定性（SSGD，3 个替换验证样本）**

| m1 \ K | K = 1000 | K = 2000 |
| :--- | :--- | :--- |
| 500  | 2.25e-4 | 6.26e-4（×2.8） |
| 2000 | 1.17e-4 | 2.10e-4（×1.8） |

→ 参数漂移随 `K` 近似线性增长、随 `m1` 增大而缩小，与 Table 1 的 `O(K/m1)` 理论界一致。

**UD（重初始化）vs TSGD（warm-start），`T = 32`，`K = 2000`，2 个 seed**

| 算法 | 验证误差 | 测试误差 | 0-1 测试误差 |
| :--- | :--- | :--- | :--- |
| UD（Alg. 3）   | 2.258 | 2.262 | 0.791 |
| TSGD（Alg. 2） | **1.268** | **1.241** | **0.376** |

→ 内层重初始化浪费算力并导致严重欠拟合，这印证了论文对"内层参数连续更新"（SSGD/TSGD）方法进行分析的动机。

<span id='tests'/>

## 🧪 单元测试

```bash
python -m pytest tests/ -v
```

共 10 个测试用例，覆盖：单步梯度正确性、超梯度向外层变量的传播、UD 重初始化语义与 warm-start 的对比、双层训练收敛性、GAP 一致性、NaN 早停、多 seed 记录聚合、MNIST 50% 标签损坏，以及真实 MNIST 收敛冒烟测试。

<span id='todo'/>

## ☑️ 待办清单

- [ ] GPU 加速与混合精度支持
- [ ] 元学习（Omniglot）实验接入同样的多 seed 流水线
- [ ] 更多双层优化应用（数据清洗、核心集选择）
- [ ] 常数步长之外的自动步长调度

<span id='citation'/>

## 📚 引用

如果您对本工作感兴趣，请参考 [IJCAI 2024 论文集](https://www.ijcai.org/proceedings/2024/609) 并引用：

```bibtex
@inproceedings{zhang2024genbo,
  title     = {Fine-grained Analysis of Stability and Generalization for Stochastic Bilevel Optimization},
  author    = {Zhang, Xuelin and Chen, Hong and Gu, Bin and Gong, Tieliang and Zheng, Feng},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  pages     = {5508--5516},
  year      = {2024}
}
```

<span id='acknowledgement'/>

## 🔗 致谢

感谢以下项目，本项目的部分代码与数据流水线改编自它们：

* [stability_ho](https://github.com/baofff/stability_ho) — Stability and generalization of bilevel programming in hyperparameter optimization（NeurIPS 2021）
* [stocBiO](https://github.com/JunjieYang97/stocBiO) — Bilevel Optimization: Nonasymptotic Analysis and Faster Algorithms（ICML 2021）
* [MNIST](http://yann.lecun.com/exdb/mnist/) 与 [Omniglot](https://github.com/brendenlake/omniglot) 数据集

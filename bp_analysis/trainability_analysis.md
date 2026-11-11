# 量子比特复用型参数化量子线路的可训练性分析

> **中文版**

---

## 摘要

参数化量子线路（PQC）的可训练性是其在量子机器学习中实际应用的前提条件，而 Barren Plateau（BP）现象——即梯度方差随量子比特数指数衰减——是制约大规模 PQC 训练的核心瓶颈。本文针对一种量子比特复用型量子图卷积网络（Qubit-Reused QGCN）的核心 PQC 组件，系统评估其梯度方差行为。通过对比复用电路（固定 4 qubits）与无复用电路（2+2N qubits）在邻居数 N ∈ {1, 2, 3, 4, 5} 范围内的梯度方差，实验结果表明：（1）复用电路的梯度方差在所有配置下均保持在 $10^{-3}$ 量级，超出 Barren Plateau 消失阈值三个数量级以上；（2）复用电路与无复用电路的梯度方差几乎完全一致（比值 ≈ 1.0），证明测量-重置循环对反向传播透明；（3）N = 5 时，复用电路以 4 qubits 达成 12-qubit 的梯度质量，实现 **3:1 的硬件压缩比**。上述结果揭示了顺序纠缠架构将有效 Hilbert 空间维度约束在常数级的结构性特征，为该类 PQC 在资源受限场景下的部署提供了可训练性方面的实验依据。

**关键词**：参数化量子线路，可训练性，梯度方差，Barren Plateau，量子比特复用，量子图神经网络

---

## 1. 引言

变分量子算法（Variational Quantum Algorithms, VQAs）是当前含噪中等规模量子（NISQ）时代最具应用前景的量子-经典混合计算范式之一[1]。该类算法的核心组件——参数化量子线路（Parameterized Quantum Circuit, PQC）——通过在量子计算机上执行含可调参数的酉变换，将优化任务委托给经典优化器，从而在量子化学模拟、组合优化和量子机器学习等领域展现出广泛的应用潜力[2,3]。

然而，PQC 的训练面临着根本性的挑战。McClean 等人[4]首次指出，对于随机初始化的深度 PQC，其梯度方差随量子比特数 $n$ 呈指数衰减，即 $\mathrm{Var}(\nabla_\theta \mathcal{L}) \propto 1/2^n$。这一现象被称作 Barren Plateau（BP），它意味着在超过一定规模后，PQC 的梯度信号将淹没于量子测量噪声之中，导致经典优化器无法有效更新参数。此后，研究者从电路架构[5]、参数初始化策略[6]和代价函数设计[7]等多个角度探索 BP 缓解方案，但如何在保持电路表达能力的同时规避 BP，仍是该领域的核心未决问题。

量子图神经网络（Quantum Graph Neural Network, QGNN）是量子机器学习的一个重要子领域[8]，其目标是将经典图结构数据编码到量子态空间中，通过量子纠缠操作实现节点间的信息传递。Qubit-Reused QGCN 是该方向的一种新型架构，其核心创新在于**量子比特复用**：邻居量子比特在每次信息聚合后被测量并重置，从而允许固定数量的量子比特处理任意数量的邻居节点。这一设计在硬件资源效率上具有显著优势，但其可训练性——尤其是复用操作对梯度传播的影响——尚未经过系统的理论或实验检验。

本文聚焦于以下核心问题：**量子比特复用型 PQC 在训练过程中是否会出现梯度消失？其梯度方差行为与等价的非复用电路相比如何？**为此，我们设计了系统的梯度方差分析实验，在控制电路参数完全一致的条件下，对比复用 PQC（固定 4 qubits，邻居 qubits 经测量-重置循环复用）与非复用 PQC（2 + 2N qubits，邻居 qubits 不复用）在不同邻居规模下的梯度方差。实验结果为理解量子比特复用策略对 PQC 可训练性的影响提供了定量依据。

## 2. 量子电路设计

### 2.1 复用 PQC

复用 PQC 由 $n_{\text{center}} = 2$ 个中心量子比特和 $n_{\text{reuse}} = 2$ 个可复用邻居量子比特组成，总计 4 个量子比特。对于给定的邻居数 $N$，电路按如下顺序执行：

1. **中心嵌入**：将中心节点特征向量 $\mathbf{x}_c \in \mathbb{R}^4$ 通过 Amplitude Embedding 编码到中心量子比特（wire 0, 1）上，归一化条件保证 $\|\mathbf{x}_c\| = 1$。
2. **邻居嵌入**：对第 $i$ 个邻居（$i = 1, \dots, N$），将其特征向量 $\mathbf{x}_{n,i} \in \mathbb{R}^4$ 同样通过 Amplitude Embedding 编码到邻居量子比特（wire 2, 3）上。
3. **纠缠与信息聚合**：在全部 4 个量子比特上施加一层 StronglyEntanglingLayers（含 2 个 layers），实现中心节点与当前邻居之间的量子纠缠。该纠缠层的参数 $\boldsymbol{\theta}_k \in \mathbb{R}^{2 \times 4 \times 3}$ 在所有邻居间共享。
4. **条件旋转**：对邻居量子比特（wire 2, 3）执行投影测量，根据测量结果条件性地对中心量子比特施加 $R_Y$ 旋转门，旋转角度由参数矩阵 $\boldsymbol{\theta}_p \in \mathbb{R}^{2 \times 2}$ 决定。测量后邻居量子比特被重置（reset）为 $|0\rangle$ 态。
5. **复用**：重置后的邻居量子比特可供下一个邻居使用，回到步骤 2。此循环重复 $N$ 次。
6. **读出**：所有邻居处理完毕后，在中心量子比特上施加一层 StronglyEntanglingLayers（参数 $\boldsymbol{\theta}_l \in \mathbb{R}^{2 \times 2 \times 3}$），随后对中心量子比特执行概率测量，输出维度为 $2^2 = 4$ 的概率向量。

### 2.2 无复用 PQC（Baseline）

无复用 PQC 与复用 PQC 在**结构上完全等价**，唯一的区别在于量子比特的分配方式：

- 复用 PQC：邻居量子比特（wire 2, 3）在每次循环后被重置，供下一个邻居使用。量子比特总数恒为 4。
- 无复用 PQC：每个邻居分配 2 个全新的量子比特（wire $2 + 2(i-1)$, $3 + 2(i-1)$），处理完毕后虽执行测量和重置，但后续邻居不再使用这些量子比特。量子比特总数为 $2 + 2N$。

两个电路的关键等价性在于：**每次 StronglyEntanglingLayers 均作用于完全相同的 4-qubit 子空间**（中心 2 qubits + 当前邻居 2 qubits）。无复用电路中多余的量子比特始终处于 $|0\rangle$ 旁观态，不参与任何纠缠操作。两种电路的参数数量（共 63 个标量参数，含 kernel weights $2 \times 4 \times 3 = 36$，pooling weights $2 \times 2 = 4$，linear weights $2 \times 2 \times 3 = 18$，及 5 个不再用的冗余）完全一致，差异仅在于量子比特总数。

## 3. 实验方法

### 3.1 梯度方差估计

Barren Plateau 的核心诊断指标是梯度方差 $\mathrm{Var}(\nabla_\theta \mathcal{L})$。遵循 McClean 等人的方法[4]，我们对所有可训练参数 $\boldsymbol{\theta} = \{\boldsymbol{\theta}_k, \boldsymbol{\theta}_p, \boldsymbol{\theta}_l\}$ 的总梯度向量计算方差：

$$\mathrm{Var}_{\text{total}} = \mathbb{E}_{\boldsymbol{\theta} \sim \mathcal{U}, \mathbf{x} \sim \mathcal{N}} \left[ \frac{1}{|\boldsymbol{\theta}|} \sum_{j=1}^{|\boldsymbol{\theta}|} \left( \frac{\partial \mathcal{L}}{\partial \theta_j} - \bar{g} \right)^2 \right]$$

其中期望通过 $M = 100$ 次随机初始化取平均来近似，每次初始化独立采样 $\boldsymbol{\theta}$ 和输入特征 $\mathbf{x}$。损失函数为随机目标的均方误差（MSE）：

$$\mathcal{L} = \sum_{k=1}^{d_{\text{out}}} (p_k - t_k)^2, \quad \mathbf{t} \sim \mathcal{U}(0, 1)$$

其中 $\mathbf{p} = (p_1, \dots, p_{d_{\text{out}}})$ 为量子线路输出的概率向量，$d_{\text{out}} = 2^{n_{\text{center}}} = 4$。

### 3.2 实验参数配置

实验对 $N \in \{1, 2, 3, 4, 5\}$ 的每个值分别评估复用电路和无复用电路的梯度方差，配置如表 1 所示。

**表 1. 实验参数配置**

| 参数 | 取值 |
|:---|---:|
| 中心量子比特数 $n_{\text{center}}$ | 2 |
| 邻居量子比特池 $n_{\text{reuse}}$ | 2 |
| 邻居数 $N$ | 1, 2, 3, 4, 5 |
| 复用电路 qubit 总数 | 4（固定） |
| 无复用电路 qubit 总数 | $2 + 2N \in \{4, 6, 8, 10, 12\}$ |
| 纠缠层结构 | StronglyEntanglingLayers, $L = 2$ |
| 随机初始化次数 $M$ | 100 |
| 损失函数 | 随机目标 MSE |
| 模拟后端 | PennyLane `default.qubit`，反向传播微分 |

### 3.3 Barren Plateau 参考基线

为量化 Barren Plateau 缓解程度，我们定义 BP 缓解因子（BP Mitigation Factor）：

$$\eta_{\text{BP}}^{(N)} = \frac{\mathrm{Var}_{\text{measured}}^{(N)}}{\mathrm{Var}_{\text{BP}}^{(N)}}, \quad \mathrm{Var}_{\text{BP}}^{(N)} = \mathrm{Var}_{\text{measured}}^{(N=1)} \cdot \frac{2^{q_1}}{2^{q_N}}$$

其中 $q_N$ 为配置 $N$ 下的量子比特总数。$\eta_{\text{BP}} > 1$ 表示梯度方差高于同等规模随机 PQC 的理论预期，即存在 BP 缓解效应。该基线假设随机 PQC 的 $\mathrm{Var} \propto 1/2^q$，并锚定于 $N = 1$ 的实验值。

所有实验代码基于 Python 3.13、PennyLane 0.41 和 PyTorch 2.7 实现，在 CPU 环境下执行。量子线路模拟采用精确状态向量模拟（无采样噪声），以确保梯度方差估计的统计精度。

## 4. 实验结果

### 4.1 总体梯度方差

图 1 展示了复用电路与无复用电路在 $N = 1$ 至 $N = 5$ 范围内的总梯度方差。表 2 给出了对应的数值结果。

**表 2. 梯度方差对比结果（M = 100 次随机初始化）**

| $N$ | 复用 qubits | 无复用 qubits | $\mathrm{Var}_{\text{reuse}}$ | $\mathrm{Var}_{\text{normal}}$ | 比值 | $\eta_{\text{BP}}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 4 | 4 | $7.89 \times 10^{-3}$ | $7.89 \times 10^{-3}$ | $\approx 1.0$ | 1.0× |
| 2 | 4 | 6 | $4.12 \times 10^{-3}$ | $4.12 \times 10^{-3}$ | $\approx 1.0$ | 2.1× |
| 3 | 4 | 8 | $2.93 \times 10^{-3}$ | $2.93 \times 10^{-3}$ | $\approx 1.0$ | 5.9× |
| 4 | 4 | 10 | $2.05 \times 10^{-3}$ | $2.05 \times 10^{-3}$ | $\approx 1.0$ | 16.7× |
| 5 | 4 | 12 | $1.81 \times 10^{-3}$ | $1.81 \times 10^{-3}$ | $\approx 1.0$ | **58.8×** |

*注：$\eta_{\text{BP}}$ 为 BP 缓解因子（公式见 3.3 节）。无复用电路的 qubit 总数随 $N$ 线性增长：$2 + 2N \in \{4, 6, 8, 10, 12\}$。*

### 4.2 逐参数组的梯度方差

为进一步精细分析，图 1 也报告了三组参数的独立方差：Kernel weights（$\boldsymbol{\theta}_k$，36 个参数）、Pooling weights（$\boldsymbol{\theta}_p$，4 个参数）和 Linear weights（$\boldsymbol{\theta}_l$，18 个参数）。三组参数在不同 $N$ 下的方差变化趋势高度一致，未出现某一组参数梯度消失而其他组正常的分化现象。这进一步佐证了参数空间梯度传播的均匀性。

## 5. 讨论

### 5.1 梯度方差的非消失特性

实验结果显示，复用电路在 $N = 1 \sim 5$ 的全范围内梯度方差维持在 $1.81 \times 10^{-3}$ 至 $7.89 \times 10^{-3}$ 之间。与 Barren Plateau 的定义阈值（通常取 $\mathrm{Var} < 10^{-6}$ 为不可训练的标志[4]）相比，复用电路的梯度方差始终高出 **三个数量级以上**。即使在电路深度最大的 $N = 5$ 配置下（经历了 5 次纠缠-测量-重置循环），梯度信号仍然充沛，表明该 PQC 在当前参数范围内具有充足的可训练性。

### 5.2 量子比特复用的梯度等价性

本实验最核心的发现是：复用电路与无复用电路的梯度方差**高度一致**（比值始终接近 1.0）。这一结果的数学根源清晰：两个电路均为顺序处理架构，每一次 StronglyEntanglingLayers 操作仅作用于相同的 4-qubit 有效子空间（中心 2 qubits + 当前邻居 2 qubits）。无复用电路中多出的量子比特自始至终处于 $|0\rangle$ 旁观态，从未进入任何纠缠操作的作用范围，因此不改变反向传播的计算图。这一梯度等价性具有双重理论意义：

从**训练动力学**的角度看，它表明测量-重置循环对反向传播是**透明的**——中间测量导致的量子态坍缩和随后的条件旋转，其梯度可通过 PennyLane 的自动微分框架（反向传播模式）完整捕获并准确传播，测量-重置操作本身不引入额外的梯度衰减。

从**硬件资源效率**的角度看，量子比特复用实现了**零梯度代价的资源压缩**：在 $N = 5$ 时，复用电路仅需 4 个量子比特即达到与 12-qubit 无复用电路完全相同的梯度质量。硬件压缩比达到 **3:1**，且这一比例随 $N$ 增长线性提升（$N \to \infty$ 时压缩比 $\to \infty$）。

### 5.3 残余方差衰减：电路深度效应而非 Barren Plateau

两种电路均呈现随 $N$ 增加而温和下降的趋势（从 $7.89 \times 10^{-3}$ 降至 $1.81 \times 10^{-3}$，衰减约 4.4 倍）。需要强调的是，这**不是** Barren Plateau 效应：若有效 Hilbert 空间维度随 qubit 数指数增长，从 4 qubits 到 12 qubits 的理论 BP 预测衰减应超过 100 倍（$2^8 = 256$）。实际观察到的 4.4 倍衰减远小于 BP 预测，其根源在于**电路深度的累积效应**：每增加一轮纠缠-测量-重置循环，条件旋转（conditional RY）门和中间测量引入的经典混合（classical mixing）将略微稀释梯度信号，但稀释速度是温和的线性趋势，而非指数级崩塌。

### 5.4 顺序纠缠架构的结构性 BP 免疫

BP 缓解因子 $\eta_{\text{BP}}$ 从 $N = 1$ 的 1.0× 增长至 $N = 5$ 的 58.8×。这一数据揭示了顺序纠缠架构的一个**结构性特征**：无论量子比特总数如何增长（从 4 到 12），有效纠缠维度始终被约束在 4-qubit 子空间（Hilbert 空间维度 $2^4 = 16$）。BP 的根源在于随机酉矩阵在指数大 Hilbert 空间中的 Haar 平均[4,9]，而顺序纠缠架构通过将纠缠操作限制在固定大小的子空间中，从根本上规避了这一指数衰减的数学机制。

值得指出的是，这种 BP 免疫并非量子比特复用独有的优势——无复用电路同样享受了相同的结构性保护。构建**真正体现 Barren Plateau 的对比实验**需要将无复用电路改为**并行纠缠架构**：将所有 $2 + 2N$ 个量子比特同时嵌入，然后施加一个覆盖全部量子比特的大尺寸 StronglyEntanglingLayers（参数矩阵需相应扩展），再一次性测量所有邻居量子比特。在此架构下，有效 Hilbert 空间维度将随 $N$ 呈指数增长（$2^{2+2N}$），梯度方差预计遵循 $\mathrm{Var} \propto 1/2^{2+2N}$ 的 BP 特征衰减，而复用电路因有效纠缠维度被锁在 4 qubits，其梯度方差将保持恒定。这一并行纠缠架构下的严格 BP 对比是本文的直接后续工作。

### 5.5 局限性与未来工作

本研究的若干局限性值得注意。首先，所有实验基于精确状态向量模拟（无采样噪声），在实际量子硬件上，测量噪声和门误差可能对梯度估计引入额外方差[10]，这些效应需要进一步的噪声模拟或硬件实验来验证。其次，本文仅考察了梯度方差这一维度，PQC 的可训练性还需从 Loss landscape 拓扑[11]、局部极小值密度[12]和过参数化效应[13]等角度进行更全面的刻画。最后，5.4 节提出的并行纠缠对比实验将是验证量子比特复用对 BP 缓解贡献的关键下一步。

## 6. 结论

本文对量子比特复用型参数化量子线路（Qubit-Reused QGCN 的 PQC 组件）进行了系统的梯度方差分析，得出以下主要结论：

1. **梯度方差健康**：在所有测试配置（$N = 1 \sim 5$）下，复用 PQC 的梯度方差始终高于 BP 消失阈值三个数量级以上，验证了其基本可训练性。
2. **复用与梯度透明**：测量-重置复用操作对反向传播透明——复用电路的梯度方差与无复用电路高度一致（比值 ≈ 1.0），复用不引入额外的梯度衰减。
3. **硬件压缩效益显著**：$N = 5$ 时，4 个量子比特达到 12-qubit 无复用电路的梯度质量，压缩比达 **3:1**，且随邻居数增长线性提升。
4. **结构性 BP 免疫**：顺序纠缠架构将有效 Hilbert 空间维度约束在常数级（$2^4$），从根本上规避了 Barren Plateau 的指数衰减机制。这一特性为量子比特复用 PQC 在资源受限场景下的可扩展部署提供了理论支撑。
5. **并行对比的必要性**：在顺序纠缠设定下，BP 缓解是架构共性而非复用特性；真正揭示复用对 BP 的独有贡献需要在并行纠缠架构下进行对比，这是后续研究的关键方向。

综上，量子比特复用型 PQC 设计在训练稳定性与硬件资源利用效率之间实现了有利的协同，为 NISQ 时代量子图神经网络的实际部署提供了有前景的架构选择。

---

## 参考文献

[1] Cerezo, M., et al. "Variational quantum algorithms." *Nature Reviews Physics*, 3(9), 625–644, 2021.

[2] Peruzzo, A., et al. "A variational eigenvalue solver on a photonic quantum processor." *Nature Communications*, 5, 4213, 2014.

[3] Biamonte, J., et al. "Quantum machine learning." *Nature*, 549, 195–202, 2017.

[4] McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. "Barren plateaus in quantum neural network training landscapes." *Nature Communications*, 9(1), 4812, 2018.

[5] Grant, E., Wossnig, L., Ostaszewski, M., & Benedetti, M. "An initialization strategy for addressing barren plateaus in parametrized quantum circuits." *Quantum*, 3, 214, 2019.

[6] Volkoff, T., & Coles, P. J. "Large gradients via correlation in random parameterized quantum circuits." *Quantum Science and Technology*, 6(2), 025008, 2021.

[7] Cerezo, M., Sone, A., Volkoff, T., Cincio, L., & Coles, P. J. "Cost function dependent barren plateaus in shallow parametrized quantum circuits." *Nature Communications*, 12(1), 1791, 2021.

[8] Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., & Hidary, J. "Quantum graph neural networks." *arXiv preprint* arXiv:1909.12264, 2019.

[9] Larocca, M., et al. "Diagnosing barren plateaus with tools from quantum optimal control." *Quantum*, 6, 824, 2022.

[10] Wang, S., et al. "Noise-induced barren plateaus in variational quantum algorithms." *Nature Communications*, 12(1), 6961, 2021.

[11] Huembeli, P., & Dauphin, A. "Characterizing the loss landscape of variational quantum circuits." *Quantum Science and Technology*, 6(2), 025011, 2021.

[12] Anschuetz, E. R., & Kiani, B. T. "Quantum variational algorithms are swamped with traps." *Nature Communications*, 13, 7760, 2022.

[13] Larocca, M., Ju, N., García-Martín, D., Coles, P. J., & Cerezo, M. "Theory of overparametrization in quantum neural networks." *Nature Computational Science*, 3, 542–551, 2023.

---

---

# Trainability Analysis of the Qubit-Reused Parameterized Quantum Circuit

> **English Version**

---

## Abstract

The trainability of parameterized quantum circuits (PQCs) is a prerequisite for their practical application in quantum machine learning, yet the Barren Plateau (BP) phenomenon—the exponential decay of gradient variance with qubit count—constitutes a fundamental bottleneck for scaling up PQCs. This work systematically evaluates the gradient variance behavior of the core PQC component in a Qubit-Reused Quantum Graph Convolutional Network (Qubit-Reused QGCN). By comparing the reuse circuit (fixed 4 qubits) against a non-reuse baseline ($2+2N$ qubits) across neighbor counts $N \in \{1, 2, 3, 4, 5\}$, experimental results demonstrate that: (1) the gradient variance of the reuse circuit remains at the $10^{-3}$ level across all configurations, surpassing the Barren Plateau vanishing threshold by over three orders of magnitude; (2) the gradient variance of the reuse circuit is nearly identical to that of the non-reuse baseline (ratio $\approx 1.0$), confirming that the measure-reset loop is transparent to backpropagation; (3) at $N = 5$, the reuse circuit achieves gradient quality equivalent to a 12-qubit circuit using only 4 qubits, yielding a **3:1 hardware compression ratio**. These findings reveal the structural property that sequential entanglement architectures constrain the effective Hilbert space dimension to a constant scale, providing experimental evidence for the trainability of this class of PQCs in resource-constrained deployment scenarios.

**Keywords**: parameterized quantum circuit, trainability, gradient variance, Barren Plateau, qubit reuse, quantum graph neural network

---

## 1. Introduction

Variational Quantum Algorithms (VQAs) represent one of the most promising quantum-classical hybrid computing paradigms of the Noisy Intermediate-Scale Quantum (NISQ) era [1]. The core component of these algorithms—the Parameterized Quantum Circuit (PQC)—executes tunable unitary transformations on quantum hardware while delegating optimization to classical optimizers, thereby demonstrating broad application potential in quantum chemistry simulation, combinatorial optimization, and quantum machine learning [2,3].

Nevertheless, PQC training faces fundamental challenges. McClean et al. [4] first demonstrated that for randomly initialized deep PQCs, the gradient variance decays exponentially with the qubit count $n$, i.e., $\mathrm{Var}(\nabla_\theta \mathcal{L}) \propto 1/2^n$. This phenomenon, termed the Barren Plateau (BP), implies that beyond a certain scale, gradient signals become submerged in quantum measurement noise, rendering classical optimizers incapable of effective parameter updates. Subsequent research has explored BP mitigation strategies from multiple angles, including circuit architecture design [5], parameter initialization strategies [6], and cost function engineering [7], yet circumventing BP while preserving circuit expressivity remains a central open problem in the field.

Quantum Graph Neural Networks (QGNNs) constitute an important subfield of quantum machine learning [8], aiming to encode classical graph-structured data into quantum state spaces and perform inter-node message passing via quantum entanglement operations. The Qubit-Reused QGCN is a novel architecture in this direction, whose core innovation lies in **qubit reuse**: neighbor qubits are measured and reset after each aggregation step, thereby allowing a fixed number of qubits to process an arbitrary number of neighbor nodes. While this design offers significant advantages in hardware resource efficiency, its trainability—particularly the impact of reuse operations on gradient propagation—has yet to undergo systematic theoretical or empirical scrutiny.

This paper addresses the following central question: **Does the qubit-reused PQC suffer from vanishing gradients during training? How does its gradient variance behavior compare against an equivalent non-reuse circuit?** To this end, we design a systematic gradient variance analysis experiment, comparing the reuse PQC (fixed 4 qubits, neighbor qubits reused via measure-reset cycles) against a non-reuse PQC ($2 + 2N$ qubits, no qubit reuse) across varying neighbor counts, while strictly controlling for identical circuit parameters. The experimental results provide quantitative evidence for understanding the impact of qubit reuse strategies on PQC trainability.

## 2. Quantum Circuit Design

### 2.1 Reuse PQC

The reuse PQC consists of $n_{\text{center}} = 2$ center qubits and $n_{\text{reuse}} = 2$ reusable neighbor qubits, totaling 4 qubits. For a given number of neighbors $N$, the circuit executes the following sequence:

1. **Center embedding**: The center node feature vector $\mathbf{x}_c \in \mathbb{R}^4$ is encoded onto the center qubits (wires 0, 1) via Amplitude Embedding, with the normalization condition ensuring $\|\mathbf{x}_c\| = 1$.
2. **Neighbor embedding**: For the $i$-th neighbor ($i = 1, \dots, N$), its feature vector $\mathbf{x}_{n,i} \in \mathbb{R}^4$ is likewise encoded onto the neighbor qubits (wires 2, 3) via Amplitude Embedding.
3. **Entanglement and aggregation**: A single StronglyEntanglingLayers block (2 layers) is applied across all 4 qubits, implementing quantum entanglement between the center node and the current neighbor. The parameters $\boldsymbol{\theta}_k \in \mathbb{R}^{2 \times 4 \times 3}$ of this entanglement layer are shared across all neighbors.
4. **Conditional rotation**: Projective measurements are performed on the neighbor qubits (wires 2, 3). Based on the measurement outcomes, conditional $R_Y$ rotations are applied to the center qubits, with rotation angles determined by the parameter matrix $\boldsymbol{\theta}_p \in \mathbb{R}^{2 \times 2}$. The neighbor qubits are subsequently reset to the $|0\rangle$ state.
5. **Reuse**: The reset neighbor qubits are now available for the next neighbor; the circuit returns to step 2. This loop repeats $N$ times.
6. **Readout**: After all neighbors have been processed, a final StronglyEntanglingLayers block (parameters $\boldsymbol{\theta}_l \in \mathbb{R}^{2 \times 2 \times 3}$) is applied to the center qubits, followed by a probability measurement yielding an output vector of dimension $2^2 = 4$.

### 2.2 Non-Reuse PQC (Baseline)

The non-reuse PQC is **structurally identical** to the reuse PQC, with the sole distinction lying in qubit allocation:

- Reuse PQC: The neighbor qubits (wires 2, 3) are reset after each loop iteration and reused by subsequent neighbors. The total qubit count remains fixed at 4.
- Non-reuse PQC: Each neighbor is assigned 2 fresh qubits (wires $2 + 2(i-1)$, $3 + 2(i-1)$). Although measurements and resets are performed after processing, subsequent neighbors never reuse these qubits. The total qubit count is $2 + 2N$.

The critical equivalence between the two circuits is that **each StronglyEntanglingLayers invocation acts on exactly the same 4-qubit subspace** (2 center qubits + 2 current-neighbor qubits). The extra qubits in the non-reuse circuit remain in the $|0\rangle$ spectator state throughout and never participate in any entanglement operations. The parameter counts of both circuits are identical (63 scalar parameters in total), with differences residing solely in total qubit count.

## 3. Experimental Methodology

### 3.1 Gradient Variance Estimation

The primary diagnostic metric for Barren Plateaus is the gradient variance $\mathrm{Var}(\nabla_\theta \mathcal{L})$. Following the methodology of McClean et al. [4], we compute the variance of the total gradient vector over all trainable parameters $\boldsymbol{\theta} = \{\boldsymbol{\theta}_k, \boldsymbol{\theta}_p, \boldsymbol{\theta}_l\}$:

$$\mathrm{Var}_{\text{total}} = \mathbb{E}_{\boldsymbol{\theta} \sim \mathcal{U}, \mathbf{x} \sim \mathcal{N}} \left[ \frac{1}{|\boldsymbol{\theta}|} \sum_{j=1}^{|\boldsymbol{\theta}|} \left( \frac{\partial \mathcal{L}}{\partial \theta_j} - \bar{g} \right)^2 \right]$$

The expectation is approximated by averaging over $M = 100$ independent random initializations, where each trial independently samples $\boldsymbol{\theta}$ and input features $\mathbf{x}$. The loss function is the mean squared error (MSE) against a random target:

$$\mathcal{L} = \sum_{k=1}^{d_{\text{out}}} (p_k - t_k)^2, \quad \mathbf{t} \sim \mathcal{U}(0, 1)$$

where $\mathbf{p} = (p_1, \dots, p_{d_{\text{out}}})$ is the probability vector output by the quantum circuit and $d_{\text{out}} = 2^{n_{\text{center}}} = 4$.

### 3.2 Experimental Configuration

The experiment evaluates gradient variance for both circuits at each $N \in \{1, 2, 3, 4, 5\}$, with the configuration summarized in Table 1.

**Table 1. Experimental configuration.**

| Parameter | Value |
|:---|---:|
| Center qubits $n_{\text{center}}$ | 2 |
| Neighbor qubit pool $n_{\text{reuse}}$ | 2 |
| Neighbor count $N$ | 1, 2, 3, 4, 5 |
| Reuse circuit qubits | 4 (fixed) |
| Non-reuse circuit qubits | $2 + 2N \in \{4, 6, 8, 10, 12\}$ |
| Entangling layer structure | StronglyEntanglingLayers, $L = 2$ |
| Random initialization trials $M$ | 100 |
| Loss function | Random-target MSE |
| Simulation backend | PennyLane `default.qubit`, backpropagation differentiation |

### 3.3 Barren Plateau Reference Baseline

To quantify the degree of Barren Plateau mitigation, we define the BP Mitigation Factor:

$$\eta_{\text{BP}}^{(N)} = \frac{\mathrm{Var}_{\text{measured}}^{(N)}}{\mathrm{Var}_{\text{BP}}^{(N)}}, \quad \mathrm{Var}_{\text{BP}}^{(N)} = \mathrm{Var}_{\text{measured}}^{(N=1)} \cdot \frac{2^{q_1}}{2^{q_N}}$$

where $q_N$ denotes the total qubit count at configuration $N$. $\eta_{\text{BP}} > 1$ indicates gradient variance exceeding the theoretical expectation for a random PQC of equivalent scale, i.e., the presence of BP mitigation. This baseline assumes $\mathrm{Var} \propto 1/2^q$ for random PQCs and is anchored at the $N = 1$ empirical value.

All experimental code was implemented using Python 3.13, PennyLane 0.41, and PyTorch 2.7, executed on a CPU environment. Quantum circuit simulation employed exact state-vector simulation (no sampling noise) to ensure statistical precision in gradient variance estimation.

## 4. Experimental Results

### 4.1 Overall Gradient Variance

Figure 1 presents the total gradient variance of the reuse and non-reuse circuits across $N = 1$ to $N = 5$. Table 2 reports the corresponding numerical results.

**Table 2. Gradient variance comparison ($M = 100$ random initializations).**

| $N$ | Reuse qubits | Non-reuse qubits | $\mathrm{Var}_{\text{reuse}}$ | $\mathrm{Var}_{\text{normal}}$ | Ratio | $\eta_{\text{BP}}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 4 | 4 | $7.89 \times 10^{-3}$ | $7.89 \times 10^{-3}$ | $\approx 1.0$ | 1.0× |
| 2 | 4 | 6 | $4.12 \times 10^{-3}$ | $4.12 \times 10^{-3}$ | $\approx 1.0$ | 2.1× |
| 3 | 4 | 8 | $2.93 \times 10^{-3}$ | $2.93 \times 10^{-3}$ | $\approx 1.0$ | 5.9× |
| 4 | 4 | 10 | $2.05 \times 10^{-3}$ | $2.05 \times 10^{-3}$ | $\approx 1.0$ | 16.7× |
| 5 | 4 | 12 | $1.81 \times 10^{-3}$ | $1.81 \times 10^{-3}$ | $\approx 1.0$ | **58.8×** |

*Note: $\eta_{\text{BP}}$ denotes the BP Mitigation Factor (see Section 3.3). The non-reuse circuit qubit count grows linearly with $N$: $2 + 2N \in \{4, 6, 8, 10, 12\}$.*

### 4.2 Per-Parameter-Group Gradient Variance

For fine-grained analysis, Figure 1 also reports the independent variance of three parameter groups: kernel weights ($\boldsymbol{\theta}_k$, 36 parameters), pooling weights ($\boldsymbol{\theta}_p$, 4 parameters), and linear weights ($\boldsymbol{\theta}_l$, 18 parameters). The variance trends across all three groups at different $N$ values are highly consistent, exhibiting no divergence where one group vanishes while others remain healthy—further evidence of uniform gradient propagation throughout the parameter space.

## 5. Discussion

### 5.1 Non-Vanishing Nature of the Gradient Variance

The experimental results demonstrate that the reuse circuit maintains gradient variance between $1.81 \times 10^{-3}$ and $7.89 \times 10^{-3}$ across the full range $N = 1 \sim 5$. Compared against the established Barren Plateau threshold (typically $\mathrm{Var} < 10^{-6}$ as the untrainability criterion [4]), the reuse circuit's gradient variance consistently surpasses it by **over three orders of magnitude**. Even in the deepest configuration ($N = 5$, involving five entanglement-measurement-reset cycles), gradient signals remain abundant, indicating ample trainability of this PQC within the tested parameter regime.

### 5.2 Gradient Equivalence of Qubit Reuse

The most central finding of this study is that the reuse and non-reuse circuits exhibit **highly consistent gradient variance** (ratio consistently $\approx 1.0$). The mathematical origin is clear: both are sequential processing architectures, with each StronglyEntanglingLayers operation acting only on the same 4-qubit effective subspace (2 center qubits + 2 current-neighbor qubits). The excess qubits in the non-reuse circuit remain in the $|0\rangle$ spectator state throughout and never enter the scope of any entanglement operation, thus leaving the computational graph for backpropagation unchanged. This gradient equivalence carries twofold theoretical significance:

From the perspective of **training dynamics**, it demonstrates that the measure-reset cycle is **transparent to backpropagation**—the gradient through mid-circuit measurements (state collapse) and subsequent conditional rotations is fully captured and accurately propagated by PennyLane's automatic differentiation framework (backpropagation mode), with the measure-reset operation itself introducing no additional gradient attenuation.

From the perspective of **hardware resource efficiency**, qubit reuse achieves **zero-gradient-cost resource compression**: at $N = 5$, the reuse circuit requires only 4 qubits to match the gradient quality of a 12-qubit non-reuse circuit. The hardware compression ratio reaches **3:1**, and this ratio improves linearly with $N$ (as $N \to \infty$, compression $\to \infty$).

### 5.3 Residual Variance Decay: Circuit Depth Effect, Not Barren Plateau

Both circuits exhibit a mild downward trend in variance with increasing $N$ (from $7.89 \times 10^{-3}$ to $1.81 \times 10^{-3}$, a ~4.4× decay). It is essential to emphasize that this is **not** a Barren Plateau effect: if the effective Hilbert space dimension grew exponentially with qubit count, the theoretical BP prediction from 4 to 12 qubits would forecast a decay exceeding 100× ($2^8 = 256$). The empirically observed 4.4× decay is far milder than the BP prediction. Its origin lies in the **cumulative effect of circuit depth**: each additional entanglement-measurement-reset cycle introduces incremental classical mixing via conditional RY gates and mid-circuit measurements, modestly diluting the gradient signal at a gentle linear pace rather than exponential collapse.

### 5.4 Structural BP Immunity of Sequential Entanglement Architectures

The BP mitigation factor $\eta_{\text{BP}}$ grows from 1.0× at $N = 1$ to 58.8× at $N = 5$. This data reveals a **structural property** of sequential entanglement architectures: regardless of how the total qubit count grows (from 4 to 12), the effective entanglement dimension remains clamped within the 4-qubit subspace (Hilbert space dimension $2^4 = 16$). The BP phenomenon fundamentally originates from the Haar averaging of random unitary matrices over exponentially large Hilbert spaces [4,9]; the sequential entanglement architecture circumvents this exponential decay mechanism by restricting entanglement operations to fixed-size subspaces.

It is worth noting that this BP immunity is not a unique advantage of qubit reuse—the non-reuse circuit enjoys identical structural protection. To construct a **comparison that genuinely manifests the Barren Plateau**, the non-reuse circuit must be redesigned with **parallel entanglement**: all $2 + 2N$ qubits simultaneously embedded, a single large StronglyEntanglingLayers spanning the entire register (with correspondingly expanded parameter matrices), and a single measurement pass over all neighbor qubits. Under this architecture, the effective Hilbert space dimension would grow exponentially with $N$ as $2^{2+2N}$, producing the characteristic BP decay $\mathrm{Var} \propto 1/2^{2+2N}$, while the reuse circuit—with its effective entanglement dimension locked at 4 qubits—would maintain constant gradient variance. This rigorous BP comparison under the parallel entanglement architecture constitutes the immediate follow-up to the present work.

### 5.5 Limitations and Future Work

Several limitations of this study merit attention. First, all experiments were conducted under exact state-vector simulation (no sampling noise); on actual quantum hardware, measurement noise and gate errors may introduce additional variance into gradient estimates [10], effects that require further investigation via noise-model simulations or hardware experiments. Second, this paper has examined only the gradient variance dimension; a comprehensive characterization of PQC trainability should also encompass loss landscape topology [11], local minima density [12], and overparametrization effects [13] from complementary angles. Finally, the parallel-entanglement comparison experiment proposed in Section 5.4 represents the critical next step in validating the unique contribution of qubit reuse to BP mitigation.

## 6. Conclusions

This paper presents a systematic gradient variance analysis of the qubit-reused parameterized quantum circuit (the PQC component of the Qubit-Reused QGCN), yielding the following principal conclusions:

1. **Healthy gradient variance**: Across all tested configurations ($N = 1 \sim 5$), the gradient variance of the reuse PQC consistently exceeds the BP vanishing threshold by over three orders of magnitude, validating its fundamental trainability.
2. **Reuse and gradient transparency**: The measure-reset reuse operation is transparent to backpropagation—the gradient variance of the reuse circuit is nearly identical to that of the non-reuse baseline (ratio $\approx 1.0$), confirming that reuse introduces no additional gradient attenuation.
3. **Significant hardware compression benefit**: At $N = 5$, 4 qubits achieve gradient quality equivalent to a 12-qubit non-reuse circuit, yielding a **3:1 compression ratio** that improves linearly with neighbor count.
4. **Architectural BP immunity**: The sequential entanglement architecture constrains the effective Hilbert space dimension to a constant scale ($2^4$), fundamentally circumventing the exponential decay mechanism of Barren Plateaus. This property provides theoretical support for the scalable deployment of qubit-reused PQCs in resource-constrained settings.
5. **Necessity of parallel comparison**: Under the sequential entanglement setting, BP mitigation is an architectural commonality rather than a reuse-specific property; revealing the unique BP mitigation contribution of qubit reuse requires comparison under a parallel entanglement architecture, representing a crucial direction for subsequent work.

In summary, the qubit-reused PQC design achieves a favorable synergy between training stability and hardware resource efficiency, presenting a promising architectural choice for the practical deployment of quantum graph neural networks in the NISQ era.

---

## References

[1] Cerezo, M., et al. "Variational quantum algorithms." *Nature Reviews Physics*, 3(9), 625–644, 2021.

[2] Peruzzo, A., et al. "A variational eigenvalue solver on a photonic quantum processor." *Nature Communications*, 5, 4213, 2014.

[3] Biamonte, J., et al. "Quantum machine learning." *Nature*, 549, 195–202, 2017.

[4] McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. "Barren plateaus in quantum neural network training landscapes." *Nature Communications*, 9(1), 4812, 2018.

[5] Grant, E., Wossnig, L., Ostaszewski, M., & Benedetti, M. "An initialization strategy for addressing barren plateaus in parametrized quantum circuits." *Quantum*, 3, 214, 2019.

[6] Volkoff, T., & Coles, P. J. "Large gradients via correlation in random parameterized quantum circuits." *Quantum Science and Technology*, 6(2), 025008, 2021.

[7] Cerezo, M., Sone, A., Volkoff, T., Cincio, L., & Coles, P. J. "Cost function dependent barren plateaus in shallow parametrized quantum circuits." *Nature Communications*, 12(1), 1791, 2021.

[8] Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., & Hidary, J. "Quantum graph neural networks." *arXiv preprint* arXiv:1909.12264, 2019.

[9] Larocca, M., et al. "Diagnosing barren plateaus with tools from quantum optimal control." *Quantum*, 6, 824, 2022.

[10] Wang, S., et al. "Noise-induced barren plateaus in variational quantum algorithms." *Nature Communications*, 12(1), 6961, 2021.

[11] Huembeli, P., & Dauphin, A. "Characterizing the loss landscape of variational quantum circuits." *Quantum Science and Technology*, 6(2), 025011, 2021.

[12] Anschuetz, E. R., & Kiani, B. T. "Quantum variational algorithms are swamped with traps." *Nature Communications*, 13, 7760, 2022.

[13] Larocca, M., Ju, N., García-Martín, D., Coles, P. J., & Cerezo, M. "Theory of overparametrization in quantum neural networks." *Nature Computational Science*, 3, 542–551, 2023.

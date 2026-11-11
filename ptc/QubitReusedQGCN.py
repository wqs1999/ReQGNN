import math

import torch.nn as nn
import torch
import torch.nn.functional as F

# torch.cuda.set_device(0)

import pennylane as qml
import datetime
import matplotlib.pyplot as plt

import Tool

class DualHeadAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=8, proj_dim=None, use_softmax=True):
        super().__init__()
        self.proj_dim = proj_dim or in_dim  # 默认不变
        self.use_softmax = use_softmax

        # 节点线性变换层
        self.center_proj = nn.Linear(in_dim, self.proj_dim)
        self.neigh_proj = nn.Linear(in_dim, self.proj_dim)

        # 两个方向的注意力 MLP
        self.att_mlp_neigh2center = nn.Sequential(
            nn.Linear(self.proj_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.att_mlp_center2neigh = nn.Sequential(
            nn.Linear(self.proj_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, center, neighbors):
        """
        center: [F]
        neighbors: [N, F]

        Returns:
            attn_neigh2center: [N]
            attn_center2neigh: [N]
        """
        N, F = neighbors.shape

        # === Step 1: 线性投影 ===
        h_c = self.center_proj(center)            # [F'] → center
        h_n = self.neigh_proj(neighbors)          # [N, F'] → neighbors
        h_c_expand = h_c.unsqueeze(0).expand(N, -1)  # [N, F']

        # === Step 2: 计算注意力 ===
        pair1 = torch.cat([h_n, h_c_expand], dim=-1)  # neighbor → center
        pair2 = torch.cat([h_c_expand, h_n], dim=-1)  # center → neighbor

        # 限幅防爆
        pair1 = torch.clamp(pair1, -1e4, 1e4)
        pair2 = torch.clamp(pair2, -1e4, 1e4)

        scores1 = self.att_mlp_neigh2center(pair1).squeeze(-1)
        scores2 = self.att_mlp_center2neigh(pair2).squeeze(-1)

        # === Step 3: softmax 或 sigmoid 归一化 ===
        if self.use_softmax:
            attn_neighbor = F.softmax(scores1, dim=0)
            attn_centor = F.softmax(scores2, dim=0)
        else:
            attn_neighbor = torch.sigmoid(scores1)
            attn_centor = torch.sigmoid(scores2)

        return attn_neighbor, attn_centor  # shape: [N], [N]


class QRGCN(nn.Module):

    def __init__(self, QGkernelIteraNum: int, InFeature: int, device, Qlinear_layer: int, num_class: int) -> None:
        super(QRGCN, self).__init__()
        self.QGkernelIteraNum = QGkernelIteraNum
        # self.QubitNum = InFeature * 2
        self.QubitNum = int(math.log(InFeature, 2))*2
        self.InFeature = InFeature
        self.dev = qml.device('default.qubit', wires=self.QubitNum)
        self.device = device

        # Quantum Kernel Weights
        shape_qkernel = qml.StronglyEntanglingLayers.shape(n_layers=QGkernelIteraNum, n_wires=self.QubitNum)
        self.Qkernelweight = nn.Parameter(torch.rand(*shape_qkernel, dtype=torch.float32))

        # Pooling Weights
        self.QGPoolingweight = nn.Parameter(torch.rand(self.QubitNum//2, self.QubitNum//2))  # shape: [4 measured bits, 4 center wires]

        # Quantum Linear Layer
        shape_qlinear = qml.StronglyEntanglingLayers.shape(n_layers=Qlinear_layer, n_wires=self.QubitNum//2)
        self.QLinearweight = nn.Parameter(torch.rand(*shape_qlinear))

        self.register_parameter(name="QuantumGraphKernel", param=self.Qkernelweight)
        self.register_parameter(name="QGPoolingweight", param=self.QGPoolingweight)
        self.register_parameter(name="Quantumlinear", param=self.QLinearweight)

        self.qlayer = qml.QNode(self.QGCC, self.dev, interface='torch')
        # Attention Network
        # self.att_mlp = DualHeadAttention(InFeature).to(device)
        # self.linear = nn.Linear(in_features=2**(self.QubitNum//2), out_features=num_class, dtype=torch.float32)
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2**(self.QubitNum//2), out_features=num_class, dtype=torch.float32)
        )
        # ====== 新增：参数扰动噪声配置 ======
        # 默认关闭噪声
        self.noise_scale = 0.0  # 噪声幅度（例如 0.05）
        self.noise_type = "gaussian"  # ["gaussian", "uniform", "relative"]
        self.noise_targets = {"qkernel", "qpool", "qlinear"}  # 作用到哪些参数
        self.noise_train_only = True  # 只在 model.train() 时启用
        self.noise_seed = None  # 固定随机种子以复现实验（可选）

        # ====== 新增：外部调用接口 ======

    def configure_noise(self, scale: float = 0.0, noise_type: str = "gaussian",
                        targets=("qkernel", "qpool", "qlinear"),
                        train_only: bool = True, seed: int | None = None):
        """
        配置参数扰动噪声。
        - scale: 噪声幅度（如 0.05）。设为 0 即关闭。
        - noise_type: "gaussian" | "uniform" | "relative"
        - targets: 作用对象集合，包含于 {"qkernel", "qpool", "qlinear"}
        - train_only: True 表示仅在 model.train() 时加噪声
        - seed: 固定噪声随机种子（可选）
        """
        self.noise_scale = float(scale)
        self.noise_type = str(noise_type).lower()
        self.noise_targets = set(targets)
        self.noise_train_only = bool(train_only)
        self.noise_seed = seed

    def disable_noise(self):
        """快速关闭噪声。"""
        self.noise_scale = 0.0

        # ====== 新增：内部工具函数（前向时创建“噪声版张量”） ======

    def _apply_noise(self, param: torch.Tensor, tag: str) -> torch.Tensor:
        """
        返回带噪版本（param + eps），不修改原参数本体，保持梯度可回传。
        tag: "qkernel" | "qpool" | "qlinear"
        """
        # 关闭条件
        if self.noise_scale <= 0:
            return param
        if self.noise_train_only and not self.training:
            return param
        if tag not in self.noise_targets:
            return param

        # 可复现实验（可选）
        if self.noise_seed is not None:
            # 用 torch.Generator 避免全局影响
            g = torch.Generator(device=param.device)
            g.manual_seed(self.noise_seed + hash(tag) % 1000003)
        else:
            g = None

        # 采样噪声
        if self.noise_type == "gaussian":
            eps = torch.randn_like(param) * self.noise_scale
        elif self.noise_type == "uniform":
            eps = (torch.rand_like(param) * 2.0 - 1.0) * self.noise_scale
        elif self.noise_type == "relative":
            # 相对幅度：按参数量级缩放（取全局均值作为尺度）
            scale = (param.detach().abs().mean() + 1e-8) * self.noise_scale
            eps = torch.randn_like(param) * scale
        else:
            # 未知类型则不加噪
            return param

        return param + eps

    def QGCC(self, inputs):

        center = inputs.x[0].float()  # float32 强制转换
        neighbors = inputs.x[1:].float()
        # # # print(neighbors)
        # #
        # === Step 1: attention score ===
        # === 在量子层使用“带噪”的权重副本（不改原参数） ===
        qkernel_w = self._apply_noise(self.Qkernelweight, tag="qkernel")
        qpool_w = self._apply_noise(self.QGPoolingweight, tag="qpool")
        qlinear_w = self._apply_noise(self.QLinearweight, tag="qlinear")

        for neighbor in neighbors:
            # 嵌入
            qml.AmplitudeEmbedding(center, wires=range(self.QubitNum // 2), normalize=True)
            qml.AmplitudeEmbedding(neighbor, wires=range(self.QubitNum // 2, self.QubitNum), normalize=True)

            # 量子卷积
            qml.StronglyEntanglingLayers(weights=qkernel_w, wires=range(self.QubitNum))

            # 测量+条件门：量子池化
            measures = [qml.measure(w, reset=True) for w in range(self.QubitNum // 2, self.QubitNum)]
            for mi, m in enumerate(measures):
                for j in range(self.QubitNum // 2):
                    # 使用“带噪”的池化权重
                    qml.cond(m, qml.RY)(qpool_w[mi][j], wires=j)

        # 量子“全连接”
        qml.StronglyEntanglingLayers(weights=qlinear_w, wires=range(self.QubitNum // 2))

        return qml.probs(wires=range(self.QubitNum // 2))

        # return [qml.expval(qml.PauliZ(i)) for i in range(self.QubitNum//2)]


    # 向前传播
    def forward(self, input_list):
        """
        参数:
            input_list: List，每个图包含若干子图，每个子图为一个 Data 对象
        返回:
            Tensor: shape [num_graphs, NumClasses]
        """
        graph_outputs = []
        # print(input_list)

        for subgraph_list in input_list:  # 每个图
            quantum_outputs = []

            # print(len(subgraph_list))

            for data in subgraph_list:    # 每个子图
                data = data.to(self.device)
                out_q = self.qlayer(data)           # shape: [16]
                # print(out_q)
                # out_q = torch.tensor(out_q, dtype=torch.float32, device=self.device).unsqueeze(0)  # shape: [1, InFeature]
                out_q = out_q.reshape(1, 2**(self.QubitNum//2))
                quantum_outputs.append(out_q)
                # print(len(quantum_outputs))

            # 对该图的所有子图输出做平均（可换成 attention 聚合）
            # print(quantum_outputs)
            graph_feat = torch.mean(torch.cat(quantum_outputs, dim=0), dim=0, keepdim=True)  # shape: [1, 16]
            # print(graph_feat)
            graph_outputs.append(graph_feat)

        out = torch.cat(graph_outputs, dim=0).float()  # shape: [num_graphs, 16]
        # print(out.shape)
        return self.linear(out)                        # shape: [num_graphs, NumClasses]

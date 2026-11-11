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
            nn.Linear(in_features=2**(self.QubitNum//2), out_features=num_class, dtype=torch.float64)
        )
    def QGCC(self, inputs):

        # print(inputs)

        # center = inputs.x[0].float()  # float32 强制转换
        # neighbors = inputs.x[1:].float()

        center = inputs['author'].x[0].float()  # print(f"每个作者节点降维后特征维度: {train_set[0]['author'].x.shape[1]}")
        neighbors = inputs['paper'].x.float()
        # # # print(neighbors)
        # #
        # === Step 1: attention score ===
        # scores = [self.att_mlp(center, n) for n in neighbors]  # list of shape [1]
        # # print(center)
        # # print(scores)
        # scores = torch.stack(scores).view(-1)  # shape: [N]
        # # scores = torch.clamp(scores, min=-100, max=100)  # 防止过大过小值造成 exp 爆炸
        # weights = torch.softmax(scores, dim=0)  # shape: [N]，归一化为概率
        #
        #
        # # === Step 2: 权重加权邻居特征 ===
        # weighted_neighbors = neighbors * weights.unsqueeze(-1)  # shape: [N, F]
        #
        # # === Step 3: 按权重排序（从小到大） ===
        # sorted_idx = torch.argsort(weights, dim=0)
        # ordered_neighbors = weighted_neighbors[sorted_idx]
        # print(ordered_neighbors)

        # === Step 2: quantum processing ===
        for neighbor in neighbors:
            # print(scores)
            # print(neighbor)
            # print(self.QubitNum)
            qml.AmplitudeEmbedding(center, wires=range(self.QubitNum//2), normalize=True)
            qml.AmplitudeEmbedding(neighbor, wires=range(self.QubitNum//2, self.QubitNum), normalize=True)
            qml.StronglyEntanglingLayers(weights=self.Qkernelweight, wires=range(self.QubitNum))

            measures = [qml.measure(w, reset=True) for w in range(self.QubitNum//2, self.QubitNum)]
            for mi, m in enumerate(measures):
                for j in range(self.QubitNum//2):
                    qml.cond(m, qml.RY)(self.QGPoolingweight[mi][j], wires=j)

        qml.StronglyEntanglingLayers(weights=self.QLinearweight, wires=range(self.QubitNum//2))
        # qml.BitFlip(p=0.001,wires=range(self.QubitNum//2))

        return qml.probs(wires=range(self.QubitNum//2))

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

        for data in input_list:  # 每个图
            # print(type(data))
            data = data.to(self.device)
            out_q = self.qlayer(data)  # shape: [16]
            graph_outputs.append(out_q)

        # out = torch.cat(graph_outputs, dim=0).float()  # shape: [num_graphs, 16]
        # print(out.shape)
        out = torch.stack(graph_outputs, dim=0)
        return self.linear(out)                        # shape: [num_graphs, NumClasses]

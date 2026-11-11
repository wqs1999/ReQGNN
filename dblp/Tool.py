

from torch_geometric.datasets import DBLP
from torch_geometric.data import HeteroData
from typing import List, Tuple
from sklearn.decomposition import PCA


import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import Planetoid, TUDataset
import torch_geometric.transforms as T
from sklearn.decomposition import PCA

from torch_geometric.utils import k_hop_subgraph, coalesce, remove_self_loops
from torch_geometric.data import Data
from collections import deque, Counter, defaultdict
import numpy as np
import random

import torch
from torch_geometric.loader import DataLoader

from collections import Counter


def load_dblp_author_paper_subgraphs(
        root: str = './data/DBLP',
        train_ratio: float = 0.7,
        test_ratio: float = 0.3,
        seed: int = 42,
        max_nodes_per_subgraph: int = 15,
        reduced_dim: int = 64,
        paper_reduced_dim: int = 64
) -> Tuple[List[HeteroData], List[HeteroData]]:
    """
    加载 DBLP 异构图并构造作者–论文一阶子图，支持训练集测试集划分、PCA 降维。
    作者和论文节点可独立指定降维维度。

    参数：
        root (str): 数据根目录
        train_ratio (float): 训练集比例
        test_ratio (float): 测试集比例
        seed (int): 随机种子
        max_nodes_per_subgraph (int): 每个子图最大节点数（作者 + 论文）
        reduced_dim (int): 作者特征降维维度
        paper_reduced_dim (int): 论文特征降维维度

    返回：
        Tuple[List[HeteroData], List[HeteroData]]: 训练子图列表, 测试子图列表
    """
    assert abs(train_ratio + test_ratio - 1.0) < 1e-5, "train_ratio + test_ratio 必须加和为 1.0"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 加载 PyG DBLP 数据集
    dataset = DBLP(root=root)
    data = dataset[0]

    # 确认边类型存在
    if ('author', 'to', 'paper') not in data.edge_types:
        raise ValueError("数据集中未找到边类型 ('author', 'to', 'paper')，请检查数据集")
    edge_index = data[('author', 'to', 'paper')].edge_index
    if edge_index is None:
        raise ValueError("edge_index 获取失败")

    # 降维作者节点特征
    author_x = data['author'].x
    author_y = data['author'].y
    num_authors = data['author'].num_nodes

    if reduced_dim < author_x.shape[1]:
        pca_author = PCA(n_components=reduced_dim)
        author_x_reduced = torch.tensor(pca_author.fit_transform(author_x.numpy()), dtype=torch.float)
    else:
        author_x_reduced = author_x

    # 降维论文节点特征
    paper_x = data['paper'].x
    if paper_reduced_dim < paper_x.shape[1]:
        pca_paper = PCA(n_components=paper_reduced_dim)
        paper_x_reduced = torch.tensor(pca_paper.fit_transform(paper_x.numpy()), dtype=torch.float)
    else:
        paper_x_reduced = paper_x

    # 构造所有子图
    all_subgraphs = []
    for author_id in range(num_authors):
        paper_ids = edge_index[1][edge_index[0] == author_id].unique()
        if paper_ids.numel() == 0 or (1 + paper_ids.numel()) > max_nodes_per_subgraph:
            continue

        sub_data = HeteroData()
        sub_data['author'].x = author_x_reduced[author_id].unsqueeze(0)
        sub_data['author'].y = author_y[author_id].unsqueeze(0)
        sub_data['author'].node_ids = torch.tensor([author_id])

        sub_data['paper'].x = paper_x_reduced[paper_ids]
        sub_data['paper'].node_ids = paper_ids

        edge_idx = torch.stack([
            torch.zeros_like(paper_ids),
            torch.arange(paper_ids.size(0))
        ])
        sub_data['author', 'to', 'paper'].edge_index = edge_idx

        all_subgraphs.append(sub_data)

    # 数据集划分
    total = len(all_subgraphs)
    indices = list(range(total))
    random.shuffle(indices)
    n_train = int(train_ratio * total)

    train_set = [all_subgraphs[i] for i in indices[:n_train]]
    test_set = [all_subgraphs[i] for i in indices[n_train:]]

    for g in train_set:
        g['author'].train_mask = torch.tensor([True])
        g['author'].test_mask = torch.tensor([False])
    for g in test_set:
        g['author'].train_mask = torch.tensor([False])
        g['author'].test_mask = torch.tensor([True])

    return train_set, test_set


def load_dblp_author_paper_subgraphs42Class(
        root: str = './data/DBLP',
        train_ratio: float = 0.7,
        test_ratio: float = 0.3,
        seed: int = 42,
        max_nodes_per_subgraph: int = 15,
        reduced_dim: int = 64,
        paper_reduced_dim: int = 64
) -> Tuple[List[HeteroData], List[HeteroData]]:
    """
    加载 DBLP 异构图并构造作者–论文一阶子图，支持训练集测试集划分、PCA 降维。
    作者和论文节点可独立指定降维维度。
    仅保留标签为0或1的作者节点。

    参数：
        root (str): 数据根目录
        train_ratio (float): 训练集比例
        test_ratio (float): 测试集比例
        seed (int): 随机种子
        max_nodes_per_subgraph (int): 每个子图最大节点数（作者 + 论文）
        reduced_dim (int): 作者特征降维维度
        paper_reduced_dim (int): 论文特征降维维度

    返回：
        Tuple[List[HeteroData], List[HeteroData]]: 训练子图列表, 测试子图列表
    """
    assert abs(train_ratio + test_ratio - 1.0) < 1e-5, "train_ratio + test_ratio 必须加和为 1.0"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 加载 PyG DBLP 数据集
    dataset = DBLP(root=root)
    data = dataset[0]

    # 确认边类型存在
    if ('author', 'to', 'paper') not in data.edge_types:
        raise ValueError("数据集中未找到边类型 ('author', 'to', 'paper')，请检查数据集")
    edge_index = data[('author', 'to', 'paper')].edge_index
    if edge_index is None:
        raise ValueError("edge_index 获取失败")

    # 降维作者节点特征
    author_x = data['author'].x
    author_y = data['author'].y
    num_authors = data['author'].num_nodes

    if reduced_dim < author_x.shape[1]:
        pca_author = PCA(n_components=reduced_dim)
        author_x_reduced = torch.tensor(pca_author.fit_transform(author_x.numpy()), dtype=torch.float)
    else:
        author_x_reduced = author_x

    # 降维论文节点特征
    paper_x = data['paper'].x
    if paper_reduced_dim < paper_x.shape[1]:
        pca_paper = PCA(n_components=paper_reduced_dim)
        paper_x_reduced = torch.tensor(pca_paper.fit_transform(paper_x.numpy()), dtype=torch.float)
    else:
        paper_x_reduced = paper_x

    # 构造所有子图（仅包含标签0和1的作者）
    all_subgraphs = []
    for author_id in range(num_authors):
        # 检查作者标签是否为0或1
        if author_y[author_id].item() not in (0, 1):
            continue  # 跳过标签不是0或1的作者

        paper_ids = edge_index[1][edge_index[0] == author_id].unique()
        if paper_ids.numel() == 0 or (1 + paper_ids.numel()) > max_nodes_per_subgraph:
            continue

        sub_data = HeteroData()
        sub_data['author'].x = author_x_reduced[author_id].unsqueeze(0)
        sub_data['author'].y = author_y[author_id].unsqueeze(0)
        sub_data['author'].node_ids = torch.tensor([author_id])

        sub_data['paper'].x = paper_x_reduced[paper_ids]
        sub_data['paper'].node_ids = paper_ids

        edge_idx = torch.stack([
            torch.zeros_like(paper_ids),
            torch.arange(paper_ids.size(0))
        ])
        sub_data['author', 'to', 'paper'].edge_index = edge_idx

        all_subgraphs.append(sub_data)

    # 数据集划分
    total = len(all_subgraphs)
    indices = list(range(total))
    random.shuffle(indices)
    n_train = int(train_ratio * total)

    train_set = [all_subgraphs[i] for i in indices[:n_train]]
    test_set = [all_subgraphs[i] for i in indices[n_train:]]

    for g in train_set:
        g['author'].train_mask = torch.tensor([True])
        g['author'].test_mask = torch.tensor([False])
    for g in test_set:
        g['author'].train_mask = torch.tensor([False])
        g['author'].test_mask = torch.tensor([True])

    return train_set, test_set
# import torch
from torch import nn
def run_model_in_batches_for_training(
    model,
    train_list,
    val_list,
    batch_size: int = 8,
    loss_fn = nn.CrossEntropyLoss(),
    optimizer = None,
    device="cpu",
    epochs: int = 20
):
    print(len(train_list))
    """
    对模型进行训练和验证（可选），支持 QRGCN 结构的 List[List[Data]] 子图输入。

    返回：
        训练损失、训练准确率、验证损失、验证准确率
    """

    def batchify(graph_list, batch_size):
        for i in range(0, len(graph_list), batch_size):
            batch_graphs = graph_list[i:i + batch_size]
            batch_labels = [g['author'].y.item() for g in batch_graphs]
            yield batch_graphs, torch.tensor(batch_labels, dtype=torch.long)

    train_loss_log, train_acc_log = [], []
    val_loss_log, val_acc_log = [], []
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_x, batch_labels in batchify(train_list, batch_size):
            preds = model(batch_x)  # shape: [B, num_classes]
            labels = batch_labels.to(device)

            loss = loss_fn(preds, labels)
            total_loss += loss.item() * len(batch_x)
            total_samples += len(batch_x)

            pred_classes = torch.argmax(preds, dim=1)
            # print(pred_classes)
            total_correct += (pred_classes == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            # checkgrad(model)
            optimizer.step()
            # print("wqs")

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        train_loss_log.append(avg_loss)
        train_acc_log.append(avg_acc)

        print(f"[Epoch {epoch}] 🟢 Train Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

        # === 验证 ===
        if val_list is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                for val_x, val_y in batchify(val_list, batch_size):
                    preds = model(val_x)
                    # print(torch.argmax(preds, dim=1))
                    labels = val_y.to(device)
                    # labels = torch.tensor([g['author'].y.item() for g in val_x], device=device)
                    loss = loss_fn(preds, labels)
                    val_loss += loss.item() * len(val_x)
                    val_total += len(val_x)
                    val_correct += (torch.argmax(preds, dim=1) == labels).sum().item()

                val_loss /= val_total
                val_acc = val_correct / val_total
                val_loss_log.append(val_loss)
                val_acc_log.append(val_acc)

                print(f"[Epoch {epoch}] 🧪 Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    return train_loss_log, train_acc_log, val_loss_log, val_acc_log


def plot_training_history(train_loss, train_acc, val_loss=None, val_acc=None, pic="result.png"):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # 📉 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss", marker='o')
    if val_loss is not None:
        plt.plot(epochs, val_loss, label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    # 📈 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Acc", marker='o')
    if val_acc is not None:
        plt.plot(epochs, val_acc, label="Val Acc", marker='s')

        # ✅ 标记最大验证准确率
        max_val = max(val_acc)
        max_val_epoch = val_acc.index(max_val) + 1
        plt.axhline(max_val, color='red', linestyle='--', linewidth=1)
        plt.annotate(f"Max Val Acc: {max_val:.4f} @ Epoch {max_val_epoch}",
                     xy=(max_val_epoch, max_val),
                     xytext=(max_val_epoch, max_val + 0.03),
                     arrowprops=dict(arrowstyle="->", color='red'),
                     fontsize=10, color='red')

    # ✅ 标记最大训练准确率
    max_train = max(train_acc)
    max_train_epoch = train_acc.index(max_train) + 1
    plt.axhline(max_train, color='blue', linestyle='--', linewidth=1)
    plt.annotate(f"Max Train Acc: {max_train:.4f} @ Epoch {max_train_epoch}",
                 xy=(max_train_epoch, max_train),
                 xytext=(max_train_epoch, max_train + 0.03),
                 arrowprops=dict(arrowstyle="->", color='blue'),
                 fontsize=10, color='blue')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(pic)
    plt.show()# dataset = DBLP(root="../data/DBLP")
# data = dataset[0]
# print(data)
#
train_set, test_set = load_dblp_author_paper_subgraphs(
    root='../data/DBLP',
    train_ratio=0.7,
    test_ratio=0.3,
    max_nodes_per_subgraph=6,
    reduced_dim=8,
    paper_reduced_dim=8
)
#
print(f"训练子图数: {len(train_set)}")
print(f"测试子图数: {len(test_set)}")
print(f"每个作者节点降维后特征维度: {train_set[0]['author'].x.shape[1]}")
print(train_set[0])
print(train_set[0]['author'].y)

import torch
from collections import Counter

# 获取训练集中的所有标签
all_train_labels = [g['author'].y.item() for g in train_set]
label_counts = Counter(all_train_labels)

print("训练集标签分布：")
for label, count in label_counts.items():
    print(f"标签 {label}: {count} 个样本")
#
#
# # 获取训练集中的所有标签
all_test_labels = [g['author'].y.item() for g in test_set]
label_counts = Counter(all_test_labels)

print("test集标签分布：")
for label, count in label_counts.items():
    print(f"标签 {label}: {count} 个样本")


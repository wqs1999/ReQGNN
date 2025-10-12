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
from copy import deepcopy
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx


def load_ptc_mr_dataset(
        root='../data',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        reduced_dim=4,
        max_node_num=0,
        seed=42
):
    """
    加载 PTC_MR 数据集，支持节点数过滤、PCA 降维、类别平衡划分。

    参数:
        root (str): 数据集路径
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        reduced_dim (int): 降维后特征维度
        max_node_num (int): 最大节点数（过滤图）
        seed (int): 随机种子

    返回:
        tuple: (dataset, train_indices, val_indices, test_indices, num_features, num_classes)
    """
    import torch
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch_geometric.datasets import TUDataset
    from copy import deepcopy

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "train+val+test 比例总和必须为1.0"

    # 固定随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 加载原始数据集
    original_dataset = TUDataset(root=root, name='PTC_MR')

    # 可选过滤：按最大节点数筛选图
    if max_node_num > 0:
        filtered_dataset = [g for g in original_dataset if g.num_nodes <= max_node_num]
    else:
        filtered_dataset = list(original_dataset)

    dataset_size = len(filtered_dataset)
    labels = np.array([g.y.item() for g in filtered_dataset])

    # === 分层划分 Stratified Split ===
    test_val_ratio = val_ratio + test_ratio
    if test_val_ratio == 0.0:
        train_indices = np.arange(dataset_size)
        val_indices = np.array([], dtype=int)
        test_indices = np.array([], dtype=int)
    else:
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=seed)
        train_indices, rest_indices = next(split.split(np.zeros(dataset_size), labels))

        if val_ratio == 0.0:
            val_indices = np.array([], dtype=int)
            test_indices = rest_indices
        elif test_ratio == 0.0:
            val_indices = rest_indices
            test_indices = np.array([], dtype=int)
        else:
            val_test_ratio = test_ratio / (val_ratio + test_ratio)
            split2 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
            val_idx_, test_idx_ = next(split2.split(np.zeros(len(rest_indices)), labels[rest_indices]))
            val_indices = rest_indices[val_idx_]
            test_indices = rest_indices[test_idx_]

    # === PCA 降维（基于训练集） ===
    train_features = []
    for idx in train_indices:
        train_features.append(filtered_dataset[idx].x.numpy())
    train_features = np.vstack(train_features)

    pca = PCA(n_components=reduced_dim, random_state=seed)
    pca.fit(train_features)

    # 降维后的新数据集
    dataset = []
    for graph in filtered_dataset:
        new_graph = deepcopy(graph)
        reduced_x = pca.transform(graph.x.numpy())
        new_graph.x = torch.tensor(reduced_x, dtype=torch.float)
        dataset.append(new_graph)

    num_features = reduced_dim
    num_classes = original_dataset.num_classes

    return dataset, train_indices, val_indices, test_indices, num_features, num_classes


def load_ptc_fm_dataset(
        root='../data',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        reduced_dim=4,
        max_node_num=0,
        seed=42
):
    """
    加载 PTC_FM 数据集，支持节点数过滤、PCA 降维、类别平衡划分。

    参数:
        root (str): 数据集路径
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        reduced_dim (int): 降维后特征维度
        max_node_num (int): 最大节点数（过滤图）
        seed (int): 随机种子

    返回:
        tuple: (dataset, train_indices, val_indices, test_indices, num_features, num_classes)
    """
    import torch
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch_geometric.datasets import TUDataset
    from copy import deepcopy

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "train+val+test 比例总和必须为1.0"

    # 固定随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 加载原始数据集
    original_dataset = TUDataset(root=root, name='PTC_FM')

    # 可选过滤：按最大节点数筛选图
    if max_node_num > 0:
        filtered_dataset = [g for g in original_dataset if g.num_nodes <= max_node_num]
    else:
        filtered_dataset = list(original_dataset)

    dataset_size = len(filtered_dataset)
    labels = np.array([g.y.item() for g in filtered_dataset])

    # === 分层划分 Stratified Split ===
    test_val_ratio = val_ratio + test_ratio
    if test_val_ratio == 0.0:
        train_indices = np.arange(dataset_size)
        val_indices = np.array([], dtype=int)
        test_indices = np.array([], dtype=int)
    else:
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=seed)
        train_indices, rest_indices = next(split.split(np.zeros(dataset_size), labels))

        if val_ratio == 0.0:
            val_indices = np.array([], dtype=int)
            test_indices = rest_indices
        elif test_ratio == 0.0:
            val_indices = rest_indices
            test_indices = np.array([], dtype=int)
        else:
            val_test_ratio = test_ratio / (val_ratio + test_ratio)
            split2 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
            val_idx_, test_idx_ = next(split2.split(np.zeros(len(rest_indices)), labels[rest_indices]))
            val_indices = rest_indices[val_idx_]
            test_indices = rest_indices[test_idx_]

    # === PCA 降维（基于训练集） ===
    train_features = []
    for idx in train_indices:
        train_features.append(filtered_dataset[idx].x.numpy())
    train_features = np.vstack(train_features)

    pca = PCA(n_components=reduced_dim, random_state=seed)
    pca.fit(train_features)

    # 降维后的新数据集
    dataset = []
    for graph in filtered_dataset:
        new_graph = deepcopy(graph)
        reduced_x = pca.transform(graph.x.numpy())
        new_graph.x = torch.tensor(reduced_x, dtype=torch.float)
        dataset.append(new_graph)

    num_features = reduced_dim
    num_classes = original_dataset.num_classes

    return dataset, train_indices, val_indices, test_indices, num_features, num_classes


def load_ptc_mm_dataset(
        root='../data',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        reduced_dim=4,
        max_node_num=0,
        seed=42
):
    """
    加载 PTC_MM 数据集，支持节点数过滤、PCA 降维、类别平衡划分。

    参数:
        root (str): 数据集路径
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        reduced_dim (int): 降维后特征维度
        max_node_num (int): 最大节点数（过滤图）
        seed (int): 随机种子

    返回:
        tuple: (dataset, train_indices, val_indices, test_indices, num_features, num_classes)
    """
    import torch
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch_geometric.datasets import TUDataset
    from copy import deepcopy

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "train+val+test 比例总和必须为1.0"

    # 固定随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 加载原始数据集
    original_dataset = TUDataset(root=root, name='PTC_MM')

    # 可选过滤：按最大节点数筛选图
    if max_node_num > 0:
        filtered_dataset = [g for g in original_dataset if g.num_nodes <= max_node_num]
    else:
        filtered_dataset = list(original_dataset)

    dataset_size = len(filtered_dataset)
    labels = np.array([g.y.item() for g in filtered_dataset])

    # === 分层划分 Stratified Split ===
    test_val_ratio = val_ratio + test_ratio
    if test_val_ratio == 0.0:
        train_indices = np.arange(dataset_size)
        val_indices = np.array([], dtype=int)
        test_indices = np.array([], dtype=int)
    else:
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=seed)
        train_indices, rest_indices = next(split.split(np.zeros(dataset_size), labels))

        if val_ratio == 0.0:
            val_indices = np.array([], dtype=int)
            test_indices = rest_indices
        elif test_ratio == 0.0:
            val_indices = rest_indices
            test_indices = np.array([], dtype=int)
        else:
            val_test_ratio = test_ratio / (val_ratio + test_ratio)
            split2 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
            val_idx_, test_idx_ = next(split2.split(np.zeros(len(rest_indices)), labels[rest_indices]))
            val_indices = rest_indices[val_idx_]
            test_indices = rest_indices[test_idx_]

    # === PCA 降维（基于训练集） ===
    train_features = []
    for idx in train_indices:
        train_features.append(filtered_dataset[idx].x.numpy())
    train_features = np.vstack(train_features)

    pca = PCA(n_components=reduced_dim, random_state=seed)
    pca.fit(train_features)

    # 降维后的新数据集
    dataset = []
    for graph in filtered_dataset:
        new_graph = deepcopy(graph)
        reduced_x = pca.transform(graph.x.numpy())
        new_graph.x = torch.tensor(reduced_x, dtype=torch.float)
        dataset.append(new_graph)

    num_features = reduced_dim
    num_classes = original_dataset.num_classes

    return dataset, train_indices, val_indices, test_indices, num_features, num_classes

def load_ptc_fr_dataset(
        root='../data',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        reduced_dim=4,
        max_node_num=0,
        seed=42
):
    """
    加载 PTC_FR 数据集，支持节点数过滤、PCA 降维、类别平衡划分。

    参数:
        root (str): 数据集路径
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        reduced_dim (int): 降维后特征维度
        max_node_num (int): 最大节点数（过滤图）
        seed (int): 随机种子

    返回:
        tuple: (dataset, train_indices, val_indices, test_indices, num_features, num_classes)
    """
    import torch
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch_geometric.datasets import TUDataset
    from copy import deepcopy

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "train+val+test 比例总和必须为1.0"

    # 固定随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 加载原始数据集
    original_dataset = TUDataset(root=root, name='PTC_FR')

    # 可选过滤：按最大节点数筛选图
    if max_node_num > 0:
        filtered_dataset = [g for g in original_dataset if g.num_nodes <= max_node_num]
    else:
        filtered_dataset = list(original_dataset)

    dataset_size = len(filtered_dataset)
    labels = np.array([g.y.item() for g in filtered_dataset])

    # === 分层划分 Stratified Split ===
    test_val_ratio = val_ratio + test_ratio
    if test_val_ratio == 0.0:
        train_indices = np.arange(dataset_size)
        val_indices = np.array([], dtype=int)
        test_indices = np.array([], dtype=int)
    else:
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=seed)
        train_indices, rest_indices = next(split.split(np.zeros(dataset_size), labels))

        if val_ratio == 0.0:
            val_indices = np.array([], dtype=int)
            test_indices = rest_indices
        elif test_ratio == 0.0:
            val_indices = rest_indices
            test_indices = np.array([], dtype=int)
        else:
            val_test_ratio = test_ratio / (val_ratio + test_ratio)
            split2 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
            val_idx_, test_idx_ = next(split2.split(np.zeros(len(rest_indices)), labels[rest_indices]))
            val_indices = rest_indices[val_idx_]
            test_indices = rest_indices[test_idx_]

    # === PCA 降维（基于训练集） ===
    train_features = []
    for idx in train_indices:
        train_features.append(filtered_dataset[idx].x.numpy())
    train_features = np.vstack(train_features)

    pca = PCA(n_components=reduced_dim, random_state=seed)
    pca.fit(train_features)

    # 降维后的新数据集
    dataset = []
    for graph in filtered_dataset:
        new_graph = deepcopy(graph)
        reduced_x = pca.transform(graph.x.numpy())
        new_graph.x = torch.tensor(reduced_x, dtype=torch.float)
        dataset.append(new_graph)

    num_features = reduced_dim
    num_classes = original_dataset.num_classes

    return dataset, train_indices, val_indices, test_indices, num_features, num_classes

def extract_all_khop_subgraphs(dataset: list[Data], khops: int = 1) -> tuple[list[list[Data]], list[torch.Tensor]]:
    """
    对每个图的每个节点提取 k-hop 子图，仅保留中心节点与邻居之间的边，标签保留原图标签。

    参数:
        dataset (List[Data]): 多个图，图级标签（data.y）
        khops (int): 跳数，默认 1 表示一阶邻居

    返回:
        Tuple[
            List[List[Data]],  # 每个图对应的子图列表
            List[Tensor]       # 每个图的标签
        ]
    """
    all_graph_subgraphs = []
    all_graph_labels = []

    for data in dataset:
        subgraph_list = []

        for center_node in range(data.num_nodes):
            subset, edge_index, mapping, _ = k_hop_subgraph(
                node_idx=center_node,
                num_hops=khops,
                edge_index=data.edge_index,
                relabel_nodes=False
            )

            # 确保中心节点在首位
            subset = subset.tolist()
            if center_node not in subset:
                continue

            if subset[0] != center_node:
                subset.remove(center_node)
                subset = [center_node] + subset

            # 原图节点编号 → 子图编号映射
            idx_map = {orig: i for i, orig in enumerate(subset)}

            # 重映射 edge_index
            new_edge_index = edge_index.clone()
            for i in range(new_edge_index.size(1)):
                new_edge_index[0, i] = idx_map[new_edge_index[0, i].item()]
                new_edge_index[1, i] = idx_map[new_edge_index[1, i].item()]

            # ✅ 只保留与中心节点（编号为0）直接相连的边
            mask = (new_edge_index[0] == 0) & (new_edge_index[1] != 0)
            filtered_edge_index = new_edge_index[:, mask]

            # 提取子图节点特征，保留图级标签
            x_sub = data.x[subset]
            y_sub = data.y.view(-1).clone()

            sub_data = Data(
                x=x_sub,
                edge_index=filtered_edge_index,
                y=y_sub,
                center_node_idx=0,
                subset=subset
            )

            subgraph_list.append(sub_data)
        # print(data.y.item())
        # print(subgraph_list[0].y.item())
        # print(("_____________________"))
        all_graph_subgraphs.append(subgraph_list)
        all_graph_labels.append(data.y.view(-1).clone())

    return all_graph_subgraphs, all_graph_labels


if __name__ == "__main__":
    dataset, train_idx, val_idx, test_idx, num_feat, num_cls = load_ptc_mr_dataset(train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, max_node_num=8)

    print(f"数据集总大小: {len(dataset)}")
    print(f"训练集大小: {len(train_idx)}")
    print(f"验证集大小: {len(val_idx)}")
    print(f"测试集大小: {len(test_idx)}")
    print(f"节点特征维度: {num_feat}")  # 应为18
    print(f"类别数量: {num_cls}")  # 应为2

    # 访问单个图数据示例
    sample_graph = dataset[1]
    print("\n示例图数据:")
    print(f"节点特征类型：{sample_graph.x}")
    print(f"节点数量: {sample_graph.num_nodes}")
    print(f"边数量: {sample_graph.num_edges}")
    print(f"节点特征形状: {sample_graph.x.shape}")
    print(f"边索引形状: {sample_graph.edge_index.shape}")
    print(f"图标签: {sample_graph.y.item()}")

    # 获取完整训练集数据
    train_set = [dataset[i] for i in train_idx]

    # 获取完整验证集数据
    val_set = [dataset[i] for i in val_idx]

    # 获取完整测试集数据
    test_set = [dataset[i] for i in test_idx]

    print(f"\n训练集第一个图的标签: {train_set[0].y.item()}")

    # 找出节点数量为 4 的图及其标签
    labels_with_4_nodes = [graph.y.item() for graph in dataset if graph.num_nodes <= 8]
    num_graphs_with_4_nodes = len(labels_with_4_nodes)

    # 输出数量
    print(f"节点数量为4的图数量: {num_graphs_with_4_nodes}")

    # 统计这些图的标签分布
    label_counter = Counter(labels_with_4_nodes)
    print("这些图的标签统计分布:")
    for label, count in label_counter.items():
        print(f"  标签 {label}: {count} 个")

    from collections import Counter

    # 获取测试集的标签
    test_labels = [dataset[i].y.item() for i in test_idx]

    # 统计标签数量
    label_counter = Counter(test_labels)

    # 打印分布结果
    print("测试集标签分布:")
    for label, count in label_counter.items():
        print(f"  标签 {label}: {count} 个")
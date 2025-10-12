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


def count_center_node_labels(subgraphs):
    """
    统计子图中每个中心节点的类别数量。

    参数:
    - subgraphs: list，存储所有子图

    返回:
    - Counter 统计每个类别的数量
    """
    labels = []
    for sg in subgraphs:
        center_label = sg.y[sg.center_node_idx].item()  # 中心节点的标签
        labels.append(center_label)

    return Counter(labels)

from math import pi

def reduce_features_pca(data, n_components=4):
    """
    对 data.x 做 PCA 降维。

    参数:
    - data: PyG Data 对象，包含 data.x
    - n_components: 降维后维度（默认 4）

    返回:
    - 降维后的特征矩阵，形状 [num_nodes, n_components]，和 data.x.device 相同
    """
    # 转成 numpy 格式
    original_features = data.x.cpu().numpy()  # 先放到 CPU，PCA 只能用 CPU

    # PCA 降维
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(original_features)

    # 将每个特征缩放到 [-pi/2, pi/2]
    min_vals = reduced_features.min(axis=0, keepdims=True)
    max_vals = reduced_features.max(axis=0, keepdims=True)

    # 防止除以 0
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-8

    # 归一化到 [0, 1]
    reduced_features = (reduced_features - min_vals) / range_vals

    # 缩放到 [-pi/2, pi/2]
    reduced_features = reduced_features * pi - pi/2

    # 转回 torch 张量，转回原来的 device
    reduced_x = torch.tensor(reduced_features, dtype=torch.float, device=data.x.device)

    # 更新 data.x
    data.x = reduced_x
    return data


def split_small_subgraphs(subgraphs, max_nodes=10, train_ratio=0.8, seed=42):
    """
    从所有子图中筛选节点数小于 max_nodes 的子图，并按比例划分为训练集和测试集，
    并根据中心节点的类别数目，返回类别权重张量。

    返回:
    - train_subgraphs: list，训练集子图
    - test_subgraphs: list，测试集子图
    - weights_tensor: tensor，类别权重张量（用于 nn.CrossEntropyLoss）
    """
    # 1. 筛选小子图
    small_subgraphs = [sg for sg in subgraphs.values() if sg.num_nodes <= max_nodes]
    print(f"一共 {len(small_subgraphs)} 个小子图（节点数 <= {max_nodes}）")

    # 2. 随机打乱子图列表
    random.seed(seed)
    random.shuffle(small_subgraphs)

    # 3. 按照比例划分
    num_total = len(small_subgraphs)
    num_train = int(train_ratio * num_total)

    train_subgraphs = small_subgraphs[:num_train]
    test_subgraphs = small_subgraphs[num_train:]

    print(f"训练集子图数量: {len(train_subgraphs)}")
    print(f"测试集子图数量: {len(test_subgraphs)}")

    # 4. 统计训练集中每个类别的数量（中心节点的标签）
    labels = []
    for sg in train_subgraphs:
        center_label = sg.y[sg.center_node_idx].item()
        labels.append(center_label)
    label_counts = Counter(labels)

    print("训练集各类别数量：")
    for label, count in label_counts.items():
        print(f"类别 {label}: {count}")

    # 5. 计算类别权重
    total_count = sum(label_counts.values())
    num_classes = max(label_counts.keys()) + 1  # 假设类别从0开始连续编号

    weights_tensor = torch.zeros(num_classes, dtype=torch.float64)
    for label, count in label_counts.items():
        weight = total_count / (len(label_counts) * count)
        weights_tensor[label] = weight

    print("类别权重张量：", weights_tensor)

    return train_subgraphs, test_subgraphs, weights_tensor


# ✅ 去除 A→B 和 B→A（重编号之后）
def remove_bidirectional_duplicates(edge_index):
    # 对每条边的两个端点排序（小的在前），这样 A-B 和 B-A 就一样了
    row, col = edge_index
    edge = torch.stack([torch.min(row, col), torch.max(row, col)], dim=0)
    # 去重
    edge, _ = torch.unique(edge, dim=1, return_inverse=True)
    return edge


# ✅ 排序边 (src, dst)
def sort_nodes_by_depth(edge_index: torch.Tensor, center_node: int, num_nodes: int):
    adj = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        adj[src].append(dst)
        adj[dst].append(src)  # 无向图

    visited = [False] * num_nodes
    depth_order = []
    queue = deque()
    queue.append(center_node)
    visited[center_node] = True

    while queue:
        node = queue.popleft()
        depth_order.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    return depth_order


def sort_nodes_by_depth_attention(edge_index: torch.Tensor, center_node: int, num_nodes: int, x: torch.Tensor):
    """
    带注意力排序的 BFS，确保所有节点访问。
    使用 dot(x_i, x_j) 作为 attention score。
    """
    # 构建邻接表
    adj = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        adj[src].append(dst)
        adj[dst].append(src)

    visited = [False] * num_nodes
    depth_order = []

    queue = deque()
    queue.append(center_node)
    visited[center_node] = True

    while queue:
        node = queue.popleft()
        depth_order.append(node)

        # 找出未访问的邻居
        unvisited_neighbors = [n for n in adj[node] if not visited[n]]

        if x is not None and unvisited_neighbors:
            # 计算 attention（点积）
            attn = torch.tensor([torch.dot(x[node], x[n]) for n in unvisited_neighbors])
            sorted_idx = torch.argsort(attn, descending=True)
            sorted_neighbors = [unvisited_neighbors[i] for i in sorted_idx]
        else:
            sorted_neighbors = unvisited_neighbors

        for n in sorted_neighbors:
            if not visited[n]:  # 双保险
                visited[n] = True
                queue.append(n)

    # 检查是否遗漏节点（补上孤立点）
    if len(depth_order) < num_nodes:
        missing = [i for i in range(num_nodes) if not visited[i]]
        depth_order.extend(missing)

    return depth_order


# ✅ 获取 BFS 深度
def get_node_depths(edge_index, root: int, num_nodes: int):
    adj = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        adj[src].append(dst)
        adj[dst].append(src)

    depth = [None] * num_nodes
    visited = [False] * num_nodes
    queue = deque([root])
    depth[root] = 0
    visited[root] = True

    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)

    return depth


# 将某个子图转换为树结构边
def build_tree_from_subgraph(edge_index: torch.Tensor, root: int = 0, num_nodes: int = None):
    """
    将子图的 edge_index 转换为树结构边，只保留 BFS 树边。

    参数：
    - edge_index: 原始无向或重复边的 [2, num_edges] 张量
    - root: 树根节点编号，默认为 0
    - num_nodes: 节点总数（可选；用于构建邻接表）

    返回：
    - tree_edge_index: 树结构边（只包含从父到子）
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    # 构建邻接表（无向图）
    adj = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        adj[src].append(dst)
        adj[dst].append(src)

    # BFS 构建树结构边
    visited = [False] * num_nodes
    tree_edges = []

    from collections import deque
    queue = deque([root])
    visited[root] = True

    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                tree_edges.append([node, neighbor])  # 保留父→子边
                queue.append(neighbor)

    tree_edge_index = torch.tensor(tree_edges, dtype=torch.long).t()  # shape [2, num_edges]
    return tree_edge_index


def TransformGraphToTree(data, neighbor_distance):
    print("开始处理图数据")
    subgraphs = []

    for center_node in range(data.num_nodes):
        # 提取子图
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=center_node,
            num_hops=neighbor_distance,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )

        # 去除自环（可选）
        edge_index, _ = remove_self_loops(edge_index)

        # 构造临时子图（还没排序）
        temp_data = Data(
            x=data.x[subset],
            edge_index=edge_index,
            y=data.y[subset] if data.y is not None else None
        )

        # # ✅ 对节点做 BFS 排序
        # bfs_order = sort_nodes_by_depth(temp_data.edge_index, center_node=0, num_nodes=temp_data.num_nodes)
        # ✅ 对节点做 BFS 排序（带注意力）
        bfs_order = sort_nodes_by_depth_attention(temp_data.edge_index, center_node=mapping.item(), num_nodes=temp_data.num_nodes,
                                                  x=temp_data.x)

        reorder_map = {old: new for new, old in enumerate(bfs_order)}

        # ✅ 重新编号 edge_index
        reordered_edge_index = torch.tensor([
            [reorder_map[src.item()] for src in edge_index[0]],
            [reorder_map[dst.item()] for dst in edge_index[1]]
        ], dtype=torch.long)

        # ✅ 去除 A→B 和 B→A（重编号之后）
        reordered_edge_index = remove_bidirectional_duplicates(reordered_edge_index)

        # ✅ 排序边 (src, dst)
        edge_np = reordered_edge_index.numpy()
        sorted_indices = np.lexsort((edge_np[1], edge_np[0]))
        reordered_edge_index = torch.tensor(edge_np[:, sorted_indices], dtype=torch.long)

        # ✅ 重排节点特征和标签
        reordered_x = temp_data.x[bfs_order]
        reordered_y = temp_data.y[bfs_order] if temp_data.y is not None else None

        # ✅ 最终构造子图
        sub_data = Data(
            x=reordered_x,
            edge_index=reordered_edge_index,
            y=reordered_y
        )
        # 将某个子图转换为树结构边
        sub_data.edge_index = build_tree_from_subgraph(sub_data.edge_index, root=0, num_nodes=sub_data.num_nodes)
        # ✅ 获取 BFS 深度
        depth_list = get_node_depths(sub_data.edge_index, root=0, num_nodes=sub_data.num_nodes)
        subgraphs.append((center_node, sub_data, depth_list))

    return subgraphs


from typing import List, Tuple
from torch_geometric.data import Data


def remove_first_n_label_subgraphs(
        subgraph_list: List[Tuple[int, Data, int]],
        label_to_remove: int,
        n: int
) -> List[Tuple[int, Data, int]]:
    """
    从子图列表中移除前 n 个指定标签的子图（根据根节点标签）。

    参数：
        subgraph_list: [(center_node, sub_data, depth), ...] 的列表
        label_to_remove: 需要移除的标签值（如 3）
        n: 要移除的子图数量（按出现顺序）

    返回：
        新的子图列表，已移除指定数量的目标标签子图
    """
    label_target = []
    label_other = []

    for item in subgraph_list:
        _, sub_data, _ = item
        label = sub_data.y[0].item()
        if label == label_to_remove:
            label_target.append(item)
        else:
            label_other.append(item)

    # 保留目标标签中除前 n 个外的部分
    label_target_kept = label_target[n:]

    # 合并并返回
    return label_other + label_target_kept


def CoraDataProcess(train_ratio=0.8, neighbor_distance=2, MiniVecNum=4, MaxVecNum=8, IfRemove=True, RemoveNum=0):
    dataset = Planetoid(root='../data', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    subgraph_list = TransformGraphToTree(data, neighbor_distance=neighbor_distance)

    # small_subgraphs = [(center, sub_data, depth) for center, sub_data, depth in subgraph_list if sub_data.num_nodes <= 20 and sub_data.num_nodes>=8]
    # print(f"共有 {len(small_subgraphs)} 个子图的节点数小于等于20,大于等于8")

    small_subgraphs = [(center, sub_data, depth) for center, sub_data, depth in subgraph_list if
                       MiniVecNum <= sub_data.num_nodes <= MaxVecNum]
    print(f"共有 {len(small_subgraphs)} 个子图的节点数大于等于{MiniVecNum},小于等于{MaxVecNum}")

    if IfRemove == True:
        small_subgraphs = remove_first_n_label_subgraphs(
            small_subgraphs, label_to_remove=3, n=RemoveNum
        )

        print(f"处理后子图总数：{len(small_subgraphs)}")
        Separated()

    # 收集根节点的标签
    root_labels = [sub_data.y[0].item() for _, sub_data, _ in small_subgraphs]

    # 统计每种标签的数量
    label_counter = Counter(root_labels)

    # 打印结果
    print("根节点标签分布统计：")
    for label, count in label_counter.items():
        print(f"标签 {label}：{count} 个")

    max_nodes = max(sub_data.num_nodes for _, sub_data, _ in small_subgraphs)
    print(f"所有子图中最大节点数为: {max_nodes}")

    # 打印第index个节点的子图
    Separated()
    index = 5
    center, sub_data, depth = small_subgraphs[index]
    print(f"第 {index}个节点示例:")
    print(f"中心节点: {center}")
    print(f"子图节点数: {sub_data.num_nodes}")
    print(f"子图边数: {sub_data.num_edges}")
    print(f"子图特征 shape: {sub_data.x.shape}")
    print(f"子图边信息:\n{sub_data.edge_index}")

    print(f"深度标签: {depth}")
    print(f"根节点（节点0）的类别: {sub_data.y[0].item()}")

    # 数据集划分
    # random.seed(1)
    random.shuffle(small_subgraphs)  # 打乱顺序
    split_idx = int(len(small_subgraphs) * train_ratio)
    train_set = small_subgraphs[:split_idx]
    test_set = small_subgraphs[split_idx:]

    Separated()
    print(f"训练集子图数：{len(train_set)}")
    print(f"测试集子图数：{len(test_set)}")

    return train_set, test_set
    # return small_subgraphs


def CoraDataProcessV2(train_ratio=0.8, neighbor_distance=2, MiniVecNum=4, MaxVecNum=8, IfRemove=True, RemoveNum=0):
    dataset = Planetoid(root='../data', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]
    subgraph_list = TransformGraphToSubGraph(data, neighbor_distance=neighbor_distance)

    # small_subgraphs = [(center, sub_data, depth) for center, sub_data, depth in subgraph_list if sub_data.num_nodes <= 20 and sub_data.num_nodes>=8]
    # print(f"共有 {len(small_subgraphs)} 个子图的节点数小于等于20,大于等于8")

    small_subgraphs = [(center, sub_data, depth) for center, sub_data, depth in subgraph_list if
                       MiniVecNum <= sub_data.num_nodes <= MaxVecNum]
    print(f"共有 {len(small_subgraphs)} 个子图的节点数大于等于{MiniVecNum},小于等于{MaxVecNum}")

    if IfRemove == True:
        small_subgraphs = remove_first_n_label_subgraphs(
            small_subgraphs, label_to_remove=3, n=RemoveNum
        )

        print(f"处理后子图总数：{len(small_subgraphs)}")
        # Separated()

    # 收集根节点的标签
    root_labels = [sub_data.y[0].item() for _, sub_data, _ in small_subgraphs]

    # 统计每种标签的数量
    label_counter = Counter(root_labels)

    # 打印结果
    print("根节点标签分布统计：")
    for label, count in label_counter.items():
        print(f"标签 {label}：{count} 个")

    max_nodes = max(sub_data.num_nodes for _, sub_data, _ in small_subgraphs)
    print(f"所有子图中最大节点数为: {max_nodes}")

    # 打印第index个节点的子图
    Separated()
    index = 5
    center, sub_data, depth = small_subgraphs[index]
    print(f"第 {index}个节点示例:")
    print(f"中心节点: {center}")
    print(f"子图节点数: {sub_data.num_nodes}")
    print(f"子图边数: {sub_data.num_edges}")
    print(f"子图特征 shape: {sub_data.x.shape}")
    print(f"子图边信息:\n{sub_data.edge_index}")

    print(f"深度标签: {depth}")
    print(f"根节点（节点0）的类别: {sub_data.y[0].item()}")

    # 数据集划分
    random.seed(0)
    random.shuffle(small_subgraphs)  # 打乱顺序
    split_idx = int(len(small_subgraphs) * train_ratio)
    train_set = small_subgraphs[:split_idx]
    test_set = small_subgraphs[split_idx:]

    Separated()
    print(f"训练集子图数：{len(train_set)}")
    print(f"测试集子图数：{len(test_set)}")

    train_labels = [sub_data.y[0].item() for _, sub_data, _ in train_set]
    test_labels = [sub_data.y[0].item() for _, sub_data, _ in test_set]

    # 统计频次
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    # 打印占比
    print("训练集标签分布：")
    for label, count in sorted(train_counts.items()):
        ratio = count / len(train_set)
        print(f"标签 {label}：{count} 个，占比 {ratio:.2%}")

    print("测试集标签分布：")
    for label, count in sorted(test_counts.items()):
        ratio = count / len(test_set)
        print(f"标签 {label}：{count} 个，占比 {ratio:.2%}")

    return train_set, test_set


def Separated():
    print("_________________________________________")


def filter_subgraphs_by_labels(subgraph_list, target_labels):
    """
    从子图列表中筛选出根节点标签属于指定集合的子图。

    参数:
        subgraph_list: [(center, sub_data, depth), ...]
        target_labels: List[int]，指定要保留的标签，如 [0, 1, 2]

    返回:
        过滤后的子图列表，仅包含目标标签
    """
    return [
        (center, sub_data, depth)
        for center, sub_data, depth in subgraph_list
        if sub_data.y[0].item() in target_labels
    ]


def analyze_graph_by_depth(edge_index, depth):
    """
    参数：
        edge_index: torch.LongTensor [2, num_edges]
        depth: list[int]，每个节点的深度标签

    返回：
        layer_map: dict[int, List[int]]，按层分的节点
        node_to_children: dict[int, List[int]]，每个节点的子节点列表
    """
    # Step 1: 构建 layer_map
    layer_map = defaultdict(list)
    for node_id, d in enumerate(depth):
        layer_map[d].append(node_id)

    # Step 2: 构建邻接表（无向图）
    edge_list = edge_index.t().tolist()
    adj = defaultdict(list)
    for src, dst in edge_list:
        adj[src].append(dst)
        adj[dst].append(src)

    # Step 3: 构建 node_to_children
    node_to_children = defaultdict(list)
    for node_id, d in enumerate(depth):
        for neighbor in adj[node_id]:
            if depth[neighbor] == d + 1:
                node_to_children[node_id].append(neighbor)

    maxdepth = max(depth)
    nodemapping = [{} for i in range(maxdepth)]
    for d, nodes in layer_map.items():
        if d != maxdepth:
            for node in nodes:
                childnodes = node_to_children.get(node)
                nodemapping[d][node] = childnodes

    return dict(layer_map), dict(node_to_children), nodemapping, maxdepth


def TransformGraphToSubGraph(data, neighbor_distance):
    print("开始处理图数据")
    subgraphs = []

    for center_node in range(data.num_nodes):
        # 提取 k-hop 子图
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=center_node,
            num_hops=neighbor_distance,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )

        # 去除自环
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = remove_bidirectional_duplicates(edge_index)

        # 构造初步子图
        temp_data = Data(
            x=data.x[subset],
            edge_index=edge_index,
            y=data.y[subset] if data.y is not None else None
        )

        # ✅ 对节点做 BFS 排序（0是中心节点）
        bfs_order = sort_nodes_by_depth(edge_index, center_node=mapping.item(), num_nodes=temp_data.num_nodes)
        # print(bfs_order)

        # ✅ 创建映射：旧节点编号 → 新节点编号
        reorder_map = {old: new for new, old in enumerate(bfs_order)}

        # ✅ 重排 edge_index
        reordered_edge_index = torch.tensor([
            [reorder_map[src.item()] for src in edge_index[0]],
            [reorder_map[dst.item()] for dst in edge_index[1]]
        ], dtype=torch.long)

        # ✅ 排序边 (src, dst)
        edge_np = reordered_edge_index.numpy()
        sorted_indices = np.lexsort((edge_np[1], edge_np[0]))
        reordered_edge_index = torch.tensor(edge_np[:, sorted_indices], dtype=torch.long)

        # ✅ 重排特征和标签
        reordered_x = temp_data.x[bfs_order]
        reordered_y = temp_data.y[bfs_order] if temp_data.y is not None else None

        # ✅ 构建最终子图
        sub_data = Data(
            x=reordered_x,
            edge_index=reordered_edge_index,
            y=reordered_y
        )
        # print(f"mapping.item:{mapping.item()}")
        # print(center_node)
        # print(temp_data.edge_index)
        # print(temp_data.y)
        # print(sub_data.edge_index)
        # print(sub_data.y)
        # Separated()

        # ✅ BFS 深度更新
        depth_list = get_node_depths(sub_data.edge_index, root=0, num_nodes=sub_data.num_nodes)

        subgraphs.append((center_node, sub_data, depth_list))

    return subgraphs


from collections import defaultdict


def analyze_graph_by_depthV2(edge_index, depth):
    """
    参数：
        edge_index: torch.LongTensor [2, num_edges]
        depth: list[int]，每个节点的深度标签

    返回：
        layer_map: dict[int, List[int]]，按层分的节点
        node_to_children: dict[int, List[int]]，每个节点的子节点列表
        nodemapping: 分层子节点映射（按深度分层的拓扑）
        maxdepth: 最深层深度
    """
    from collections import defaultdict

    # Step 1: 构建 layer_map
    layer_map = defaultdict(list)
    for node_id, d in enumerate(depth):
        if d is not None:
            layer_map[d].append(node_id)

    # Step 2: 构建邻接表（无向图）
    edge_list = edge_index.t().tolist()
    adj = defaultdict(list)
    for src, dst in edge_list:
        adj[src].append(dst)
        adj[dst].append(src)

    # Step 3: 构建 node_to_children
    node_to_children = defaultdict(list)
    for node_id, d in enumerate(depth):
        for neighbor in adj[node_id]:
            if depth[neighbor] is not None and depth[neighbor] >= d:
                node_to_children[node_id].append(neighbor)

    # Step 4: 构建 nodemapping
    maxdepth = max(depth)
    nodemapping = [{} for _ in range(maxdepth)]
    for d, nodes in layer_map.items():
        if d != maxdepth:
            for node in nodes:
                neighbors = node_to_children.get(node, [])
                if node == 0:
                    filtered = neighbors
                else:
                    filtered = [n for n in neighbors if n != 0]
                nodemapping[d][node] = filtered

    return dict(layer_map), dict(node_to_children), nodemapping, maxdepth


def CoraDataProcessPCA(train_ratio=0.8, neighbor_distance=2, MiniVecNum=4, MaxVecNum=8, IfRemove=True, RemoveNum=0, pca_dim=1):
    # 1. 加载数据
    dataset = Planetoid(root='../data', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0]

    # 2. 对 1433维特征做 PCA 降维
    original_features = data.x.numpy()  # [N, 1433]
    pca = PCA(n_components=pca_dim)
    reduced_features = pca.fit_transform(original_features)  # [N, pca_dim]
    # Step 3: 检查是否出现近似零向量（避免 AngleEmbedding 无输入）
    eps = 1e-6
    norms = np.linalg.norm(reduced_features, axis=1)
    zero_mask = norms < eps
    if np.any(zero_mask):
        print(f"警告：共有 {np.sum(zero_mask)} 个样本 PCA 后近似为零向量，已添加小扰动修复")
        reduced_features[zero_mask] = np.random.normal(0, 1e-3, size=(np.sum(zero_mask), pca_dim))
    data.x = torch.tensor(reduced_features, dtype=torch.float)

    # 3. 构建子图
    subgraph_list = TransformGraphToSubGraph(data, neighbor_distance=neighbor_distance)

    small_subgraphs = [(center, sub_data, depth) for center, sub_data, depth in subgraph_list
                       if MiniVecNum <= sub_data.num_nodes <= MaxVecNum]
    print(f"共有 {len(small_subgraphs)} 个子图的节点数大于等于{MiniVecNum},小于等于{MaxVecNum}")

    if IfRemove:
        small_subgraphs = remove_first_n_label_subgraphs(
            small_subgraphs, label_to_remove=3, n=RemoveNum
        )
        print(f"处理后子图总数：{len(small_subgraphs)}")

    # 根节点标签统计
    root_labels = [sub_data.y[0].item() for _, sub_data, _ in small_subgraphs]
    label_counter = Counter(root_labels)
    print("根节点标签分布统计：")
    for label, count in label_counter.items():
        print(f"标签 {label}：{count} 个")

    max_nodes = max(sub_data.num_nodes for _, sub_data, _ in small_subgraphs)
    print(f"所有子图中最大节点数为: {max_nodes}")

    # 示例展示
    Separated()
    index = 5
    center, sub_data, depth = small_subgraphs[index]
    print(f"第 {index} 个节点示例:")
    print(f"中心节点: {center}")
    print(f"子图节点数: {sub_data.num_nodes}")
    print(f"子图边数: {sub_data.num_edges}")
    print(f"子图特征 shape: {sub_data.x.shape}")
    print(f"子图边信息:\n{sub_data.edge_index}")
    print(f"深度标签: {depth}")
    print(f"根节点（节点0）的类别: {sub_data.y[0].item()}")

    # 数据划分
    random.seed(0)
    random.shuffle(small_subgraphs)
    split_idx = int(len(small_subgraphs) * train_ratio)
    train_set = small_subgraphs[:split_idx]
    test_set = small_subgraphs[split_idx:]

    Separated()
    print(f"训练集子图数：{len(train_set)}")
    print(f"测试集子图数：{len(test_set)}")

    # 标签分布
    train_labels = [sub_data.y[0].item() for _, sub_data, _ in train_set]
    test_labels = [sub_data.y[0].item() for _, sub_data, _ in test_set]
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    print("训练集标签分布：")
    for label, count in sorted(train_counts.items()):
        ratio = count / len(train_set)
        print(f"标签 {label}：{count} 个，占比 {ratio:.2%}")

    print("测试集标签分布：")
    for label, count in sorted(test_counts.items()):
        ratio = count / len(test_set)
        print(f"标签 {label}：{count} 个，占比 {ratio:.2%}")

    return train_set, test_set


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
    plt.show()


from copy import deepcopy

def checkgrad(model):
    for param in model.parameters():
        if param.grad is not None:
            print(f"grad norm: {param.grad}")
        else:
            print(f"grad is None")


def checkParam(model):
    i = 0
    for param in model.parameters():
        i = i + 1
        print(f"第{i}组参数：{param}")
    Separated()


def load_ptc_mr_dataset(
        root='./data',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        reduced_dim=4
):
    """
    加载PTC_MR数据集，将节点特征降维到指定维度，并返回整个数据集及划分索引

    参数:
        root (str): 数据集存储路径 (默认: './data')
        train_ratio (float): 训练集比例 (默认: 0.7)
        val_ratio (float): 验证集比例 (默认: 0.15)
        test_ratio (float): 测试集比例 (默认: 0.15)
        seed (int): 随机种子 (默认: 42)
        reduced_dim (int): 降维后的特征维度 (默认: 4)

    返回:
        tuple: (dataset, train_indices, val_indices, test_indices, num_features, num_classes)
        - dataset: 降维后的数据集对象
        - train_indices: 训练集索引列表
        - val_indices: 验证集索引列表
        - test_indices: 测试集索引列表
        - num_features: 降维后的节点特征维度
        - num_classes: 类别数量
    """
    # 验证比例总和为1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例总和必须为1.0"

    # 加载完整数据集
    original_dataset = TUDataset(root=root, name='PTC_MR')

    # 设置随机种子以确保可重复性
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 创建随机排列的索引
    dataset_size = len(original_dataset)
    indices = np.random.permutation(dataset_size)

    # 计算各集合大小
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # 划分索引
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 步骤1: 收集训练集的所有节点特征用于PCA拟合
    train_features = []
    for idx in train_indices:
        graph = original_dataset[idx]
        train_features.append(graph.x.numpy())
    train_features = np.vstack(train_features)

    # 步骤2: 使用PCA进行特征降维
    pca = PCA(n_components=reduced_dim, random_state=seed)
    pca.fit(train_features)

    # 步骤3: 创建数据集副本并应用PCA降维
    dataset = []
    for i in range(len(original_dataset)):
        # 创建图的深拷贝
        graph = deepcopy(original_dataset[i])

        # 应用PCA转换
        original_features = graph.x.numpy()
        reduced_features = pca.transform(original_features)

        # 更新节点特征
        graph.x = torch.tensor(reduced_features, dtype=torch.float)

        dataset.append(graph)

    # 获取数据特征信息
    num_features = reduced_dim
    num_classes = original_dataset.num_classes

    return dataset, train_indices, val_indices, test_indices, num_features, num_classes

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def run_model_in_batches_for_training(
    model,
    train_list,
    train_labels,
    val_list,
    val_labels,
    batch_size: int = 8,
    loss_fn = nn.CrossEntropyLoss(),
    optimizer = None,
    device="cpu",
    epochs: int = 20
):
    """
    对模型进行训练和验证（可选），支持 QRGCN 结构的 List[List[Data]] 子图输入。

    返回：
        训练损失、训练准确率、验证损失、验证准确率
    """

    def batchify(X, y, size):
        for i in range(0, len(X), size):
            yield X[i:i + size], y[i:i + size]



    train_loss_log, train_acc_log = [], []
    val_loss_log, val_acc_log = [], []
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        model.configure_noise(scale=0.05, noise_type="gaussian",
                              targets=("qkernel", "qpool", "qlinear"),
                              train_only=True, seed=42)

        for batch_x, batch_y in batchify(train_list, train_labels, batch_size):
            preds = model(batch_x)  # shape: [B, num_classes]
            labels = torch.stack(batch_y).to(device).view(-1)

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
        if val_list is not None and val_labels is not None:
            model.disable_noise()
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                for val_x, val_y in batchify(val_list, val_labels, batch_size):
                    preds = model(val_x)
                    print(torch.argmax(preds, dim=1))
                    labels = torch.stack(val_y).to(device).view(-1)
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



@torch.no_grad()
def evaluate_model_in_batches(model, subgraph_list, batch_size=8, loss_fn=None, device='cpu'):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    def batchify(data, size):
        for i in range(0, len(data), size):
            yield data[i:i + size]

    for batch in batchify(subgraph_list, batch_size):
        if len(batch) == 0:
            continue

        preds = model(batch)
        labels = torch.tensor([sub_data.y[0].item() for sub_data in batch], device=device)

        if preds.dim() == 1:
            preds = preds.unsqueeze(1)

        loss = loss_fn(preds, labels)
        total_loss += loss.item() * len(batch)
        total_samples += len(batch)

        pred_classes = torch.argmax(preds, dim=1)
        print(pred_classes)
        total_correct += (pred_classes == labels).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc


from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import torch

def extract_khop_subgraphs(data: Data, khops: int = 1) -> list[Data]:
    """
    为单个图中的每个节点提取 k-hop 子图，返回子图列表。

    参数:
        data (Data): 单个 PyG 图（图级标签）。
        khops (int): 邻居跳数（默认为 1）。

    返回:
        List[Data]: 每个中心节点对应一个 k-hop 子图。
    """
    subgraph_list = []

    for center_node in range(data.num_nodes):
        subset, edge_index, mapping, _ = k_hop_subgraph(
            node_idx=center_node,
            num_hops=khops,
            edge_index=data.edge_index,
            relabel_nodes=False
        )

        # 确保中心节点是第一个
        subset = subset.tolist()
        if subset[0] != center_node:
            subset.remove(center_node)
            subset = [center_node] + subset

        # 构建原始编号 -> 子图编号映射
        idx_map = {orig: i for i, orig in enumerate(subset)}

        # 重映射 edge_index
        new_edge_index = edge_index.clone()
        for i in range(new_edge_index.size(1)):
            new_edge_index[0, i] = idx_map[new_edge_index[0, i].item()]
            new_edge_index[1, i] = idx_map[new_edge_index[1, i].item()]

        # 保留与中心节点有关的边
        mask = (new_edge_index[0] == 0) | (new_edge_index[1] == 0)
        filtered_edge_index = new_edge_index[:, mask]

        # 构建子图
        x_sub = data.x[subset]
        y_sub = data.y.view(-1).clone()  # 图级标签

        sub_data = Data(
            x=x_sub,
            edge_index=filtered_edge_index,
            y=y_sub,
            subset=subset,
            center_node_idx=0  # 中心节点为子图中第 0 个
        )

        subgraph_list.append(sub_data)

    return subgraph_list

if __name__ == "__main__":
    dataset, train_idx, val_idx, test_idx, num_feat, num_cls = load_ptc_mr_dataset()

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
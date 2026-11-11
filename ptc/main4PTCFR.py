import torch.nn as nn
import torch
import torch.nn.functional as F

# torch.cuda.set_device(0)

import pennylane as qml
import datetime
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
import Tool
import random
from QubitReusedQGCN import QRGCN
import CustomDataLoader


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    # device = torch.device("cpu")
    in_feature = 8
    # 加载数据与降维
    dataset, train_idx, val_idx, test_idx, num_feat, num_cls = CustomDataLoader.load_ptc_fr_dataset(train_ratio=0.72, val_ratio=0.0, test_ratio=0.28, max_node_num=6, reduced_dim=in_feature, seed=9)
    # 通过索引获取图对象子集
    # print(train_idx)
    train_graphs = [dataset[i] for i in train_idx]
    test_graphs = [dataset[i] for i in test_idx]

    # 提取每个图的所有节点及其一阶邻居
    trainsubgraph_lists, train_labels = CustomDataLoader.extract_all_khop_subgraphs(train_graphs)
    testsubgraph_lists, test_labels = CustomDataLoader.extract_all_khop_subgraphs(test_graphs)

    print(f"训练集数量：{len(trainsubgraph_lists)}")  # 图的数量
    print(f"测试集数量：{len(testsubgraph_lists)}")  # 图的数量

    # 将张量标签转换为整数
    train_label_ids = [int(y.item()) for y in train_labels]
    test_label_ids = [int(y.item()) for y in test_labels]

    # 统计每个类别出现的次数
    from collections import Counter
    train_counter = Counter(train_label_ids)
    test_counter = Counter(test_label_ids)

    # 打印结果
    print("训练集类别分布：")
    for label, count in sorted(train_counter.items()):
        print(f"类别 {label} ：{count} 个")

    print("\n测试集类别分布：")
    for label, count in sorted(test_counter.items()):
        print(f"类别 {label} ：{count} 个")

    # print(trainsubgraph_lists[0][0].edge_index)

    model = QRGCN(QGkernelIteraNum=2, InFeature=in_feature, device=device, Qlinear_layer=2, num_class=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    train_loss, train_acc, val_loss, val_acc = Tool.run_model_in_batches_for_training(
        model,
        train_list=trainsubgraph_lists,
        train_labels=train_labels,
        val_list=testsubgraph_lists,
        val_labels=test_labels,
        optimizer=optimizer,
        device=device,
        epochs=100,
        batch_size=2,
    )
    Tool.plot_training_history(train_loss, train_acc, val_loss, val_acc, pic="PTC_FR")

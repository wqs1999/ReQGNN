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
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    # device = torch.device("cpu")
    in_feature = 8
    # 加载数据与降维
    trainset, testset= Tool.load_dblp_author_paper_subgraphs(
            root='../data/DBLP',
            train_ratio=0.7,
            test_ratio=0.3,
            max_nodes_per_subgraph=6,
            reduced_dim=8,
            paper_reduced_dim=8
        )


    model = QRGCN(QGkernelIteraNum=2, InFeature=in_feature, device=device, Qlinear_layer=2, num_class=4)

    # model.to(device)
    # out = model(trainset[0:2])
    # print(out)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, train_acc, val_loss, val_acc = Tool.run_model_in_batches_for_training(
        model,
        train_list=trainset,
        val_list=testset,
        optimizer=optimizer,
        device=device,
        epochs=500,
        batch_size=25,
    )
    Tool.plot_training_history(train_loss, train_acc, val_loss, val_acc, pic="DBLP.png")

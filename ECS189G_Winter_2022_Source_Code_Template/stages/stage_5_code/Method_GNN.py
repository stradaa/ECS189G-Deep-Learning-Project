'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from ..base_class.method import method
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
# import torch_geometric.nn as pyg_nn
# import torch_geometric.utils as pyg_utils

# import time
# from datetime import datetime
#
# import networkx as nx
# import numpy as np
# import torch.optim as optim
#
# from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
#
import torch_geometric.transforms as T
#
# from tensorboardX import SummaryWriter
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import accuracy, recall, f1_score, precision, true_positive, true_negative, false_negative, false_positive
from typing_extensions import Literal, TypedDict
from typing import Callable, List, Optional, Tuple


class Method_GCN(torch.nn.Module):

    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_dim: int = 16,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor) -> torch.Tensor:
        x = self.dropout1(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.conv2(x, edge_index)
        return x


LossFn = Callable[[Tensor, Tensor], Tensor]
Stage = Literal["train", "val", "test"]


def train_step(
        model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer, loss_fn: LossFn
) -> Tuple[float, float]:
    model.train()
    optimizer.zero_grad()
    mask = data.train_mask
    logits = model(data.x, data.edge_index)[mask]
    preds = logits.argmax(dim=1)
    y = data.y[mask]
    loss = loss_fn(logits, y)

    acc = accuracy(preds, y)

    loss.backward()
    optimizer.step()
    return loss.item(), acc


@torch.no_grad()
def eval_step(model: torch.nn.Module, data: Data, loss_fn: LossFn, stage: Stage) -> Tuple[float, float]:
    model.eval()
    num = 7     # num of classes
    mask = getattr(data, f"{stage}_mask")
    logits = model(data.x, data.edge_index)[mask]
    preds = logits.argmax(dim=1)
    y = data.y[mask]
    loss = loss_fn(logits, y)

    acc = accuracy(preds, y)
    f1 = f1_score(preds, y, num)
    P = precision(preds, y, num)
    R = recall(preds, y, num)

    return loss.item(), acc, f1, R, P


class HistoryDict(TypedDict):
    loss: List[float]
    acc: List[float]
    val_loss: List[float]
    val_acc: List[float]
    R: List[float]
    P: List[float]
    f1: List[float]


def train(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn = torch.nn.CrossEntropyLoss(),
    max_epochs: int = 200,
    early_stopping: int = 10,
    print_interval: int = 10,
    verbose: bool = True,
) -> HistoryDict:
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": [], "f1": [], "R": [], "P": []}
    num = 6     # num of classes -1
    for epoch in range(max_epochs):
        loss, acc = train_step(model, data, optimizer, loss_fn)
        val_loss, val_acc, f1, R, P = eval_step(model, data, loss_fn, "val")
        history["loss"].append(loss)
        history["acc"].append(acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["f1"].append(f1)
        history["R"].append(R)
        history["P"].append(P)

        # The official implementation in TensorFlow is a little different from what is described in the paper...
        if epoch > early_stopping and val_loss > np.mean(history["val_loss"][-(early_stopping + 1): -1]):
            if verbose:
                print("\nEarly stopping...")
            break

        if verbose and epoch % print_interval == 0:
            print(f"\nEpoch: {epoch}\n----------")
            print(f"Train loss: {loss:.4f} | Train acc: {acc:.4f}")
            print(f"  Val loss: {val_loss:.4f} |   Val acc: {val_acc:.4f}")

    test_loss, test_acc, f1, R, P = eval_step(model, data, loss_fn, "test")
    if verbose:
        print(f"\nEpoch: {epoch}\n----------")
        print(f"Train loss: {loss:.4f} | Train acc: {acc:.4f}")
        print(f"  Val loss: {val_loss:.4f} |   Val acc: {val_acc:.4f}")
        print(f" Test loss: {test_loss:.4f} |  Test acc: {test_acc:.4f}")
        print("Recall:", R[num])
        print("Precision:", P[num])
        print("F1:", torch.max(f1))

    return history


def plot_history(history: HistoryDict, title: str, font_size: Optional[int] = 14) -> None:
    plt.suptitle(title, fontsize=font_size)
    ax1 = plt.subplot(121)
    ax1.set_title("Loss")
    ax1.plot(history["loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    ax1.legend()

    ax2 = plt.subplot(122)
    ax2.set_title("Accuracy")
    ax2.plot(history["acc"], label="train")
    ax2.plot(history["val_acc"], label="val")
    plt.xlabel("Epoch")
    ax2.legend()

    plt.show()


# class GNNStack(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
#         super(GNNStack, self).__init__()
#         self.task = task
#         self.convs = nn.ModuleList()
#         self.convs.append(self.build_conv_model(input_dim, hidden_dim))
#         self.lns = nn.ModuleList()
#         self.lns.append(nn.LayerNorm(hidden_dim))
#         self.lns.append(nn.LayerNorm(hidden_dim))
#         for l in range(2):
#             self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
#
#         # post-message-passing
#         self.post_mp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
#             nn.Linear(hidden_dim, output_dim))
#         if not (self.task == 'node' or self.task == 'graph'):
#             raise RuntimeError('Unknown task.')
#
#         self.dropout = 0.25
#         self.num_layers = 3
#
#     def build_conv_model(self, input_dim, hidden_dim):
#         # refer to pytorch geometric nn module for different implementation of GNNs.
#         if self.task == 'node':
#             return pyg_nn.GCNConv(input_dim, hidden_dim)
#         else:
#             return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
#                                   nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         if data.num_node_features == 0:
#           x = torch.ones(data.num_nodes, 1)
#
#         for i in range(self.num_layers):
#             x = self.convs[i](x, edge_index)
#             emb = x
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             if not i == self.num_layers - 1:
#                 x = self.lns[i](x)
#
#         if self.task == 'graph':
#             x = pyg_nn.global_mean_pool(x, batch)
#
#         x = self.post_mp(x)
#
#         return emb, F.log_softmax(x, dim=1)
#
#     def loss(self, pred, label):
#         return F.nll_loss(pred, label)



# class Method_GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = GCNConv(dataset.num_node_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#
#         return F.log_softmax(x, dim=1)
#
#

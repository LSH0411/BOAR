import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add


class GraphConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, norm_type, sigma=20.0, dropout=0.4):
        super(GraphConvLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.sigma = sigma
        self.dropout = dropout

    def gcn_norm(self, edge_index, num_nodes):
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def dense_norm(self, edge_index, edge_weight, num_nodes):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.clamp(min=1e-6).pow(-0.5)
        norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, norm_weight

    def forward(self, x, edge_index, edge_weight=None, target_emb=None):
        num_nodes = x.size(0)
        
        if self.norm_type == 'gcn':
            edge_index, edge_weight = self.gcn_norm(edge_index, num_nodes)
        elif self.norm_type == 'dense':
            edge_index, edge_weight = self.dense_norm(edge_index, edge_weight, num_nodes)
        else:
            raise ValueError(f'Invalid normalization type: {self.norm_type}')

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if self.training:
            edge_weight = F.dropout(edge_weight, p=self.dropout, training=True)
        return edge_weight.view(-1, 1) * x_j

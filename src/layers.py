from torch_geometric.nn import HypergraphConv
import torch.nn as nn
import torch
from torch_geometric.utils import softmax 

class Classifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0):
        super(Classifier, self).__init__()

        self.sequential = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.sequential(x)

class StructuralFeatureRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0, activation: nn.Module = nn.LeakyReLU()):
        super(StructuralFeatureRefiner, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.linear = nn.Linear(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)

        self.hgcn1 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False)
        self.graph_norm1 = nn.LayerNorm(hidden_channels)
        self.skip1 = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x)

        res1 = x
        x = self.hgcn1(x, edge_index)
        x = self.activation(x)
        x = self.graph_norm1(x)
        x = x + self.skip1(res1)

        return x

class SemanticFeatureRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0, activation: nn.Module = nn.LeakyReLU()):
        super(SemanticFeatureRefiner, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.linear = nn.Linear(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)

        # self.hgcn1 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False)
        # self.graph_norm1 = nn.LayerNorm(hidden_channels)
        # self.skip1 = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x)

        # res1 = x
        # x = self.hgcn1(x, edge_index)
        # x = self.activation(x)
        # x = self.graph_norm1(x)
        # x = x + self.skip1(res1)

        return x

class SemanticHyperedgeRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: int = 0.0, activation: nn.Module = nn.LeakyReLU()):
        super(SemanticHyperedgeRefiner, self).__init__()
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.linear_edge = nn.Linear(in_channels, hidden_channels)
        self.norm_edge = nn.LayerNorm(hidden_channels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear_edge(x)
        x = self.activation(x)
        x = self.norm_edge(x)

        return x

import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing

class NodeHyperedgeAttention(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim=None, num_heads=4, dropout=0.2, aggr='mean'):
        super().__init__(aggr=aggr, node_dim=0)

        if hidden_dim is None:
            hidden_dim = node_dim

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.aggr = aggr  # 'sum', 'mean', 'max', 'min'

        self.node_k = nn.Linear(node_dim, hidden_dim)
        self.node_v = nn.Linear(node_dim, hidden_dim)
        self.edge_q = nn.Linear(edge_dim, hidden_dim)

        self.edge_k = nn.Linear(edge_dim, hidden_dim)
        self.edge_v = nn.Linear(edge_dim, hidden_dim)
        self.node_q = nn.Linear(node_dim, hidden_dim)

        self.node_out = nn.Linear(hidden_dim, node_dim)
        self.edge_out = nn.Linear(hidden_dim, edge_dim)

        self.node_ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.LeakyReLU(),
            nn.Linear(node_dim, node_dim),
            nn.Dropout(dropout)
        )

        self.edge_ffn = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.LeakyReLU(),
            nn.Linear(edge_dim, edge_dim),
            nn.Dropout(dropout)
        )

        self.node_norm1 = nn.LayerNorm(node_dim)
        self.node_norm2 = nn.LayerNorm(node_dim)
        self.edge_norm1 = nn.LayerNorm(edge_dim)
        self.edge_norm2 = nn.LayerNorm(edge_dim)

    def _split_heads(self, x):
        return x.view(x.size(0), self.num_heads, self.head_dim)

    def _combine_heads(self, x):
        return x.view(x.size(0), -1)

    def propagate_node_to_edge(self, x_node, x_edge, edge_index):
        src, dst = edge_index

        Q_e = self._split_heads(self.edge_q(x_edge))
        K_n = self._split_heads(self.node_k(x_node))[src]
        V_n = self._split_heads(self.node_v(x_node))[src]

        attn_scores = (Q_e[dst] * K_n).sum(-1) / (self.head_dim ** 0.5)
        attn_weights = softmax(attn_scores, dst)
        attn_weights = self.dropout(attn_weights).unsqueeze(-1)

        msg = attn_weights * V_n

        agg_edge = scatter(msg, dst, dim=0, dim_size=Q_e.size(0), reduce=self.aggr)

        return self.edge_out(self._combine_heads(agg_edge))

    def propagate_edge_to_node(self, x_node, x_edge, edge_index):
        src, dst = edge_index

        Q_n = self._split_heads(self.node_q(x_node))
        K_e = self._split_heads(self.edge_k(x_edge))[dst]
        V_e = self._split_heads(self.edge_v(x_edge))[dst]

        attn_scores = (Q_n[src] * K_e).sum(-1) / (self.head_dim ** 0.5)
        attn_weights = softmax(attn_scores, src)
        attn_weights = self.dropout(attn_weights).unsqueeze(-1)

        msg = attn_weights * V_e

        agg_node = scatter(msg, src, dim=0, dim_size=Q_n.size(0), reduce=self.aggr)

        return self.node_out(self._combine_heads(agg_node))

    def forward(self, x_node, x_edge, edge_index):

        edge_res = x_edge
        edge_msg = self.propagate_node_to_edge(x_node, x_edge, edge_index)
        x_edge = self.edge_norm1(edge_res + edge_msg)
        
        node_res = x_node
        node_msg = self.propagate_edge_to_node(x_node, x_edge, edge_index)
        x_node = self.node_norm1(node_res + node_msg)

        x_edge = self.edge_norm2(x_edge + self.edge_ffn(x_edge))
        x_node = self.node_norm2(x_node + self.node_ffn(x_node))

        return x_node, x_edge
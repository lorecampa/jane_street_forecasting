import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GCNConv


class GraphConvEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward_mult=4, dropout_rate=0.1):
        super(GraphConvEncoderLayer, self).__init__()
        
        self.graph_conv = GCNConv(
            in_channels=hidden_dim, 
            out_channels=hidden_dim
        )

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * dim_feedforward_mult),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * dim_feedforward_mult, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        batch_size, num_nodes, num_features = x.size()

        residual = x
        x = x.reshape(batch_size * num_nodes, num_features)
        x = self.graph_conv(x, edge_index)
        x = x.reshape(batch_size, num_nodes, num_features)        
        x = self.dropout1(x) + residual
        x = self.norm1(x)

        residual = x
        x = self.feedforward(x)
        x = self.dropout2(x) + residual
        x = self.norm2(x)

        return x


class GraphConvEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, dim_feedforward_mult=4, dropout_rate=0.1):
        super(GraphConvEncoder, self).__init__()
        self.layers = nn.ModuleList([
            GraphConvEncoderLayer(
                hidden_dim=hidden_dim,
                dim_feedforward_mult=dim_feedforward_mult,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])

    def forward(self, x, adj):
        batch_size, num_nodes, _ = x.size()

        edge_indices = []
        for batch_idx in range(batch_size):
            adj_matrix = adj[batch_idx]
            src, tgt = torch.nonzero(adj_matrix, as_tuple=True)
            src = src + batch_idx * num_nodes
            tgt = tgt + batch_idx * num_nodes
            edge_indices.append(torch.stack([src, tgt], dim=0))

        edge_index = torch.cat(edge_indices, dim=1).to(x.device)
        
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GraphTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dim_feedforward_mult=4, dropout_rate=0.2):
        super(GraphTransformerEncoderLayer, self).__init__()
        self.gat = GATv2Conv(
            in_channels=hidden_dim, 
            out_channels=hidden_dim // num_heads, 
            heads=num_heads, 
            dropout=dropout_rate,
            add_self_loops=False
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * dim_feedforward_mult),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * dim_feedforward_mult, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        batch_size, num_nodes, num_features = x.size()
        x = x.reshape(batch_size * num_nodes, num_features)
        x = self.gat(x, edge_index)
        x = x.reshape(batch_size, num_nodes, num_features)
        x = x + self.dropout(self.feedforward(self.norm1(x)))
        x = self.norm2(x)
        return x


class GraphTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, dim_feedforward_mult=4, dropout_rate=0.1):
        super(GraphTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            GraphTransformerEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dim_feedforward_mult=dim_feedforward_mult,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])

    def forward(self, x, adj):
        batch_size, num_nodes, _ = x.size()

        edge_indices = []
        for batch_idx in range(batch_size):
            adj_matrix = adj[batch_idx]
            src, tgt = torch.nonzero(adj_matrix, as_tuple=True)
            src = src + batch_idx * num_nodes
            tgt = tgt + batch_idx * num_nodes
            edge_indices.append(torch.stack([src, tgt], dim=0))

        edge_index = torch.cat(edge_indices, dim=1).to(x.device)

        for layer in self.layers:
            x = layer(x, edge_index)
        return x
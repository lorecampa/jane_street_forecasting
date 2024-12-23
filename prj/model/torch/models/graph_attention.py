import torch
from torch import nn
import torch.nn.functional as F
from .layers import TransposeLayer, GraphTransformerEncoder


class StockGNNModel(nn.Module):
    def __init__(
        self,
        input_features,
        hidden_dim=64,
        output_dim=1,
        num_heads=4,
        num_layers=2,
        num_stocks=39,
        embedding_dim=16,
        use_embeddings=False,
        dropout_rate=0.2,
        dim_feedforward_mult=4,
    ):
        super(StockGNNModel, self).__init__()

        self.use_embeddings = use_embeddings

        self.feature_projector = [
            TransposeLayer(),
            nn.BatchNorm1d(input_features),
            TransposeLayer(),
            nn.Dropout(dropout_rate),
        ]
        if use_embeddings:
            self.feature_projector.append(nn.Linear(input_features + embedding_dim, hidden_dim))
            self.embedding_layer = nn.Embedding(num_stocks, embedding_dim)
        else:
            self.feature_projector.append(nn.Linear(input_features, hidden_dim))
        self.feature_projector += [
            TransposeLayer(),
            nn.BatchNorm1d(hidden_dim),
            TransposeLayer(),
            nn.Dropout(dropout_rate),
        ]
        self.feature_projector = nn.Sequential(*self.feature_projector)

        self.encoder = self.encoder = GraphTransformerEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward_mult=dim_feedforward_mult,
            dropout_rate=dropout_rate
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            TransposeLayer(),
            nn.BatchNorm1d(hidden_dim),
            TransposeLayer(),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, symbols, adj):
        batch_size, num_stocks, num_features = x.size()

        if self.use_embeddings:
            stock_embeddings = self.embedding_layer(symbols)
            x = torch.cat([x, stock_embeddings], dim=-1)
        x = self.feature_projector(x)
        x = self.encoder(x, adj)

        output = self.predictor(x)
        return 5 * torch.tanh(output)
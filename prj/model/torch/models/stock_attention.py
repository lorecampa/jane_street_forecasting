import torch
from torch import nn


class StockAttentionModel(nn.Module):
    def __init__(self, input_features, hidden_dim=64, num_heads=4, output_dim=1, num_layers=2):
        super(StockAttentionModel, self).__init__()

        self.feature_projector = nn.Linear(input_features, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation="relu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, mask=None):
        stock_features = self.feature_projector(x)
        stock_features = self.encoder(stock_features, src_key_padding_mask=mask)
        return self.predictor(stock_features)
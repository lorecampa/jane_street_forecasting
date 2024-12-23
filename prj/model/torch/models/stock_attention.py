import torch
from torch import nn
import torch.nn.functional as F
from .layers import TransposeLayer


class StockAttentionModel(nn.Module):
    def __init__(self, input_features, hidden_dims=[64], num_heads=4, output_dim=1, num_layers=2, 
                 num_stocks=39, embedding_dim=16, use_embeddings=False, 
                 dim_feedforward_mult=4, dropout_rate=0.2):
        super(StockAttentionModel, self).__init__()

        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.embedding_layer = nn.Embedding(num_stocks, embedding_dim)

        self.init_layers = nn.Sequential(
            TransposeLayer(), 
            nn.BatchNorm1d(input_features), 
            TransposeLayer(), 
            nn.Dropout(dropout_rate)
        )
        
        in_features = input_features if not use_embeddings else input_features + embedding_dim
        self.feature_projector = []
        for hidden_dim in hidden_dims:
            self.feature_projector.append(nn.Linear(in_features, hidden_dim))
            self.feature_projector.append(TransposeLayer())
            self.feature_projector.append(nn.BatchNorm1d(hidden_dim))
            self.feature_projector.append(TransposeLayer())
            self.feature_projector.append(nn.SiLU())
            self.feature_projector.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        self.feature_projector = nn.Sequential(*self.feature_projector)
        hidden_dim = in_features

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * dim_feedforward_mult,
            activation=F.silu,
            batch_first=True,
            dropout=dropout_rate,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            TransposeLayer(), 
            nn.BatchNorm1d(hidden_dim), 
            TransposeLayer(),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, symbols, mask=None):
        batch_size, num_stocks, num_features = x.size()
        x = self.init_layers(x)
        if self.use_embeddings:
            stock_embeddings = self.embedding_layer(symbols)
            x = torch.cat([x, stock_embeddings], dim=-1)
        stock_features = self.feature_projector(x)
        transformer_features = self.encoder(stock_features, src_key_padding_mask=mask)
        output = self.predictor(stock_features + transformer_features)
        return 5 * F.tanh(output)
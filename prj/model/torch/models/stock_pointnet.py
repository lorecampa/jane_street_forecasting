import torch
from torch import nn
import torch.nn.functional as F
from .layers import TransposeLayer


class StockPointNetModel(nn.Module):
    def __init__(self, input_features, hidden_dims_extractor=[64, 64], 
                 hidden_dims_aggregator=[64], output_dim=1, dropout_rate=0.1,
                 num_stocks=39, embedding_dim=16, use_embeddings=False, 
                 norm='batch', bn_momentum=0.1):
        super(StockPointNetModel, self).__init__()
        assert norm in ['batch', 'instance'], 'Param norm must be batch or instance'
        norm_fn = nn.BatchNorm1d if norm == 'batch' else nn.InstanceNorm1d

        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.embedding_layer = nn.Embedding(num_stocks, embedding_dim)

        in_dim = input_features
        self.feature_extractor = [TransposeLayer(), nn.BatchNorm1d(input_features), TransposeLayer(), nn.Dropout(dropout_rate)]
        for hidden_dim in hidden_dims_extractor:
            self.feature_extractor.append(nn.Linear(in_dim, hidden_dim))
            self.feature_extractor.append(TransposeLayer())
            self.feature_extractor.append(norm_fn(hidden_dim, momentum=bn_momentum))
            self.feature_extractor.append(TransposeLayer())
            self.feature_extractor.append(nn.SiLU())
            self.feature_extractor.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        self.global_aggregator = []
        in_dim = hidden_dims_extractor[-1] * 2
        for hidden_dim in hidden_dims_aggregator:
            self.global_aggregator.append(nn.Linear(in_dim, hidden_dim))
            self.global_aggregator.append(nn.BatchNorm1d(hidden_dim, momentum=bn_momentum))
            self.global_aggregator.append(nn.SiLU())
            self.global_aggregator.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        self.global_aggregator = nn.Sequential(*self.global_aggregator)

        hidden_dim = hidden_dims_extractor[-1] + hidden_dims_aggregator[-1]
        if use_embeddings:
            hidden_dim += embedding_dim
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dims_extractor[-1]),
            TransposeLayer(),
            norm_fn(hidden_dims_extractor[-1], momentum=bn_momentum),
            TransposeLayer(),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims_extractor[-1], output_dim),
            nn.Tanh()
        )

    def forward(self, x, symbols, mask):
        batch_size, num_stocks, num_features = x.size()

        if self.use_embeddings:
            embeddings = self.embedding_layer(symbols)

        # this mask is for attention models so it is inverted
        mask = (~mask).unsqueeze(2)
        x = self.feature_extractor(x)

        expanded_mask = mask.float()
        masked_x = x * expanded_mask

        # Compute max and mean ignoring padding
        global_feature_max = torch.max(masked_x, dim=1, keepdim=False)[0]
        global_feature_mean = torch.sum(masked_x, dim=1, keepdim=False) / torch.sum(expanded_mask, dim=1, keepdim=False).clamp(min=1e-6)
        global_feature = torch.cat([global_feature_max, global_feature_mean], dim=-1)
        global_feature = self.global_aggregator(global_feature)
        global_feature = global_feature.unsqueeze(1)

        global_feature_repeated = global_feature.repeat(1, num_stocks, 1)
        if self.use_embeddings:
            x = torch.cat([x, global_feature_repeated, embeddings], dim=-1)
        else:
            x = torch.cat([x, global_feature_repeated], dim=-1)
        return 5 * self.predictor(x)
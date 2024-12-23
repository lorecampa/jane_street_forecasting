import torch
import torch.nn as nn


class WideDeepModel(nn.Module):
    def __init__(self, input_features, hidden_dims=[], dropout_rate=0.1, output_dim=1):
        super(WideDeepModel, self).__init__()
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        
        self.init_layers = nn.Sequential(nn.BatchNorm1d(input_features), nn.Dropout(dropout_rate))
        in_features = input_features

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.predictor = nn.Sequential(nn.Linear(in_features + input_features, output_dim), nn.Tanh())
        
    def forward(self, x):
        x = self.init_layers(x)
        features = self.feature_extractor(x)
        x = torch.cat([x, features], dim=-1)
        return 5 * self.predictor(x)
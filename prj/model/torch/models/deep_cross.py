import torch
import torch.nn as nn


class DeepCrossModel(nn.Module):
    def __init__(self, input_features, hidden_dims=[], n_cross_layers=2, dropout_rate=0.1, output_dim=1):
        super(DeepCrossModel, self).__init__()
        self.hidden_dims = hidden_dims
        self.n_cross_layers = n_cross_layers
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        
        self.init_layers = nn.Sequential(nn.BatchNorm1d(input_features), nn.Dropout(dropout_rate))
        in_features = input_features

        deep_layers = []
        for hidden_dim in hidden_dims:
            deep_layers.append(nn.Linear(in_features, hidden_dim))
            deep_layers.append(nn.BatchNorm1d(hidden_dim))
            deep_layers.append(nn.SiLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        self.deep_extractor = nn.Sequential(*deep_layers)
        self.cross_layers = nn.ModuleList([nn.Linear(input_features, input_features) for _ in range(n_cross_layers)])
        self.cross_post = nn.Sequential(nn.BatchNorm1d(input_features), nn.Dropout(dropout_rate))
        self.predictor = nn.Sequential(nn.Linear(in_features + input_features, output_dim), nn.Tanh())
        
    def forward(self, x):
        x0 = self.init_layers(x)
        
        deep_features = self.deep_extractor(x0)
        
        x_cross = x0
        for i in range(self.n_cross_layers):
            x = self.cross_layers[i](x_cross)
            x_cross = x0 * x + x_cross
        cross_features = self.cross_post(x_cross)
            
        x = torch.cat([cross_features, deep_features], dim=-1)
        return 5 * self.predictor(x)
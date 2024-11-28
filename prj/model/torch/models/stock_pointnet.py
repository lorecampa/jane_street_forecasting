import torch
from torch import nn


class StockPointNetModel(nn.Module):

    def __init__(self, input_features, hidden_dim=64, output_dim=1):
        super(StockPointNetModel, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.global_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, mask=None):
        batch_size, num_stocks, num_features = x.size()
        stock_features = self.feature_extractor(x)  # (batch_size, N, hidden_dim)
        
        global_feature = torch.max(stock_features, dim=1, keepdim=True)[0]  # (batch_size, 1, hidden_dim)
        global_feature = self.global_aggregator(global_feature)  # (batch_size, 1, hidden_dim)

        global_feature_repeated = global_feature.repeat(1, num_stocks, 1)  # (batch_size, N, hidden_dim)
        combined_features = torch.cat([stock_features, global_feature_repeated], dim=-1)  # (batch_size, N, hidden_dim * 2)

        output = self.predictor(combined_features)  # (batch_size, N, 1)
        return output
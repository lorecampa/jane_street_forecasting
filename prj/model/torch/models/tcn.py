import torch
from torch import nn
import torch.nn.functional as F
from .layers import DilatedResNetBlock


class TimeConvolutionsModel(nn.Module):
    def __init__(self, input_features, hidden_dim=64, num_layers=3, kernel_size=3, output_dim=1, dropout_rate=0.2):
        super(TimeConvolutionsModel, self).__init__()

        self.feature_projector = nn.Sequential(
            nn.BatchNorm1d(input_features),
            nn.Dropout(dropout_rate),
            nn.Conv1d(input_features, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )

        # Residual blocks with increasing dilation rates
        self.res_blocks = nn.ModuleList([
            DilatedResNetBlock(
                hidden_dim, hidden_dim, kernel_size, dilation=2**i, dropout_rate=dropout_rate
            ) for i in range(num_layers)
        ])

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feature_projector(x)

        for block in self.res_blocks:
            x = block(x)

        x = x[:, :, -1]  # (batch_size, hidden_dim)
        x = self.predictor(x)  # (batch_size, output_dim)
        return 5 * x
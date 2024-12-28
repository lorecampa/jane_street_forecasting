import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class DilatedResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout_rate=0.2):
        super(DilatedResNetBlock, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, 
                dilation=dilation, padding=0, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout_rate),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(
                out_channels, out_channels, kernel_size=kernel_size, 
                dilation=dilation, padding=0, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout_rate)
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else None
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        residual = x
        pad = (self.dilation * (self.kernel_size - 1), 0)  # Causal padding
        x = F.pad(x, pad)
        x = self.block1(x)

        x = F.pad(x, pad)
        x = self.block2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        
        x += residual
        return self.silu(x)
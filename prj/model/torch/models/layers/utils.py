import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class TransposeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input.transpose(1, 2)
    
    
class CausalPadding(nn.Module):
    def __init__(self, left_padding) -> None:
        super().__init__()
        self.left_padding = left_padding

    def forward(self, x: Tensor) -> Tensor:
        return F.pad(x, (self.left_padding, 0))
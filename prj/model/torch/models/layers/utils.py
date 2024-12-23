import torch
from torch import Tensor
import torch.nn as nn

class TransposeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input.transpose(1, 2)
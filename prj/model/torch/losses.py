from torch import nn, Tensor


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, predictions: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        squared_diff = (predictions - targets) ** 2
        weighted_squared_diff = weights * squared_diff
        return weighted_squared_diff.sum() / weights.sum()
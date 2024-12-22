from torch import Tensor

def weighted_r2_score(preds, targets, weights):
    ss_res = (weights * (targets - preds) ** 2).sum()
    ss_tot = (weights * (targets ** 2)).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
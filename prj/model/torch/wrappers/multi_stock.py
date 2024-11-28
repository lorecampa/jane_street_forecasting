from pathlib import Path
import gc
from typing import List, Union, Dict, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchmetrics.functional as tmf
import lightning as L
import lightning.pytorch.callbacks as C

from prj.model.torch.metrics import weighted_r2_score


class JaneStreetMultiStockModel(L.LightningModule):

    def __init__(self, 
                 model: nn.Module,
                 losses: List[nn.Module] | nn.Module, 
                 loss_weights: List[float], 
                 l1_lambda: float = 1e-4,
                 l2_lambda: float = 1e-4,
                 optimizer: str = 'Adam',
                 optimizer_cfg: Dict[str, Any] = dict()):
        super(JaneStreetMultiStockModel, self).__init__()   
        assert isinstance(losses, nn.Module) or len(losses) == len(loss_weights), 'Each loss must have a weight'
        assert len(loss_weights) == 0 or min(loss_weights) > 0, 'Losses must have positive weights'
        self.model = model
        losses = [losses] if isinstance(losses, nn.Module) else losses
        self.losses = nn.ModuleList(losses) 
        self.loss_weights = [1.0] if isinstance(losses, nn.Module) else loss_weights
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.optimizer_name = optimizer
        self.optimizer_cfg = optimizer_cfg

    def forward(self, x, mask):
        return self.model(x, mask=mask)

    def training_step(self, batch, batch_idx):
        x, y, mask, weights = batch
        y_hat = self.forward(x, mask)
        loss = self._compute_loss(y_hat.squeeze(), y, weights)
        with torch.no_grad():
            metrics = self._compute_metrics(y_hat.squeeze(), y, weights, prefix='train')
        metrics['train_loss'] = loss
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss + self._regularization_loss()

    def validation_step(self, batch, batch_idx):
        x, y, mask, weights = batch
        y_hat = self.forward(x, mask)
        metrics = self._compute_metrics(y_hat.squeeze(), y, weights, prefix='val')
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def _compute_metrics(self, preds, targets, weights, prefix='val'):
        metrics = dict()
        metrics[f'{prefix}_wmse'] = (weights * (preds - targets) ** 2).sum() / weights.sum()
        metrics[f'{prefix}_wmae'] = (weights * (preds - targets).abs()).sum() / weights.sum()
        metrics[f'{prefix}_wr2'] = weighted_r2_score(preds, targets, weights)
        return metrics

    def _compute_loss(self, preds, targets, weights):
        loss = 0
        for i in range(len(self.losses)):
            loss += self.losses[i](preds, targets, weights=weights) * self.loss_weights[i]
        return loss
    
    def _regularization_loss(self):
        reg_loss = 0
        
        if self.l1_lambda > 0:
            l1_loss = sum(p.abs().sum() for p in self.parameters())
            reg_loss += l1_loss * self.l1_lambda
            
        if self.l2_lambda > 0:
            l2_loss = sum(p.pow(2).sum() for p in self.parameters())
            reg_loss += l2_loss * self.l2_lambda
            
        return reg_loss

    def configure_optimizers(self):
        return getattr(torch.optim, self.optimizer_name)(self.parameters(), **self.optimizer_cfg)
import os
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from prj.model.torch.metrics import weighted_r2_score


class JaneStreetBaseModel(L.LightningModule):

    def __init__(self, 
                 model: nn.Module,
                 losses: List[nn.Module] | nn.Module, 
                 loss_weights: List[float], 
                 l1_lambda: float = 1e-4,
                 l2_lambda: float = 1e-4,
                 optimizer: str = 'Adam',
                 optimizer_cfg: Dict[str, Any] = dict(),
                 scheduler: str = None,
                 scheduler_cfg: Dict[str, Any] = dict()):
        super(JaneStreetBaseModel, self).__init__()   
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
        self.scheduler_name = scheduler
        self.scheduler_cfg = scheduler_cfg
        self.acc_metrics = dict(ss_res=0.0, ss_tot=0.0, abs_err_sum=0.0, weights_sum=0.0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, weights = batch
        y_hat = self.forward(x).squeeze()
        loss = self._compute_loss(y_hat, y, weights)
        reg_loss, l1_loss, l2_loss = self._regularization_loss()
        metrics = dict()
        with torch.no_grad():
            metrics['train_wmse'] = (weights * (y_hat - y) ** 2).sum() / weights.sum()
            metrics['train_wmae'] = (weights * (y_hat - y).abs()).sum() / weights.sum()
            metrics['train_wr2'] = weighted_r2_score(y_hat, y, weights)
        metrics['train_loss'] = loss
        if self.l1_lambda > 0:
            metrics['train_l1_reg'] = l1_loss
        if self.l2_lambda > 0:
            metrics['train_l2_reg'] = l2_loss
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=x.size(0))
        return loss + reg_loss

    def validation_step(self, batch, batch_idx):
        x, y, weights = batch
        y_hat = self.forward(x).squeeze()
        loss = self._compute_loss(y_hat, y, weights)
        ss_res_step = (weights * (y_hat - y) ** 2).sum()
        ss_tot_step = (weights * (y ** 2)).sum()
        abs_err_step = (weights * (y_hat - y).abs()).sum()
        weights_sum_step = weights.sum()
        self.acc_metrics['ss_res'] += ss_res_step
        self.acc_metrics['ss_tot'] += ss_tot_step
        self.acc_metrics['abs_err_sum'] += abs_err_step
        self.acc_metrics['weights_sum'] += weights_sum_step
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=x.size(0))

    def on_validation_epoch_end(self):
        metrics = dict()
        metrics['val_wmse'] = self.acc_metrics['ss_res'] / self.acc_metrics['weights_sum']
        metrics['val_wmae'] = self.acc_metrics['abs_err_sum'] / self.acc_metrics['weights_sum']
        metrics['val_wr2'] = 1 - self.acc_metrics['ss_res'] / self.acc_metrics['ss_tot']
        self.acc_metrics = dict(ss_res=0.0, ss_tot=0.0, abs_err_sum=0.0, weights_sum=0.0)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _compute_loss(self, preds, targets, weights):
        loss = 0
        for i in range(len(self.losses)):
            loss += self.losses[i](preds, targets, weights=weights) * self.loss_weights[i]
        return loss
    
    def _regularization_loss(self):
        reg_loss = 0
        l1_loss = 0
        l2_loss = 0
        
        if self.l1_lambda > 0:
            l1_loss = sum(p.abs().sum() for p in self.parameters())
            reg_loss += l1_loss * self.l1_lambda
            
        if self.l2_lambda > 0:
            l2_loss = sum(p.pow(2).sum() for p in self.parameters())
            reg_loss += l2_loss * self.l2_lambda
            
        return reg_loss, l1_loss, l2_loss

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(), **self.optimizer_cfg)
        if self.scheduler_name is None:
            return optimizer
        scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_name)(optimizer, **self.scheduler_cfg)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_wr2',
            }
        }
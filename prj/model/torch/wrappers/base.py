import os
from typing import Any, Dict, List
import joblib
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from prj.model.torch.metrics import weighted_r2_score

class JaneStreetModelWrapper(L.LightningModule):
    def __init__(self, 
                model: nn.Module,
                losses: List[nn.Module] | nn.Module, 
                loss_weights: List[float], 
                l1_lambda: float = 1e-4,
                l2_lambda: float = 1e-4,
                use_gaussian_noise: bool = False,
                gaussian_noise_std: float = 0.1,
                optimizer: str = 'Adam',
                optimizer_cfg: Dict[str, Any] = dict(),
                scheduler: str = None,
                scheduler_cfg: Dict[str, Any] = dict()):
        
        super(JaneStreetModelWrapper, self).__init__()   
        assert isinstance(losses, nn.Module) or len(losses) == len(loss_weights), 'Each loss must have a weight'
        assert len(loss_weights) == 0 or min(loss_weights) > 0, 'Losses must have positive weights'
        self.model = model
        losses = [losses] if isinstance(losses, nn.Module) else losses
        self.losses = nn.ModuleList(losses) 
        self.loss_weights = [1.0] if isinstance(losses, nn.Module) else loss_weights
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.use_gaussian_noise = use_gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std
        self.optimizer_name = optimizer
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_name = scheduler
        self.scheduler_cfg = scheduler_cfg

    def forward(self, x):
        if self.use_gaussian_noise:
            x = x + torch.randn_like(x) * self.gaussian_noise_std
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, weights = batch
        y_hat = self.forward(x).squeeze()
        loss = self._compute_loss(y_hat, y, weights)
        with torch.no_grad():
            metrics = self._compute_metrics(y_hat, y, weights, prefix='train')
        metrics['train_loss'] = loss
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss + self._regularization_loss()

    def validation_step(self, batch, batch_idx):
        x, y, weights = batch
        y_hat = self.forward(x).squeeze()
        loss = self._compute_loss(y_hat, y, weights)
        metrics = self._compute_metrics(y_hat, y, weights, prefix='val')
        metrics['val_loss'] = loss
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
        optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(), **self.optimizer_cfg)
        if self.scheduler_name is None:
            return optimizer
        scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_name)(optimizer, **self.scheduler_cfg)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            y_hat: np.ndarray = self.model(torch.from_numpy(X)).squeeze().numpy()
        
        return y_hat
    
    
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.model is not None:
            torch.save(self.model, os.path.join(path, 'model.pth'))
          
        _model = self.model
        self.model = None
        joblib.dump(self, os.path.join(path, 'class.joblib'))
        self.model = _model


    @staticmethod
    def load(path: str) -> 'JaneStreetModelWrapper':
        _class = joblib.load(os.path.join(path, 'class.joblib'))
        _model_path = os.path.join(path, 'model.pth')
        if os.path.exists(_model_path):
            _class.model = torch.load(_model_path)
        return _class
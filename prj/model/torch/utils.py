from pathlib import Path
from typing import List, Union, Dict, Any
import torch
from torch.utils.data import DataLoader
import lightning as L
import lightning.pytorch.callbacks as C
from prj.model.torch.callback import EpochStatsCallback


def train(model: L.LightningModule,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          checkpoint_path: Union[str, Path] = None,
          max_epochs: int = 1000,
          eval_frequency: int = 1,
          log_every_n_steps: int = 10,
          precision: str = "32-mixed",
          accumulate_grad_batches: int = 1,
          gradient_clip_val: int = 10,
          gradient_clip_algorithm: str = 'norm',
          use_swa : bool = False,
          swa_cfg: Dict[str, Any] = None,
          use_early_stopping: bool = False,
          early_stopping_cfg: Dict[str, Any] = None,
          use_model_ckpt: bool = True,
          model_ckpt_cfg: Dict[str, Any] = None,
          seed: int = 42,
          accelerator: str = 'auto',
          compile: bool = False):
    if compile:
        model = torch.compile(model)

    L.seed_everything(seed, workers=True)

    callbacks = [C.ModelSummary(max_depth=3), 
                 C.LearningRateMonitor(logging_interval='epoch'),
                 EpochStatsCallback()]
    if use_swa:
        callbacks.append(C.StochasticWeightAveraging(**swa_cfg))
    if use_early_stopping:
        callbacks.append(C.EarlyStopping(**early_stopping_cfg))
    if use_model_ckpt:
        callbacks.append(C.ModelCheckpoint(**model_ckpt_cfg))

    trainer = L.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=eval_frequency,
        log_every_n_steps=log_every_n_steps,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        callbacks=callbacks,
        accelerator=accelerator
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader, ckpt_path=checkpoint_path)
    return model



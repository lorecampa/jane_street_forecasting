import sys
sys.path.append('/home/it4i-carlos00/jane_street_forecasting/prj/model/torch')
sys.path.append('/home/it4i-carlos00/jane_street_forecasting')

from wrappers import JaneStreetMultiStockGraphModel
from datasets import JaneStreetMultiStockDynamicGraphDataset
from models import StockGNNModel
from metrics import weighted_r2_score
from losses import WeightedMSELoss
from utils import train

import optuna
import argparse
from datetime import datetime
import os
import logging
import json
import polars as pl
from lightning.pytorch.accelerators import find_usable_cuda_devices
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import CSVLogger
import torch
import numpy as np
import gc
from pathlib import Path
from copy import copy

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def optimize_parameters(output_dir, train_dataset, val_dataset, study_name, n_trials, storage, 
                        n_gpu, n_gpu_per_trial, num_workers_per_dataloader):   
    def obj_function(trial):
        
        corr_threshold = trial.suggest_float("corr_threshold", 0.05, 0.3, step=0.01)
        train_ds = copy(train_dataset)
        train_ds.corr_thr = corr_threshold
        val_ds = copy(val_dataset)
        val_ds.corr_thr = corr_threshold
        
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096])
        train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers_per_dataloader)
        val_dataloader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=num_workers_per_dataloader)
        
        use_embeddings = trial.suggest_categorical("use_embeddings", [True, False]) 
        num_layers = trial.suggest_int("num_layers", 1, 12)
        dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5, step=0.005)
        embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64, 128]) if use_embeddings else 0
        dim_feedforward_mult = trial.suggest_categorical("dim_feedforward_mult", [2, 3, 4])
        num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
        hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step=8)
        base_model = StockGNNModel(
            input_features=79,
            output_dim=1,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            use_embeddings=use_embeddings,
            embedding_dim=embedding_dim,
            dim_feedforward_mult=dim_feedforward_mult)
        
        optimizer = 'Adam'
        optimizer_cfg = dict(lr=trial.suggest_float("lr", 1e-4, 1e-3, step=1e-4))
        scheduler = 'ReduceLROnPlateau'
        scheduler_cfg = dict(mode='max', factor=0.1, patience=5, verbose=True, min_lr=1e-5)
        model = JaneStreetMultiStockGraphModel(
            base_model, 
            losses=[WeightedMSELoss()], 
            loss_weights=[1], 
            l1_lambda=0, 
            l2_lambda=0,
            scheduler=scheduler, 
            scheduler_cfg=scheduler_cfg,
            optimizer=optimizer, 
            optimizer_cfg=optimizer_cfg)
        
        accumulate_grad_batches = trial.suggest_int('accumulate_grad_batches', 1, 8)
        gradient_clip_val = trial.suggest_float('gradient_clip_val', 0.1, 20, step=0.1)
        tmp_checkpoint_dir = os.path.join(output_dir, f'trial_{trial.number}')
        early_stopping = {'monitor': 'val_wr2', 'min_delta': 0.00, 'patience': 10, 'verbose': True, 'mode': 'max'}
        ckpt_config = {'dirpath': tmp_checkpoint_dir, 'filename': 'gat-epoch={epoch:03d}-val_wr2={val_wr2:.6f}', 'save_top_k': 1,
                       'monitor': 'val_wr2', 'verbose': True, 'mode': 'max'}
        use_swa = trial.suggest_categorical('use_swa', [True, False])
        if use_swa:
            swa_config = {'swa_lrs': trial.suggest_float('swa_lrs', 1e-3, 0.1, step=1e-3), 'swa_epoch_start': trial.suggest_int('swa_epoch_start', 5, 10)}
        else:
            swa_config = dict()
        logger = CSVLogger(os.path.join(output_dir, 'logs'), name=None, version=trial.number)
        model, best_model_path, best_epoch = train(model, train_dataloader, val_dataloader, max_epochs=100, precision='32-true', 
                                                   use_model_ckpt=True, gradient_clip_val=gradient_clip_val, use_early_stopping=True, 
                                                   early_stopping_cfg=early_stopping, model_ckpt_cfg=ckpt_config, model_name='gat',
                                                   accumulate_grad_batches=accumulate_grad_batches, log_every_n_steps=100, return_best_epoch=True,
                                                   use_swa=use_swa, swa_cfg=swa_config, logger=logger,
                                                   accelerator='cuda', devices=find_usable_cuda_devices(n_gpu_per_trial)) # gpus must be in exclusive mode
        trial.set_user_attr("epochs", best_epoch)
        
        del model
        gc.collect()
        
        val_dataloader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=num_workers_per_dataloader)
        base_model = StockGNNModel(
            input_features=79,
            output_dim=1,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            use_embeddings=use_embeddings,
            embedding_dim=embedding_dim,
            dim_feedforward_mult=dim_feedforward_mult)
        model = JaneStreetMultiStockGraphModel.load_from_checkpoint(
            best_model_path, 
            model=base_model,
            losses=[WeightedMSELoss()],
            loss_weights=[1])
        model.to(f'cuda:{find_usable_cuda_devices(1)[0]}')
        model.eval()
        
        y_hat = []
        y = []
        weights = []
        for x, targets, masks, w, s, adj in iter(val_dataloader):
            with torch.no_grad():
                preds = model(x.to(model.device), s.to(model.device), adj.to(model.device)).cpu()
            y_hat.append(preds.numpy().flatten())
            y.append(targets.numpy().flatten())
            weights.append(w.numpy().flatten())

        y = np.concatenate(y)
        y_hat = np.concatenate(y_hat)
        weights = np.concatenate(weights)
        score = weighted_r2_score(y_hat, y, weights)
        
        del model, y, y_hat, weights
        gc.collect()
        
        return score
    
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)    
    study.optimize(obj_function, n_trials=n_trials, n_jobs=n_gpu // n_gpu_per_trial)
    return study.best_params, study.trials_dataframe()
        

def main(dataset_path, output_dir, study_name, n_trials, storage, n_gpu, n_gpu_per_trial, num_workers_per_dataloader):
    
    train_ds = pl.concat([
        pl.scan_parquet(dataset_path / f'partition_id={i}' / 'part-0.parquet')
        for i in range(6, 9)
    ])
    val_ds = pl.scan_parquet(dataset_path / 'partition_id=9' / 'part-0.parquet')
    corr_matrices = np.load(dataset_path / 'correlations.npy')
    
    train_dataset = JaneStreetMultiStockDynamicGraphDataset(train_ds, corr_matrices)
    val_dataset = JaneStreetMultiStockDynamicGraphDataset(val_ds, corr_matrices)
    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    best_params, trials_df = optimize_parameters(output_dir, train_dataset, val_dataset, study_name, n_trials, 
                                                 storage, n_gpu, n_gpu_per_trial, num_workers_per_dataloader)
    
    params_file_path = os.path.join(output_dir, 'best_params.json')
    logging.info(f'Best parameters: {best_params}')
    logging.info(f'Saving the best parameters at: {params_file_path}')
    with open(params_file_path, 'w') as params_file:
        json.dump(best_params, params_file)    
        
    trials_file_path = os.path.join(output_dir, 'trials_dataframe.csv')
    logging.info(f'Saving the trials dataframe at: {trials_file_path}')
    trials_df.to_csv(trials_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script")
    parser.add_argument("-output_dir", default="/mnt/proj2/dd-24-8/frustum_datasets/last", type=str,
                        help="The directory where the models will be placed")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Parquet file where the training dataset is placed")
    parser.add_argument("-n_trials", default=100, type=int, required=False,
                        help="Number of optuna trials to perform")
    parser.add_argument("-study_name", default=None, type=str, required=False,
                        help="Optional name of the study. Should be used if a storage is provided")
    parser.add_argument("-storage", default=None, type=str, required=False,
                        help="Optional storage url for saving the trials")
    parser.add_argument('-n_gpus', type=int, default=8, 
                        help='The number of gpus to use')
    parser.add_argument('-n_gpus_per_trial', default=1, type=int,
                        help='The number of gpus to use for a single trial')
    parser.add_argument('-num_workers_per_dataloader', default=1, type=int,
                        help='The number of workers for a single dataloader')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    DATASET_DIR = args.dataset_path
    N_TRIALS = args.n_trials
    STUDY_NAME = args.study_name
    STORAGE = args.storage
    N_GPU = args.n_gpus
    N_GPU_PER_TRIAL = args.n_gpus_per_trial
    NUM_WORKERS = args.num_workers_per_dataloader
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'gat_tuning_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(Path(DATASET_DIR), output_dir, STUDY_NAME, N_TRIALS, STORAGE, N_GPU, N_GPU_PER_TRIAL, NUM_WORKERS)
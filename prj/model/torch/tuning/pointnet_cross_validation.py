import sys
sys.path.append('/home/it4i-carlos00/jane_street_forecasting/prj/model/torch')
sys.path.append('/home/it4i-carlos00/jane_street_forecasting')

from wrappers import JaneStreetMultiStockModel
from datasets import JaneStreetMultiStockDataset
from models import StockPointNetModel
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
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch
import numpy as np
import gc
from pathlib import Path

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def optimize_parameters(output_dir, train_datasets, val_datasets, study_name, n_trials, storage, 
                        n_gpu, n_gpu_per_trial, num_workers_per_dataloader):   
    def obj_function(trial):
        
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096])
        
        num_layers_extractor = trial.suggest_int("num_layers_extractor", 1, 6)
        final_hidden_dims_extractor = trial.suggest_int("final_hidden_dims_extractor", 16, 256, step=16)
        hidden_dims_extractor = [final_hidden_dims_extractor]
        if num_layers_extractor > 1:
            hidden_dims_decay_extractor = trial.suggest_float("hidden_dims_decay_extractor", 1, 3, step=0.1)
            for _ in range(num_layers_extractor - 1):
                hidden_dims_extractor = [int(hidden_dims_extractor[0] * hidden_dims_decay_extractor)] + hidden_dims_extractor
                
        num_layers_aggregator = trial.suggest_int("num_layers_aggregator", 1, 6)
        final_hidden_dims_aggregator = trial.suggest_int("final_hidden_dims_aggregator", 16, 256, step=16)
        hidden_dims_aggregator = [final_hidden_dims_aggregator]
        if num_layers_aggregator > 1:
            hidden_dims_decay_aggregator = trial.suggest_float("hidden_dims_decay_aggregator", 1, 3, step=0.1)
            for _ in range(num_layers_aggregator - 1):
                hidden_dims_aggregator = [int(hidden_dims_aggregator[0] * hidden_dims_decay_aggregator)] + hidden_dims_aggregator
        
        use_embeddings = trial.suggest_categorical("use_embeddings", [True, False]) 
        dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5)
        embedding_dim = trial.suggest_categorical("embedding_dim", [8, 16, 32, 64, 128]) if use_embeddings else 0
        bn_momentum = trial.suggest_float("bn_momentum", 0.001, 0.3, log=True)
        
        optimizer = 'AdamW'
        lr = trial.suggest_float("lr", 1e-4, 1e-3, step=1e-4)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 0.01, log=True)
        amsgrad = trial.suggest_categorical("amsgrad", [True, False])
        beta1 = trial.suggest_float("beta1", 0.8, 0.99, step=0.01)
        beta2 = trial.suggest_float("beta2", 0.9, 0.999, step=0.001)
        scheduler = trial.suggest_categorical('scheduler', ['ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'ExponentialLR'])
        if scheduler == 'CosineAnnealingWarmRestarts':
            T_0 = trial.suggest_int("T_0", 3, 10)
        elif scheduler == 'ExponentialLR':
            gamma = trial.suggest_float("gamma", 0.95, 0.999)
        
        accumulate_grad_batches = trial.suggest_int('accumulate_grad_batches', 1, 8)
        gradient_clip_val = trial.suggest_float('gradient_clip_val', 0.1, 50, step=0.1)
        use_swa = trial.suggest_categorical('use_swa', [True, False])
        if use_swa:
            swa_lrs = trial.suggest_float('swa_lrs', 1e-4, 0.1, log=True)
            swa_epoch_start = trial.suggest_int('swa_epoch_start', 2, 20)
            annealing_epochs = trial.suggest_int('annealing_epochs', 5, 20)
        else:
            swa_lrs, swa_epoch_start, annealing_epochs = None, None, None
        
        scores = []
        epochs = []
        for i in range(len(train_datasets)):
            train_dataloader = DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True, num_workers=num_workers_per_dataloader)
            val_dataloader = DataLoader(val_datasets[i], batch_size=2048, shuffle=False, num_workers=num_workers_per_dataloader)
            
            optimizer_cfg = dict(lr=lr, weight_decay=weight_decay, amsgrad=amsgrad, betas=(beta1, beta2))
            if scheduler == 'ReduceLROnPlateau':
                scheduler_cfg = dict(mode='max', factor=0.1, patience=5, verbose=True, min_lr=1e-5)
            elif scheduler == 'CosineAnnealingWarmRestarts':
                scheduler_cfg = dict(T_0=T_0, T_mult=1)
            else:
                scheduler_cfg = dict(gamma=gamma)
            
            base_model = StockPointNetModel(
                input_features=79,
                output_dim=1,
                hidden_dims_extractor=hidden_dims_extractor,
                hidden_dims_aggregator=hidden_dims_aggregator,
                dropout_rate=dropout_rate,
                use_embeddings=use_embeddings,
                embedding_dim=embedding_dim,
                bn_momentum=bn_momentum)
            model = JaneStreetMultiStockModel(
                base_model,
                losses=[WeightedMSELoss()],
                loss_weights=[1],
                l1_lambda=0, 
                l2_lambda=0,
                scheduler=scheduler,
                scheduler_cfg=scheduler_cfg,
                optimizer=optimizer,
                optimizer_cfg=optimizer_cfg)
            
            tmp_checkpoint_dir = os.path.join(output_dir, f'trial_{trial.number}_fold_{i}')
            early_stopping = {'monitor': 'val_wr2', 'min_delta': 0.00, 'patience': 15, 'verbose': True, 'mode': 'max'}
            ckpt_config = {'dirpath': tmp_checkpoint_dir, 'filename': 'pointnet-epoch={epoch:03d}-val_wr2={val_wr2:.6f}', 'save_top_k': 1,
                           'monitor': 'val_wr2', 'verbose': True, 'mode': 'max'}
            swa_config = {'swa_lrs': swa_lrs, 'swa_epoch_start': swa_epoch_start, 'annealing_epochs': annealing_epochs}
            logger = TensorBoardLogger(os.path.join(output_dir, 'logs'), name=None, version=f'trial_{trial.number}_fold_{i}')
            model, best_model_path, best_epoch = train(model, train_dataloader, val_dataloader, max_epochs=200, precision='32-true', 
                                                       use_model_ckpt=True, gradient_clip_val=gradient_clip_val, use_early_stopping=True, 
                                                       early_stopping_cfg=early_stopping, model_ckpt_cfg=ckpt_config, model_name='pointnet',
                                                       accumulate_grad_batches=accumulate_grad_batches, log_every_n_steps=100, return_best_epoch=True,
                                                       use_swa=use_swa, swa_cfg=swa_config, logger=logger, 
                                                       accelerator='cuda', devices=find_usable_cuda_devices(n_gpu_per_trial)) # gpus must be in exclusive mode
            epochs.append(best_epoch)
            
            del model
            gc.collect()
            
            val_dataloader = DataLoader(val_datasets[i], batch_size=2048, shuffle=False, num_workers=num_workers_per_dataloader)
            base_model = StockPointNetModel(
                input_features=79,
                output_dim=1,
                hidden_dims_extractor=hidden_dims_extractor,
                hidden_dims_aggregator=hidden_dims_aggregator,
                dropout_rate=dropout_rate,
                use_embeddings=use_embeddings,
                embedding_dim=embedding_dim,
                bn_momentum=bn_momentum)
            model = JaneStreetMultiStockModel.load_from_checkpoint(
                best_model_path, 
                model=base_model,
                losses=[WeightedMSELoss()],
                loss_weights=[1])
            model.to(f'cuda:{find_usable_cuda_devices(1)[0]}')
            model.eval()
            
            y_hat = []
            y = []
            weights = []
            for x, targets, masks, w, s in iter(val_dataloader):
                with torch.no_grad():
                    preds = model(x.to(model.device), masks.to(model.device), s.to(model.device)).cpu()
                y_hat.append(preds.numpy().flatten())
                y.append(targets.numpy().flatten())
                weights.append(w.numpy().flatten())

            y = np.concatenate(y)
            y_hat = np.concatenate(y_hat)
            weights = np.concatenate(weights)
            scores.append(weighted_r2_score(y_hat, y, weights))
            
            del model, y, y_hat, weights
            gc.collect()
        
        trial.set_user_attr("epochs", epochs)
        trial.set_user_attr("scores", scores)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)    
    study.optimize(obj_function, n_trials=n_trials, n_jobs=n_gpu // n_gpu_per_trial)
    return study.best_params, study.trials_dataframe()
        

def main(dataset_path, output_dir, study_name, n_trials, storage, n_gpu, n_gpu_per_trial, num_workers_per_dataloader):
    
    train_datasets = []
    val_datasets = []
    
    for start_partition in [2, 3, 4, 5, 6]:
        train_ds = pl.concat([
            pl.scan_parquet(dataset_path / f'partition_id={i}' / 'part-0.parquet')
            for i in range(start_partition, start_partition+3)
        ])
        val_ds = pl.scan_parquet(dataset_path / f'partition_id={start_partition+3}' / 'part-0.parquet')
        train_datasets.append(JaneStreetMultiStockDataset(train_ds))
        val_datasets.append(JaneStreetMultiStockDataset(val_ds))
    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    best_params, trials_df = optimize_parameters(output_dir, train_datasets, val_datasets, study_name, n_trials, 
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
    model_name = f'pointnet_tuning_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(Path(DATASET_DIR), output_dir, STUDY_NAME, N_TRIALS, STORAGE, N_GPU, N_GPU_PER_TRIAL, NUM_WORKERS)
import sys
sys.path.append('/home/it4i-carlos00/jane_street_forecasting/prj/model/torch')
sys.path.append('/home/it4i-carlos00/jane_street_forecasting')

from wrappers import JaneStreetBaseModel
from datasets import JaneStreetBaseDataset
from models import Mlp
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
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import numpy as np
import gc
from pathlib import Path
from copy import deepcopy

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
FEATURE_COLS = [f'feature_{i:02d}' for i in range(79)]
COLUMNS = FEATURE_COLS + ['date_id', 'time_id', 'symbol_id', 'weight', 'responder_6']


class DateGroupedDataset:
    def __init__(self, dataset: pl.DataFrame):
        self.dataset = dataset
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        
        self.X = torch.FloatTensor(self.dataset.select(feature_cols).to_numpy().astype(np.float32))
        self.y = torch.FloatTensor(self.dataset.select(['responder_6']).to_numpy().flatten().astype(np.float32))
        self.weights = torch.FloatTensor(self.dataset.select(['weight']).to_numpy().flatten().astype(np.float32))        
        
        self.grouped_indices = self.dataset.with_row_index().group_by("date_id").agg(pl.col('index'))['index'].to_list()
        
    def __len__(self):
        return len(self.grouped_indices)
    
    def __getitem__(self, idx):
        indices = self.grouped_indices[idx]
        return self.X[indices], self.y[indices], self.weights[indices]

    
def evaluate_model(model, val_dataset, device):
    ss_res = 0.0
    ss_tot = 0.0
    for idx in range(len(val_dataset)):
        x, targets, w = val_dataset[idx]
        with torch.no_grad():
            y_out = model(x.to(device)).squeeze()
        w = w.to(device)
        targets = targets.to(device)
        ss_res += (w * (y_out - targets) ** 2).sum().cpu()
        ss_tot += (w * (targets ** 2)).sum().cpu()
    return 1 - ss_res / ss_tot
    
    
def train_with_es(model, optimizer, train_dataset, val_dataset, epochs_max, gradient_clipping, loss_fn, is_weighted_loss, output_dir, es_patience, device):
    
    save_path = os.path.join(output_dir, 'best_model.pth')
    torch.save(model.state_dict(), save_path)
    best_score = -1e10
    best_epoch = -1
    for epoch in range(epochs_max):
        model.train()
        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)
        for idx in indices:
            x, targets, w = train_dataset[idx]
            optimizer.zero_grad()
            y_out = model.forward(x.to(device)).squeeze()
            if is_weighted_loss:
                loss = loss_fn(y_out, targets.to(device), w.to(device))
            else:
                loss = loss_fn(y_out, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            
        model.eval()
        score = evaluate_model(model, val_dataset, device).item()
        logging.info(f'Epoch {epoch} weighted r2: {score}')
        if score > best_score:
            torch.save(model.state_dict(), save_path)
            best_epoch = epoch
            best_score = score
        elif epoch - best_epoch >= es_patience:
            logging.info(f'Stopping after {epoch} epochs')
            break
        
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model = model.to(f'cuda:{device}')
    return model, score


def optimize_parameters(output_dir, pretrain_dataset, pretrain_val_dataset, online_learning_dataset, study_name, n_trials, storage, 
                        n_gpu, n_gpu_per_trial, num_workers_per_dataloader, start_eval_from=1360):   
    def obj_function(trial):
        
        logging.info(f'Trial {trial.number}')
        
        use_dropout = trial.suggest_categorical('use_dropout', [True, False])      
        use_norm = trial.suggest_categorical('use_norm', [True, False])
        initial_bn = trial.suggest_categorical('initial_bn', [True, False])
        dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5) if use_dropout else None
        bn_momentum = trial.suggest_float("bn_momentum", 0.001, 0.3, log=True) if use_norm or initial_bn else None
        model = Mlp(
            input_features=79,
            output_dim=1,
            hidden_dims=[512, 256],
            initial_bn=initial_bn,
            use_dropout=use_dropout,
            use_norm=use_norm,
            dropout_rate=dropout_rate,
            bn_momentum=bn_momentum)
        device = find_usable_cuda_devices(1)[0]
        model = model.to(f'cuda:{device}')
                
        gradient_clipping = trial.suggest_float('gradient_clipping', 0.1, 100, step=0.1)
        online_lr = trial.suggest_float("online_lr", 1e-7, 1e-5, step=1e-7)
        online_gradient_clipping = trial.suggest_float('online_gradient_clipping', 0.1, 50, step=0.1)
        
        optimizer = trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'SGD'])
        lr = trial.suggest_float("lr", 1e-5, 1e-3, step=1e-5)
        use_weight_decay = trial.suggest_categorical('use_weight_decay', [True, False]) if optimizer != 'AdamW' else True
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 0.01, log=True) if use_weight_decay else 0
        if optimizer != 'SGD':
            amsgrad = trial.suggest_categorical("amsgrad", [True, False])
            beta1 = trial.suggest_float("beta1", 0.8, 0.99, step=0.01)
            beta2 = trial.suggest_float("beta2", 0.9, 0.999, step=0.001)
            optimizer_cls = torch.optim.AdamW if optimizer == 'AdamW' else torch.optim.Adam
            optimizer_params = dict(lr=lr, weight_decay=weight_decay, amsgrad=amsgrad, betas=(beta1, beta2))
        else:
            momentum = trial.suggest_float('momentum', 0.8, 0.99)
            nesterov = trial.suggest_categorical('nesterov', [True, False])
            optimizer_cls = torch.optim.SGD
            optimizer_params = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
            
        use_weighted_loss = trial.suggest_categorical('use_weighted_loss', [True, False])
        loss_fn = WeightedMSELoss() if use_weighted_loss else nn.MSELoss()
        
        tmp_checkpoint_dir = os.path.join(output_dir, f'trial_{trial.number}')
        os.makedirs(tmp_checkpoint_dir)
        
        logging.info('Starting pretrain')
        optimizer = optimizer_cls(model.parameters(), **deepcopy(optimizer_params))
        train_dataloader = DateGroupedDataset(pretrain_dataset)
        val_dataloader = DateGroupedDataset(pretrain_val_dataset)
        model, initial_score = train_with_es(model, optimizer, train_dataloader, val_dataloader, 100, gradient_clipping, loss_fn, use_weighted_loss, tmp_checkpoint_dir, 5, device)
        trial.set_user_attr("initial_wr2_score", initial_score)
        model.eval()
        offline_score = evaluate_model(model, DateGroupedDataset(online_learning_dataset), device).item()
        trial.set_user_attr("offline_score", offline_score)
        logging.info(f'Offline evaluation: {offline_score}')
        
        optimizer_params['lr'] = online_lr
        optimizer = optimizer_cls(model.parameters(), **deepcopy(optimizer_params))
        
        replay_buffer = None
        
        y_hat = []
        y = []
        weights = []
        daily_r2 = []
        partition_9_id_start = 0
        
        for date_id, test in online_learning_dataset.group_by('date_id', maintain_order=True):
            
            if replay_buffer is not None:
                model.train()
                x_train = torch.tensor(replay_buffer.select(FEATURE_COLS).to_numpy(), dtype=torch.float32).to(device)
                y_train = torch.tensor(replay_buffer.select(['responder_6']).to_numpy().flatten(), dtype=torch.float32).to(device)
                if use_weighted_loss:
                    weights_train = torch.tensor(replay_buffer.select(['weight']).to_numpy().flatten(), dtype=torch.float32).to(device)
                optimizer.zero_grad()
                y_out = model.forward(x_train).squeeze()
                if use_weighted_loss:
                    loss = loss_fn(y_out, y_train, weights_train)
                else:
                    loss = loss_fn(y_out, y_train)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), online_gradient_clipping)
                optimizer.step()
                model.eval()

            test_ = test.select(COLUMNS)  
            replay_buffer = test_
            
            if date_id[0] == 1530:
                partition_9_id_start = len(y_hat)
                
            x = torch.tensor(test_.select(FEATURE_COLS).to_numpy(), dtype=torch.float32).to(device)
            with torch.no_grad():
                preds = model(x).cpu().numpy().flatten()
            y_hat.append(preds)
            y.append(test_.select(['responder_6']).to_numpy().flatten())
            weights.append(test_.select(['weight']).to_numpy().flatten())
            daily_r2.append(weighted_r2_score(preds, y[-1], weights[-1]))
        
        score = weighted_r2_score(np.concatenate(y_hat), np.concatenate(y), np.concatenate(weights))
        partition_9_score = weighted_r2_score(np.concatenate(y_hat[partition_9_id_start:]), np.concatenate(y[partition_9_id_start:]), np.concatenate(weights[partition_9_id_start:]))
        daily_r2 = np.array(daily_r2)
        sharpe = np.mean(daily_r2) / np.std(daily_r2)
        stability_index = np.sum(daily_r2 > 0) / daily_r2.shape[0]
        trial.set_user_attr("partition_9_score", partition_9_score)
        trial.set_user_attr("stability_index", stability_index)
        
        logging.info(f'r2={score}, sharpe_ratio={sharpe}, partition_9_r2={partition_9_score}, stability={stability_index}')
        
        del model, y, y_hat, weights, replay_buffer
        gc.collect()
        
        return score, sharpe
    
    study = optuna.create_study(directions=['maximize', 'maximize'], study_name=study_name, storage=storage, load_if_exists=True)    
    study.optimize(obj_function, n_trials=n_trials, n_jobs=n_gpu // n_gpu_per_trial)
    return study.trials_dataframe()
        

def main(dataset_path, output_dir, study_name, n_trials, storage, n_gpu, n_gpu_per_trial, num_workers_per_dataloader):
    
    pretraining_dataset = pl.concat(
        pl.scan_parquet(dataset_path / f'partition_id={i}' / 'part-0.parquet')
        for i in range(5, 8)
    ).select(COLUMNS).sort(['date_id', 'time_id', 'symbol_id']).fill_nan(0).fill_null(0)
    max_date = pretraining_dataset.select(pl.col('date_id').max()).collect().item()
    pretraining_val_dataset = pretraining_dataset.filter(pl.col('date_id') > max_date - 30).collect()
    pretraining_dataset = pretraining_dataset.filter(pl.col('date_id') <= max_date - 30).collect()

    online_learning_dataset = pl.concat(
        pl.scan_parquet(dataset_path / f'partition_id={i}' / 'part-0.parquet')
        for i in range(8, 10)
    ).sort(['date_id', 'time_id', 'symbol_id']).fill_nan(0).fill_null(0).collect()
        
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    trials_df = optimize_parameters(output_dir, pretraining_dataset, pretraining_val_dataset, online_learning_dataset, study_name, n_trials, 
                                    storage, n_gpu, n_gpu_per_trial, num_workers_per_dataloader)
    
    # params_file_path = os.path.join(output_dir, 'best_params.json')
    # logging.info(f'Best parameters: {best_params}')
    # logging.info(f'Saving the best parameters at: {params_file_path}')
    # with open(params_file_path, 'w') as params_file:
    #     json.dump(best_params, params_file)    
        
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
    model_name = f'mlp_online_tuning_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(Path(DATASET_DIR), output_dir, STUDY_NAME, N_TRIALS, STORAGE, N_GPU, N_GPU_PER_TRIAL, NUM_WORKERS)
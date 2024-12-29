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

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
FEATURE_COLS = [f'feature_{i:02d}' for i in range(79)]
COLUMNS = FEATURE_COLS + ['date_id', 'time_id', 'symbol_id', 'weight', 'responder_6']


class BaseDataset(Dataset):
    
    def __init__(self, dataset: pl.DataFrame):
        super(BaseDataset, self).__init__()   
        self.dataset = dataset
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        self.X = torch.FloatTensor(self.dataset.select(feature_cols).to_numpy().astype(np.float32))
        self.y = torch.FloatTensor(self.dataset.select(['responder_6']).to_numpy().flatten().astype(np.float32))
        self.weights = torch.FloatTensor(self.dataset.select(['weight']).to_numpy().flatten().astype(np.float32))        
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):       
        return self.X[idx], self.y[idx], self.weights[idx]


def optimize_parameters(output_dir, model_path, initial_dataset, online_learning_dataset, study_name, n_trials, storage, 
                        n_gpu, n_gpu_per_trial, num_workers_per_dataloader):   
    def obj_function(trial):
        
        base_model = Mlp(79, hidden_dims=[512, 256], dropout_rate=0.3, final_mult=5.0, use_tanh=True)
        model = JaneStreetBaseModel.load_from_checkpoint(model_path, model=base_model, losses=[WeightedMSELoss()], loss_weights=[1])
        device = find_usable_cuda_devices(1)[0]
        model = model.to(f'cuda:{device}')
        model.eval()
        
        # hyperparameters
        train_every = trial.suggest_int('train_every', 10, 40)
        n_epochs_per_train = trial.suggest_int('n_epochs_per_train', 1, 20)
        old_data_fraction = trial.suggest_float('old_data_fraction', 0.01, 0.9, step=0.01)
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
        gradient_clipping = trial.suggest_float('gradient_clipping', 0.1, 20, step=0.1)
        freeze_bn = trial.suggest_categorical("freeze_bn", [True, False])
        disable_dropout = trial.suggest_categorical("disable_dropout", [True, False])
        
        optimizer = trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'SGD'])
        lr = trial.suggest_float("lr", 1e-7, 1e-4, step=1e-7)
        use_weight_decay = trial.suggest_categorical('use_weight_decay', [True, False]) if optimizer != 'AdamW' else True
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 0.01, log=True) if use_weight_decay else 0
        if optimizer != 'SGD':
            amsgrad = trial.suggest_categorical("amsgrad", [True, False])
            beta1 = trial.suggest_float("beta1", 0.8, 0.99, step=0.01)
            beta2 = trial.suggest_float("beta2", 0.9, 0.999, step=0.001)
            optimizer_cls = torch.optim.AdamW if optimizer == 'AdamW' else torch.optim.Adam
            optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad, betas=(beta1, beta2))
        else:
            momentum = trial.suggest_float('momentum', 0.8, 0.99)
            nesterov = trial.suggest_categorical('nesterov', [True, False])
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
            
        use_weighted_loss = trial.suggest_categorical('use_weighted_loss', [True, False])
        loss_fn = WeightedMSELoss() if use_weighted_loss else nn.MSELoss()
        
        old_dataset = initial_dataset.clone()
        new_dataset = None
        date_idx = 0
        
        y_hat = []
        y = []
        weights = []
        daily_r2 = []
        partition_9_id_start = None
        
        # in this setting, we assume that in one predict we can do all the epochs (it is impossible to simulate otherwise,
        # due to different gpu environments), but is a reasonable approximation as in 9/10 time_ids a kaggle cpu env can
        # handle 5 epochs of a dataset of 1e6 samples for this exact same mlp model
        for date_id, test in online_learning_dataset.group_by('date_id', maintain_order=True):
            
            if (date_idx + 1) % train_every == 0:
                model.train()
                if freeze_bn:
                    for module in model.modules():
                        # freeze batch normalization parameters
                        if isinstance(module, nn.BatchNorm2d):
                            if hasattr(module, 'weight'):
                                module.weight.requires_grad_(False)
                            if hasattr(module, 'bias'):
                                module.bias.requires_grad_(False)
                            module.eval() # do not change the running mean and var
                if disable_dropout:
                    for module in model.modules():
                        if isinstance(module, nn.Dropout):
                            module.p = 0 # disable dropout by putting probability to zero
                            
                old_data_len = min(old_dataset.shape[0], old_data_fraction * new_dataset.shape[0] / (1 - old_data_fraction))
                train_dataloader = BaseDataset(pl.concat([old_dataset.sample(n=old_data_len), new_dataset]))
                train_dataloader = DataLoader(train_dataloader, shuffle=True, batch_size=batch_size, num_workers=num_workers_per_dataloader)
                # logger = CSVLogger(os.path.join(output_dir, 'logs'), name=None, version=f'trial_{trial.number}_date_id_{date_idx}')
                # model = train(model, train_dataloader, None, use_model_ckpt=False, accelerator='cuda', devices=[device], logger=logger, 
                #               log_every_n_steps=100, max_epochs=n_epochs_per_train)
                for _ in range(n_epochs_per_train):
                    for x, targets, w in train_dataloader:
                        optimizer.zero_grad()
                        y_out = model.forward(x.to(model.device)).squeeze()
                        if use_weighted_loss:
                            loss = loss_fn(y_out, targets.to(model.device), w.to(model.device))
                        else:
                            loss = loss_fn(y_out, targets.to(model.device))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        optimizer.step()
                model.eval()
                old_dataset = old_dataset.vstack(new_dataset)
                new_dataset = None

            date_idx += 1
            test_ = test.select(COLUMNS).fill_null(0).fill_nan(0)      
            new_dataset = test_ if new_dataset is None else new_dataset.vstack(test_)
            
            # if date_id[0] == 1530:
            #     partition_9_id_start = len(y_hat)
            
            x = torch.tensor(test_.select(FEATURE_COLS).to_numpy(), dtype=torch.float32).to(model.device)
            with torch.no_grad():
                preds = model(x).cpu().numpy().flatten()
            y_hat.append(preds)
            y.append(test_.select(['responder_6']).to_numpy().flatten())
            weights.append(test_.select(['weight']).to_numpy().flatten())
            daily_r2.append(weighted_r2_score(preds, y[-1], weights[-1]))
        
        score = weighted_r2_score(np.concatenate(y_hat), np.concatenate(y), np.concatenate(weights))
        # partition_9_score = weighted_r2_score(np.concatenate(y_hat[partition_9_id_start:]), np.concatenate(y[partition_9_id_start:]), np.concatenate(weights[partition_9_id_start:]))
        daily_r2 = np.array(daily_r2)
        sharpe = np.mean(daily_r2) / np.std(daily_r2)
        stability_index = np.sum(daily_r2 > 0) / daily_r2.shape[0]
        
        del model, y, y_hat, weights, old_dataset, new_dataset
        gc.collect()
        
        return score, sharpe, stability_index
    
    study = optuna.create_study(directions=['maximize', 'maximize', 'maximize'], study_name=study_name, storage=storage, load_if_exists=True)    
    study.optimize(obj_function, n_trials=n_trials, n_jobs=n_gpu // n_gpu_per_trial)
    return study.trials_dataframe()
        

def main(dataset_path, output_dir, study_name, n_trials, storage, n_gpu, n_gpu_per_trial, num_workers_per_dataloader):
    
    initial_dataset = pl.scan_parquet(dataset_path / 'partition_id=8' / 'part-0.parquet').select(COLUMNS) \
        .sort(['date_id', 'time_id', 'symbol_id']).fill_null(0).collect()
    online_learning_dataset = pl.scan_parquet(dataset_path / 'partition_id=9' / 'part-0.parquet').sort(['date_id', 'time_id', 'symbol_id']).collect()
    
    model_path = '/mnt/proj2/dd-24-8/frustum_datasets/last/kaggle_models/mlp_notebook_v64/model1/baseline_all.ckpt'
    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    trials_df = optimize_parameters(output_dir, model_path, initial_dataset, online_learning_dataset, study_name, n_trials, 
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
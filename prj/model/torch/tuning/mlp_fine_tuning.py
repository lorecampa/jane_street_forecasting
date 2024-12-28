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
        model = JaneStreetBaseModel.load_from_checkpoint(
            model_path, 
            model=base_model,
            losses=[WeightedMSELoss()],
            loss_weights=[1],
            l1_lambda=0,
            l2_lambda=0)
        device = find_usable_cuda_devices(1)[0]
        model = model.to(f'cuda:{device}')
        model.eval()
        
        # hyperparameters
        train_every = trial.suggest_int('train_every', 10, 40)
        n_epochs_per_train = trial.suggest_int('n_epochs_per_train', 1, 30)
        dataset_max_size = trial.suggest_int('dataset_max_size', 1000000, 10000000, step=1000000)
        batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192, 16384, 32768, 65536])
        lr = trial.suggest_float("lr", 1e-6, 1e-4, step=1e-6)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 0.01, log=True)
        amsgrad = trial.suggest_categorical("amsgrad", [True, False])
        beta1 = trial.suggest_float("beta1", 0.8, 0.99, step=0.01)
        beta2 = trial.suggest_float("beta2", 0.9, 0.999, step=0.001)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad, betas=(beta1, beta2))
        dataset = initial_dataset[-dataset_max_size:, :].clone()
        date_idx = 0
        
        y_hat = []
        y = []
        weights = []
        
        # in this setting, we assume that in one predict we can do all the epochs (it is impossible to simulate otherwise,
        # due to different gpu environments), but is a reasonable approximation as in 9/10 time_ids a kaggle cpu env can
        # handle 5 epochs of a dataset of 1e6 samples for this exact same mlp model
        for date_id, test in online_learning_dataset.group_by('date_id', maintain_order=True):
            
            if (date_idx + 1) % train_every == 0:
                model.train()
                train_dataloader = BaseDataset(dataset)
                train_dataloader = DataLoader(train_dataloader, shuffle=True, batch_size=batch_size, num_workers=num_workers_per_dataloader)
                # logger = CSVLogger(os.path.join(output_dir, 'logs'), name=None, version=f'trial_{trial.number}_date_id_{date_idx}')
                # model = train(model, train_dataloader, None, use_model_ckpt=False, accelerator='cuda', devices=[device], logger=logger, 
                #               log_every_n_steps=100, max_epochs=n_epochs_per_train)
                for _ in range(n_epochs_per_train):
                    for x, targets, w in train_dataloader:
                        optimizer.zero_grad()
                        y_out = model.forward(x.to(model.device)).squeeze()
                        loss = model._compute_loss(y_out, targets.to(model.device), w.to(model.device))
                        loss.backward()
                        optimizer.step()
                model.eval()

            date_idx += 1
            test_ = test.select(COLUMNS).fill_null(0).fill_nan(0)      
            dataset = dataset.vstack(test_)
            dataset = dataset[-dataset_max_size:]
            
            x = torch.tensor(test_.select(FEATURE_COLS).to_numpy(), dtype=torch.float32).to(model.device)
            with torch.no_grad():
                preds = model(x).cpu().numpy().flatten()
            y_hat.append(preds)
            y.append(test_.select(['responder_6']).to_numpy().flatten())
            weights.append(test_.select(['weight']).to_numpy().flatten())
        
        score = weighted_r2_score(y_hat, y, weights)
        
        del model, y, y_hat, weights, dataset
        gc.collect()
        
        return score
    
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage, load_if_exists=True)    
    study.optimize(obj_function, n_trials=n_trials, n_jobs=n_gpu // n_gpu_per_trial)
    return study.best_params, study.trials_dataframe()
        

def main(dataset_path, output_dir, study_name, n_trials, storage, n_gpu, n_gpu_per_trial, num_workers_per_dataloader):
    
    initial_dataset = pl.scan_parquet(dataset_path / 'partition_id=8' / 'part-0.parquet').select(COLUMNS) \
        .sort(['date_id', 'time_id', 'symbol_id']).fill_null(0).collect()
    online_learning_dataset = pl.scan_parquet(dataset_path / 'partition_id=9' / 'part-0.parquet').sort(['date_id', 'time_id', 'symbol_id']).collect()
    
    model_path = '/mnt/proj2/dd-24-8/frustum_datasets/last/kaggle_models/mlp_notebook_v64/model1/baseline_all.ckpt'
    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    best_params, trials_df = optimize_parameters(output_dir, model_path, initial_dataset, online_learning_dataset, study_name, n_trials, 
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
import numpy as np
import polars as pl
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


class JaneStreetMultiTimeDataset(Dataset):
    
    def __init__(self, dataset: pl.LazyFrame = None, num_stocks: int = 39, num_timesteps: int = 50, load=True):
        self.dataset = dataset
        self.num_stocks = num_stocks
        self.num_timesteps = num_timesteps
        if dataset is not None:
            self.dataset_len = self.dataset.select(['date_id', 'time_id', 'symbol_id']).unique().collect().shape[0]
        if load:
            self._load()
            
    def __copy__(self):
        dataset = JaneStreetMultiTimeDataset(None, self.num_stocks, self.num_timesteps, False)
        dataset.dataset_len = self.dataset_len
        dataset.X = self.X
        dataset.y = self.y
        dataset.weights = self.weights
        dataset.stocks = self.stocks
        dataset.stocks_start_id = self.stocks_start_id
        return dataset
    
    def _load(self):       
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        self.X = np.empty(shape=(0, 79), dtype=np.float32)
        self.y = np.empty(shape=(0,), dtype=np.float32)
        self.weights = np.empty(shape=(0,), dtype=np.float32)
        self.stocks = np.empty(shape=(0,), dtype=np.int8)
        self.stocks_start_id = {}
        for stock_id in self.dataset.select(pl.col('symbol_id').unique()).collect()['symbol_id'].to_list():
            stock_batch = (
                self.dataset
                .filter(pl.col('symbol_id') == stock_id) 
                .sort(['date_id', 'time_id'])
                .with_columns(pl.col(feature_cols).fill_null(strategy='forward', limit=10).over('symbol_id').fill_null(0))
            )
            X = stock_batch.select(feature_cols).collect().to_numpy().astype(np.float32)
            y = stock_batch.select(['responder_6']).collect().to_numpy().flatten().astype(np.float32)
            weights = stock_batch.select(['weight']).collect().to_numpy().flatten().astype(np.float32)
            self.stocks_start_id[stock_id] = self.X.shape[0]
            self.stocks = np.concatenate([self.stocks, np.ones(X.shape[0], dtype=np.int8) * stock_id], axis=0).astype(np.int8)
            self.X = np.concatenate([self.X, X], axis=0).astype(np.float32)
            self.y = np.concatenate([self.y, y], axis=0).astype(np.float32)
            self.weights = np.concatenate([self.weights, weights], axis=0).astype(np.float32)
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        stock_id = self.stocks[idx]
        start_period = max(self.stocks_start_id[stock_id], idx - self.num_timesteps)
        features = self.X[start_period:idx, :]
        targets = self.y[idx]
        weights = self.weights[idx]

        if features.shape[0] < self.num_timesteps:
            padding = np.zeros(shape=(self.num_timesteps - features.shape[0], features.shape[1]))
            features = np.concatenate([padding, features], axis=0)
        
        return (
            torch.tensor(features, dtype=torch.float32), 
            torch.tensor(targets, dtype=torch.float32), 
            torch.tensor(weights, dtype=torch.float32)
        )
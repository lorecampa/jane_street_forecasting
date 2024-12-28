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


class JaneStreetMultiStockDynamicGraphDataset(Dataset):
    
    def __init__(self, dataset: pl.LazyFrame, correlation_matrices: np.ndarray, num_stocks: int = 39, corr_thr: float = 0.1, load: bool = True):
        self.dataset = dataset
        self.abs_correlation_matrices = np.abs(correlation_matrices)
        self.num_stocks = num_stocks
        self.diag_inds = np.arange(self.num_stocks)
        self.corr_thr = corr_thr
        if dataset is not None:
            self.dataset_len = self.dataset.select(['date_id', 'time_id']).unique().collect().shape[0]
        if load:
            self._load()
        
    def __copy__(self):
        dataset = JaneStreetMultiStockDynamicGraphDataset(None, self.abs_correlation_matrices, self.num_stocks, self.corr_thr, False)
        dataset.dataset_len = self.dataset_len
        dataset.X = self.X
        dataset.y = self.y
        dataset.weights = self.weights
        dataset.date_ids = self.date_ids
        dataset.masks = self.masks
        dataset.s = self.s
        return dataset
    
    def _load(self):
        all_combinations = (
            self.dataset.select(['date_id', 'time_id'])
            .unique()
            .join(pl.DataFrame({'symbol_id': list(range(self.num_stocks))}, 
                               schema={'symbol_id': pl.Int8}).lazy(), how="cross")
        )
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        self.batch = (
            all_combinations
            .join(self.dataset.with_columns(pl.lit(1).alias('mask')) \
                      .sort(['date_id', 'time_id']) \
                      .with_columns(pl.col(feature_cols).fill_null(strategy='forward', limit=10) \
                                        .over('symbol_id').fill_null(0)), 
                  on=['date_id', 'time_id', 'symbol_id'], how="left")
            .fill_null(0)  # fill all columns with 0 for missing stocks (including the mask)
            .sort(['date_id', 'time_id', 'symbol_id'])
        )
        # num_stocks rows for each date and time
        self.X = self.batch.select(feature_cols).collect().to_numpy().astype(np.float32)
        self.y = self.batch.select(['responder_6']).collect().to_numpy().flatten().astype(np.float32)
        self.s = self.batch.select(['symbol_id']).collect().to_numpy().flatten().astype(np.int32)
        self.date_ids = self.batch.select(['date_id']).collect().to_numpy().flatten()
        self.masks = self.batch.select(['mask']).collect().to_numpy().flatten() == 0
        self.weights = self.batch.select(['weight']).collect().to_numpy().flatten().astype(np.float32)
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        start_row = idx * self.num_stocks
        features = self.X[start_row:start_row+self.num_stocks, :]
        targets = self.y[start_row:start_row+self.num_stocks]
        masks = self.masks[start_row:start_row+self.num_stocks]
        weights = self.weights[start_row:start_row+self.num_stocks]
        symbols = self.s[start_row:start_row+self.num_stocks]

        date_id = self.date_ids[start_row]
        adj_matrix = self.abs_correlation_matrices[date_id].copy()
        adj_matrix[self.diag_inds, self.diag_inds] = 0
        adj_matrix = (adj_matrix > self.corr_thr).astype(np.int32)
        
        return (
            torch.tensor(features), 
            torch.tensor(targets), 
            torch.tensor(masks), 
            torch.tensor(weights), 
            torch.tensor(symbols),
            torch.tensor(adj_matrix, dtype=torch.int)
        )
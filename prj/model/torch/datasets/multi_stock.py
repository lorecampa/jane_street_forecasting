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


class JaneStreetMultiStockDataset(Dataset):
    
    def __init__(self, dataset: pl.LazyFrame, num_days_batch: int = 10, num_stocks: int = 39):
        self.dataset = dataset
        self.num_days_batch = num_days_batch
        self.num_stocks = num_stocks

        self.max_date = self.dataset.select(pl.col('date_id').max()).collect().item()
        self.min_date = self.dataset.select(pl.col('date_id').min()).collect().item()
        self.batches = np.array(list(range(self.min_date, self.max_date + 1, self.num_days_batch)))
        self._shuffle_batches()
        self.current_batch_idx = -1
        self.dataset_len = self.dataset.select(['date_id', 'time_id']).unique().collect().shape[0]
        self.current_start_row = None
        self.X = None
        self.y = None
        self.date_ids = None
        self.masks = None
        self.weights = None
    
    def _shuffle_batches(self):
        np.random.shuffle(self.batches)
    
    def _load_batch(self):
        self.current_batch_idx += 1
        del self.X, self.y, self.masks, self.date_ids, self.weights
        gc.collect()
        
        if self.current_batch_idx >= len(self.batches):
            self.current_batch_idx = 0
            self._shuffle_batches()
        
        start_date_id = self.batches[self.current_batch_idx]
        filtered_dataset = (
            self.dataset
            .filter(pl.col('date_id').is_between(start_date_id, start_date_id+self.num_days_batch))
        )
        all_combinations = (
            filtered_dataset.select(['date_id', 'time_id'])
            .unique()
            .join(pl.DataFrame({'symbol_id': list(range(self.num_stocks))}, 
                               schema={'symbol_id': pl.Int8}).lazy(), how="cross")
        )
        feature_cols = [f'feature_{i:02d}' for i in range(79)]
        self.batch = (
            all_combinations
            .join(filtered_dataset.with_columns(pl.lit(1).alias('mask')) \
                      .sort(['date_id', 'time_id']) \
                      .with_columns(pl.col(feature_cols).fill_null(strategy='forward', limit=10) \
                                        .over('symbol_id').fill_null(0)), 
                  on=['date_id', 'time_id', 'symbol_id'], how="left")
            .fill_null(0)  # fill all columns with 0 for missing stocks (including the mask)
            .sort(['date_id', 'time_id', 'symbol_id'])
        )
        # 39 rows for each date and time
        self.X = self.batch.select(feature_cols).collect().to_numpy().astype(np.float32)
        self.y = self.batch.select(['responder_6']).collect().to_numpy().flatten().astype(np.float32)
        self.date_ids = self.batch.select(['date_id']).collect().to_numpy().flatten()
        self.masks = self.batch.select(['mask']).collect().to_numpy().flatten() == 0
        self.weights = self.batch.select(['weight']).collect().to_numpy().flatten().astype(np.float32)
        self.current_start_row = 0
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        if self.current_start_row is None or self.current_start_row >= self.X.shape[0]:
            self._load_batch()
        
        features = self.X[self.current_start_row:self.current_start_row+self.num_stocks, :]
        targets = self.y[self.current_start_row:self.current_start_row+self.num_stocks]
        masks = self.masks[self.current_start_row:self.current_start_row+self.num_stocks]
        weights = self.weights[self.current_start_row:self.current_start_row+self.num_stocks]

        self.current_start_row += self.num_stocks
        
        return torch.tensor(features), torch.tensor(targets), torch.tensor(masks), torch.tensor(weights)
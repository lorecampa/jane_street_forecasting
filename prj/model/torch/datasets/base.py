import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
import gc
BASE_FEATURES = [f'feature_{i:02d}' for i in range(79)]

class JaneStreetBatchDataset(Dataset):
    
    def __init__(
        self, 
        dataset: pl.LazyFrame, 
        shuffle: bool = True,
        features: list[str] = BASE_FEATURES,
        num_days_batch: int = 10,):
        super(JaneStreetBaseDataset, self).__init__()   
        self.dataset = dataset
        self.num_days_batch = num_days_batch
        self.shuffle = shuffle
        self.features = features

        self.max_date = self.dataset.select(pl.col('date_id').max()).collect().item()
        self.min_date = self.dataset.select(pl.col('date_id').min()).collect().item()
        self.num_batches = len(list(range(self.min_date, self.max_date + 1, self.num_days_batch)))
        self.dates = list(range(self.min_date, self.max_date + 1))
        if self.shuffle:
            self._shuffle_batches()
        self.current_batch_idx = -1
        self.dataset_len = self.dataset.select(['date_id', 'time_id', 'symbol_id']).unique().collect().shape[0]
        self.current_row = None
        self.X = None
        self.y = None
        self.weights = None
        self.indices = None
    
    def _shuffle_batches(self):
        np.random.shuffle(self.dates)
    
    def _shuffle_indices(self):
        np.random.shuffle(self.indices)
    
    def _load_batch(self):
        self.current_batch_idx += 1
        del self.X, self.y, self.weights
        gc.collect()
        
        if self.current_batch_idx >= self.num_batches:
            self.current_batch_idx = 0
            if self.shuffle:
                self._shuffle_batches()

        # start_date_id = self.batches[self.current_batch_idx]
        batch_dates = self.dates[self.current_batch_idx*self.num_days_batch:(self.current_batch_idx+1)*self.num_days_batch]
        filtered_dataset = (
            self.dataset
            .filter(pl.col('date_id').is_in(batch_dates))
            .sort(['date_id', 'time_id'])
        )
        
        self.X = filtered_dataset.select(self.features).collect().to_numpy().astype(np.float32)
        self.y = filtered_dataset.select(['responder_6']).collect().to_numpy().flatten().astype(np.float32)
        self.weights = filtered_dataset.select(['weight']).collect().to_numpy().flatten().astype(np.float32)
        self.indices = np.arange(self.X.shape[0])
        if self.shuffle:
            self._shuffle_indices()
        self.current_row = 0
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        if self.current_row is None or self.current_row >= self.X.shape[0]:
            self._load_batch()

        row_id = self.indices[self.current_row]
        features = self.X[row_id, :]
        targets = self.y[row_id]
        weights = self.weights[row_id]

        self.current_row += 1
        
        return (
            torch.tensor(features, dtype=torch.float32), 
            torch.tensor(targets, dtype=torch.float32), 
            torch.tensor(weights, dtype=torch.float32)
        )

class JaneStreetBaseDataset(Dataset):
    
    def __init__(self, 
                 dataset: pl.LazyFrame,
                 features: list[str] = BASE_FEATURES,
                 device: str = 'cpu'):
        super(JaneStreetBaseDataset, self).__init__()   
        
        self.dataset = dataset
        self.features = features
        self.device = device
        self.times_col = ['date_id', 'time_id']
        self._load()
        
    def _load(self):
        preprocessed_dataset = (
            self.dataset \
            .select(self.times_col + self.features + ['responder_6', 'weight']) \
            .sort(self.times_col)
        )
        self.X = torch.FloatTensor(preprocessed_dataset.select(self.features).cast(pl.Float32).collect().to_numpy()).to(self.device)
        self.y = torch.FloatTensor(preprocessed_dataset.select(['responder_6']).cast(pl.Float32).collect().to_numpy().flatten()).to(self.device)
        self.weights = torch.FloatTensor(preprocessed_dataset.select(['weight']).cast(pl.Float32).collect().to_numpy().flatten()).to(self.device)

        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):       
        return self.X[idx], self.y[idx], self.weights[idx]
        


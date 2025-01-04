import argparse
import ast
from datetime import timedelta
import json
from pathlib import Path
import pickle
import random
import typing
import polars as pl
import gc
import numpy as np
import polars.selectors as cs
import tensorflow as tf
import torch as th
from sklearn.model_selection import BaseCrossValidator
from tqdm import tqdm



def reduce_mem_usage(df: pl.DataFrame, verbose:bool = False):
    start_mem = df.estimated_size('mb')
    if verbose:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        # Integer types
        if col_type in [pl.Int16, pl.Int32, pl.Int64]:
            c_min = df[col].fill_null(0).min()
            c_max = df[col].fill_null(0).max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(pl.col(col).cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(pl.col(col).cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
        elif col_type in [pl.UInt16, pl.UInt32, pl.UInt64]:
            c_min = df[col].fill_null(0).min()
            c_max = df[col].fill_null(0).max()
            if c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                df = df.with_columns(pl.col(col).cast(pl.UInt8))
            elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                df = df.with_columns(pl.col(col).cast(pl.UInt16))
            elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                df = df.with_columns(pl.col(col).cast(pl.UInt32))
        # Float types
        elif col_type == pl.Float64:
            c_min = df[col].fill_null(0).min()
            c_max = df[col].fill_null(0).max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df = df.with_columns(pl.col(col).cast(pl.Float32))
        # List types
        elif col_type in [pl.List(pl.Int16), pl.List(pl.Int32), pl.List(pl.Int64)]:
            c_min = df[col].list.min().min()
            c_max = df[col].list.max().max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.Int8)))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.Int16)))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.Int32)))
        elif col_type in [pl.List(pl.UInt16), pl.List(pl.UInt32), pl.List(pl.UInt64)]:
            c_min = df[col].list.min().min()
            c_max = df[col].list.max().max()
            if c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.UInt8)))
            elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.UInt16)))
            elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.UInt32)))

    gc.collect()
    if verbose:
        end_mem = df.estimated_size('mb')
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)
    
    # Set the tensorflow seed
    tf.random.set_seed(seed)

    
    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        
        # Tensorflow determinism
        tf.config.experimental.enable_op_determinism()
    

def load_json(path: str | Path) -> dict | None:
    with open(path, 'r') as file:
        return json.load(file)

def load_pickle(path: str | Path) -> dict | None:
    with open(path, 'rb') as file:
        return pickle.load(file)
    
def save_dict_to_json(d: dict, path: str | Path, indent=4):
    if isinstance(path, str):
        path = Path(path)

    with open(path, 'w') as f:
        json.dump(d, f, indent=indent)

def save_dict_to_pickle(data_dict: dict, path: str | Path):
    if isinstance(path, str):
        path = Path(path)
    with open(path, 'wb') as f:
        pickle.dump(data_dict, f)  
        

def merge_dicts(dicts: list[dict]) -> dict:
    final_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key not in final_dict:
                final_dict[key] = []
            final_dict[key].append(value)
    return final_dict


def check_for_inf(df: pl.DataFrame):
    rows_with_inf = df.select(cs.numeric().is_infinite()).select(
        pl.sum_horizontal(pl.all()).alias('sum_infinite')
    ).filter(pl.col('sum_infinite') > 0).shape[0]

    cols_with_inf = df.select(cs.numeric().is_infinite())\
        .sum().transpose(include_header=True, header_name='column', column_names=['sum_infinite'])\
        .filter(pl.col('sum_infinite') > 0).to_dicts()
    return rows_with_inf, cols_with_inf


                
def build_rolling_stats(df: pl.DataFrame, cols: list = None, window: int = 30, group_by: list[str] = ['symbol_id']) -> pl.DataFrame:
    time_cols = ['date_id', 'time_id']
    df = df.select(time_cols + group_by + cols).sort(time_cols)
    timestep_id_df = df.select(time_cols).unique(maintain_order=True).with_row_index('index')
    df = df.join(timestep_id_df, on=time_cols, how='left')
    
    df = df.rolling(
        'index',
        period=f"{window}i",
        group_by=group_by
    ).agg(
        pl.col(cols).mean().name.suffix('_rm'),
        pl.col(cols).std(ddof=0).name.suffix('_rstd')
    )
    return df.join(timestep_id_df, on='index', how='left').drop('index')
    

    

def moving_z_score_norm(df: pl.DataFrame, rolling_stats_df: pl.DataFrame, cols: list, eps:float=1e-6, clip_bound:typing.Optional[float]=None, group_by:list[str]=['symbol_id']) -> pl.DataFrame:
    assert eps > 0, 'eps must be greater than 0'
    assert all({f'{col}_rm', f'{col}_rstd'}.issubset(rolling_stats_df.columns) for col in cols), 'One or more columns not found in the rolling stats DataFrame'    
    assert clip_bound is None or clip_bound >= 0, "Clip bound should be None or greater that zero"
    
    time_cols = ['date_id', 'time_id']
    

    df = df.join(rolling_stats_df, how='left', on=time_cols + group_by)\
        .with_columns(
            *[((pl.col(col) - pl.col(f'{col}_rm')) / (pl.col(f'{col}_rstd') + eps))\
                .clip(
                    lower_bound=-clip_bound if clip_bound else None, 
                    upper_bound=clip_bound if clip_bound else None
                ).alias(col) for col in cols]
        ).drop([col for col in rolling_stats_df.columns if col not in time_cols + group_by])
    
    return df
    
    
def get_null_count(df: pl.DataFrame):
    n_rows = df.shape[0]
    return df.fill_nan(None).null_count().transpose(include_header=True).sort('column_0', descending=True).rename({'column_0': 'count'}).with_columns(
        pl.col('count').truediv(n_rows).mul(100).alias('count (%)')
    )
    

def interquartile_mean(data: np.ndarray, q_min: int = 25, q_max: int = 75, axis=0) -> float:
    assert data.ndim == 1, "Input data must be 1D"
    sorted_data = np.sort(data, axis=axis)
    
    q_min = np.percentile(sorted_data, q_min, axis=axis)
    q_max = np.percentile(sorted_data, q_max, axis=axis)
    filtered_data = sorted_data[(sorted_data >= q_min) & (sorted_data <= q_max)]    
    iqm = np.mean(filtered_data, axis=0)
    return iqm

def str_to_dict_arg(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary string: {string}") from e
    

def wrapper_pbar(pbar, func):
    def foo(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)
    return foo


class BlockingTimeSeriesSplit():
    def __init__(self, n_splits:int, val_ratio:float = 0.2):
        assert val_ratio > 0 and val_ratio < 1, "val_ratio must be in the range (0, 1)"
        self.n_splits = n_splits
        self.val_ratio = val_ratio
    
    def get_n_splits(self, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
    
        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int((1-self.val_ratio) * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


class CombinatorialPurgedKFold(BaseCrossValidator):
    def __init__(self, n_splits=5, purge_length=1):
        self.n_splits = n_splits
        self.purge_length = purge_length

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        test_size = n_samples // self.n_splits
        test_starts = [(i * test_size) for i in range(self.n_splits)]

        for test_start in test_starts:
            test_end = test_start + test_size
            if test_end + self.purge_length >= n_samples:
                continue  # Skip if purge length exceeds data boundaries

            test_indices = indices[test_start:test_end]
            train_indices = np.setdiff1d(indices, test_indices)
            purge_start = max(0, test_start - self.purge_length)
            purge_end = min(n_samples, test_end + self.purge_length)
            purge_indices = np.arange(purge_start, purge_end)

            train_indices = np.setdiff1d(train_indices, purge_indices)
            yield train_indices, test_indices
            

def online_iterator(df: pl.DataFrame, show_progress: bool = True):
    assert df.select('date_id').n_unique() > 1, 'Dataset must contain at least 2 days'
    
    df_date_time_id = df.select('date_id', 'time_id').unique().sort('date_id', 'time_id').with_row_index('date_time_id')
    df = df.join(df_date_time_id, on=['date_id', 'time_id'], how='left', maintain_order='left').with_row_index('row_id')
    
    max_date_time_id = df_date_time_id['date_time_id'].max()
    min_date_id = df.select('date_id').min().item()
    
    responders = [f'responder_{i}' for i in range(9)]
    
    curr_idx:int = df_date_time_id.filter(pl.col('date_id').eq(min_date_id + 1))['date_time_id'].min()
    old_day = min_date_id

    
    with tqdm(total=max_date_time_id - curr_idx + 1, disable=not show_progress) as pbar:
        while curr_idx <= max_date_time_id:
            curr_day = df_date_time_id[curr_idx]['date_id'].item()
            is_new_day = curr_day != old_day
            lags = None
            if is_new_day:
                lags = df.filter(pl.col('date_id').eq(old_day)).select(pl.col('date_id').add(1), 'time_id', 'symbol_id', *[pl.col(r).alias(f'{r}_lag_1') for r in responders])
            
            old_day = curr_day

            batch = df.filter(pl.col('date_time_id').eq(curr_idx)).with_columns(pl.lit(True).alias('is_scored')).drop('date_time_id')
            
            yield batch, lags if lags is not None else None
            
            curr_idx += 1
            pbar.update(1)
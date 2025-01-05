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
    df = df.join(df_date_time_id, on=['date_id', 'time_id'], how='left').sort('date_id', 'time_id', 'symbol_id').with_row_index('row_id')
    
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

            batch = df.filter(pl.col('date_time_id').eq(curr_idx)).with_columns(pl.lit(True).alias('is_scored')).drop('date_time_id', *responders)
            
            yield batch, lags if lags is not None else None
            
            curr_idx += 1
            pbar.update(1)
    

def online_iterator_daily(df: pl.DataFrame, show_progress: bool = True):
    assert df.select('date_id').n_unique() > 1, 'Dataset must contain at least 2 days'
    
    df = df.sort('date_id', 'time_id', 'symbol_id').with_row_index('row_id').with_columns(pl.lit(True).alias('is_scored'))
    
    df = df.with_columns(
        pl.col('date_id').sub(pl.col('date_id').min().add(1))
    )
    max_date_id = df['date_id'].max()
    
    responders = [f'responder_{i}' for i in range(9)]
    
    curr_date_idx = 0
    n_dates = df['date_id'].n_unique()
    
    with tqdm(total=n_dates-1, disable=not show_progress) as pbar:
        while curr_date_idx <= max_date_id:
            lags = df.filter(pl.col('date_id').eq(curr_date_idx-1)).select(pl.col('date_id').add(1), 'time_id', 'symbol_id', *[pl.col(r).alias(f'{r}_lag_1') for r in responders])
            batch = df.filter(pl.col('date_id').eq(curr_date_idx)).drop(*responders)
            
            yield batch, lags
            
            curr_date_idx += 1
            pbar.update(1)
            
            

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from lightgbm.callback import CallbackEnv, _format_eval_result, _should_enable_early_stopping
from datetime import datetime
from lightgbm import EarlyStopException
from lightgbm.basic import (
    Booster,
    _ConfigAliases,
    _LGBM_BoosterEvalMethodResultType,
    _LGBM_BoosterEvalMethodResultWithStandardDeviationType,
    _log_info,
    _log_warning,
)

if TYPE_CHECKING:
    from lightgbm.engine import CVBooster

_ListOfEvalResultTuples = Union[
    List[_LGBM_BoosterEvalMethodResultType],
    List[_LGBM_BoosterEvalMethodResultWithStandardDeviationType],
]

class LGBMEarlyStoppingCallbackWithTimeout:
    """Internal early stopping callable class."""

    def __init__(
        self,
        stopping_rounds: int,
        first_metric_only: bool = False,
        verbose: bool = True,
        min_delta: Union[float, List[float]] = 0.0,
        timeout_seconds: Optional[int] = None
    ) -> None:
        self.enabled = _should_enable_early_stopping(stopping_rounds)

        self.order = 30
        self.before_iteration = False
        
        self.stopping_rounds = stopping_rounds
        self.first_metric_only = first_metric_only
        self.verbose = verbose
        self.min_delta = min_delta
        self.timeout_seconds = timeout_seconds
        self._start = datetime.utcnow()

        self._reset_storages()

    def _reset_storages(self) -> None:
        self.best_score: List[float] = []
        self.best_iter: List[int] = []
        self.best_score_list: List[_ListOfEvalResultTuples] = []
        self.cmp_op: List[Callable[[float, float], bool]] = []
        self.first_metric = ""

    def _gt_delta(self, curr_score: float, best_score: float, delta: float) -> bool:
        return curr_score > best_score + delta

    def _lt_delta(self, curr_score: float, best_score: float, delta: float) -> bool:
        return curr_score < best_score - delta

    def _is_train_set(self, ds_name: str, eval_name: str, env: CallbackEnv) -> bool:
        """Check, by name, if a given Dataset is the training data."""
        # for lgb.cv() with eval_train_metric=True, evaluation is also done on the training set
        # and those metrics are considered for early stopping
        if ds_name == "cv_agg" and eval_name == "train":
            return True

        # for lgb.train(), it's possible to pass the training data via valid_sets with any eval_name
        if isinstance(env.model, Booster) and ds_name == env.model._train_data_name:
            return True

        return False

    def _init(self, env: CallbackEnv) -> None:
        if env.evaluation_result_list is None or env.evaluation_result_list == []:
            raise ValueError("For early stopping, at least one dataset and eval metric is required for evaluation")

        is_dart = any(env.params.get(alias, "") == "dart" for alias in _ConfigAliases.get("boosting"))
        if is_dart:
            self.enabled = False
            _log_warning("Early stopping is not available in dart mode")
            return

        # validation sets are guaranteed to not be identical to the training data in cv()
        if isinstance(env.model, Booster):
            only_train_set = len(env.evaluation_result_list) == 1 and self._is_train_set(
                ds_name=env.evaluation_result_list[0][0],
                eval_name=env.evaluation_result_list[0][1].split(" ")[0],
                env=env,
            )
            if only_train_set:
                self.enabled = False
                _log_warning("Only training set found, disabling early stopping.")
                return

        if self.verbose:
            _log_info(f"Training until validation scores don't improve for {self.stopping_rounds} rounds")

        self._reset_storages()

        n_metrics = len({m[1] for m in env.evaluation_result_list})
        n_datasets = len(env.evaluation_result_list) // n_metrics
        if isinstance(self.min_delta, list):
            if not all(t >= 0 for t in self.min_delta):
                raise ValueError("Values for early stopping min_delta must be non-negative.")
            if len(self.min_delta) == 0:
                if self.verbose:
                    _log_info("Disabling min_delta for early stopping.")
                deltas = [0.0] * n_datasets * n_metrics
            elif len(self.min_delta) == 1:
                if self.verbose:
                    _log_info(f"Using {self.min_delta[0]} as min_delta for all metrics.")
                deltas = self.min_delta * n_datasets * n_metrics
            else:
                if len(self.min_delta) != n_metrics:
                    raise ValueError("Must provide a single value for min_delta or as many as metrics.")
                if self.first_metric_only and self.verbose:
                    _log_info(f"Using only {self.min_delta[0]} as early stopping min_delta.")
                deltas = self.min_delta * n_datasets
        else:
            if self.min_delta < 0:
                raise ValueError("Early stopping min_delta must be non-negative.")
            if self.min_delta > 0 and n_metrics > 1 and not self.first_metric_only and self.verbose:
                _log_info(f"Using {self.min_delta} as min_delta for all metrics.")
            deltas = [self.min_delta] * n_datasets * n_metrics

        # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
        self.first_metric = env.evaluation_result_list[0][1].split(" ")[-1]
        for eval_ret, delta in zip(env.evaluation_result_list, deltas):
            self.best_iter.append(0)
            if eval_ret[3]:  # greater is better
                self.best_score.append(float("-inf"))
                self.cmp_op.append(partial(self._gt_delta, delta=delta))
            else:
                self.best_score.append(float("inf"))
                self.cmp_op.append(partial(self._lt_delta, delta=delta))

    def _final_iteration_check(self, env: CallbackEnv, eval_name_splitted: List[str], i: int) -> None:
        if env.iteration == env.end_iteration - 1:
            if self.verbose:
                best_score_str = "\t".join([_format_eval_result(x, show_stdv=True) for x in self.best_score_list[i]])
                _log_info(
                    "Did not meet early stopping. " f"Best iteration is:\n[{self.best_iter[i] + 1}]\t{best_score_str}"
                )
                if self.first_metric_only:
                    _log_info(f"Evaluated only: {eval_name_splitted[-1]}")
            raise EarlyStopException(self.best_iter[i], self.best_score_list[i])

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if not self.enabled:
            return
        if env.evaluation_result_list is None:
            raise RuntimeError(
                "early_stopping() callback enabled but no evaluation results found. This is a probably bug in LightGBM. "
                "Please report it at https://github.com/microsoft/LightGBM/issues"
            )
        # self.best_score_list is initialized to an empty list
        first_time_updating_best_score_list = self.best_score_list == []
        for i in range(len(env.evaluation_result_list)):
            score = env.evaluation_result_list[i][2]
            if first_time_updating_best_score_list or self.cmp_op[i](score, self.best_score[i]):
                self.best_score[i] = score
                self.best_iter[i] = env.iteration
                if first_time_updating_best_score_list:
                    self.best_score_list.append(env.evaluation_result_list)
                else:
                    self.best_score_list[i] = env.evaluation_result_list
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            eval_name_splitted = env.evaluation_result_list[i][1].split(" ")
            if self.first_metric_only and self.first_metric != eval_name_splitted[-1]:
                continue  # use only the first metric for early stopping
            if self._is_train_set(
                ds_name=env.evaluation_result_list[i][0],
                eval_name=eval_name_splitted[0],
                env=env,
            ):
                continue  # train data for lgb.cv or sklearn wrapper (underlying lgb.train)
            elif env.iteration - self.best_iter[i] >= self.stopping_rounds:
                if self.verbose:
                    eval_result_str = "\t".join(
                        [_format_eval_result(x, show_stdv=True) for x in self.best_score_list[i]]
                    )
                    _log_info(f"Early stopping, best iteration is:\n[{self.best_iter[i] + 1}]\t{eval_result_str}")
                    if self.first_metric_only:
                        _log_info(f"Evaluated only: {eval_name_splitted[-1]}")
                raise EarlyStopException(self.best_iter[i], self.best_score_list[i])
            elif self.timeout_seconds is not None and (datetime.utcnow() - self._start).total_seconds() > self.timeout_seconds:
                if self.verbose:
                    eval_result_str = "\t".join(
                        [_format_eval_result(x, show_stdv=True) for x in self.best_score_list[i]]
                    )
                    _log_info(f"Early stopping, best iteration is:\n[{self.best_iter[i] + 1}]\t{eval_result_str}. Stop due to timeout, reached {self.timeout_seconds} seconds.")
                    if self.first_metric_only:
                        _log_info(f"Evaluated only: {eval_name_splitted[-1]}")
                raise EarlyStopException(self.best_iter[i], self.best_score_list[i])
            
            self._final_iteration_check(env, eval_name_splitted, i)
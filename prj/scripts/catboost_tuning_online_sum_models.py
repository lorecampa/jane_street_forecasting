
import time
import typing
from catboost import CatBoostRegressor, Pool, sum_models
from sklearn.metrics import r2_score
import torch
from tqdm import tqdm

from prj.config import DATA_DIR, EXP_DIR
from prj.data.data_loader import PARTITIONS_DATE_INFO, DataConfig, DataLoader
import lightgbm as lgb
import optuna
import argparse
from datetime import datetime
import os
import logging
import polars as pl
import numpy as np
import gc
from pathlib import Path

from prj.utils import LGBMEarlyStoppingCallbackWithTimeout

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
AUX_COLS = ['date_id', 'time_id', 'symbol_id', 'weight', 'responder_6', 'partition_id']

MAX_BOOST_ROUNDS = 1000
OLD_DATASET_MAX_HISTORY = 30
LOG_PERIOD=50
LR_LOWER_BOUND=5e-6
verbose=True


def weighted_r2_metric(y_pred, dataset):
    y_true = dataset.get_label()
    sample_weight = dataset.get_weight()
    
    r2 = r2_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    return 'weighted_r2', r2, True


def evaluate_model(model, X, y, w) -> float:
    y_pred = model.predict(X).clip(-5, 5).flatten()
    return r2_score(y_true=y, y_pred=y_pred, sample_weight=w)

def build_splits(df: pl.DataFrame, features: list):
    X = df.select(features).to_numpy()
    y = df['responder_6'].to_numpy().flatten()
    w = df['weight'].to_numpy().flatten()
    return X, y, w

METRIC_FN_DICT = {
    'weighted_r2': weighted_r2_metric
}

class CatboostTimeLimitCallback:
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.start_time = None

    def after_iteration(self, info):
        if self.start_time is None:
            self.start_time = time.time()

        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit:
            print(f"Stopping training after {elapsed_time:.2f} seconds (time limit reached). Iteration {info.iteration}")
            return False
        
        return True

def train(init_model: CatBoostRegressor, params: dict, train_df: pl.DataFrame, val_df: pl.DataFrame, use_weighted_loss, features, iterations, task_type = 'CPU'):
    start_time = time.time()
    _params = params.copy()
    _params.pop('iterations', None)
    _params.pop('task_type', None)
        
    X_train, y_train, w_train = build_splits(train_df, features)
    train_pool = Pool(data=X_train, label=y_train, weight=w_train if use_weighted_loss else None)
    del X_train, y_train, w_train
    gc.collect()
    
    is_early_stopping = val_df is not None and val_df.shape[0] > 0
    
    if is_early_stopping:
        X_val, y_val, w_val = build_splits(val_df, features)
        val_pool = Pool(data=X_val, label=y_val, weight=w_val if use_weighted_loss else None)
        del X_val, y_val, w_val
        gc.collect()

    
    print(f"Learning rate: {_params['learning_rate']:e}")
    model = CatBoostRegressor(
        iterations=iterations,
        task_type=task_type,
        **_params
    )
        
    model.fit(
        train_pool,
        init_model=init_model,
        eval_set=val_pool if is_early_stopping else None,
        early_stopping_rounds=50 if is_early_stopping else None,
    )
    print(f'Train completed in {((time.time() - start_time)/60):.3f} minutes')
    
    
    return model


def optimize_parameters(model_file_path: str, y_pred_offline, output_dir, online_old_dataset: pl.LazyFrame, online_learning_dataset: pl.LazyFrame, features: list, study_name, n_trials, storage):   
    def obj_function(trial):
        logging.info(f'Trial {trial.number}')
        
        model = CatBoostRegressor()
        model.load_model(model_file_path)
        params = model.get_params().copy()
        initial_model_lr = params['learning_rate']
        
        train_every = trial.suggest_int('train_every', 20, 50)
        old_data_fraction = trial.suggest_float('old_data_fraction', 0.01, 0.2, step=0.01)
        
        iterations = trial.suggest_int('iterations', 50, 700)
        
        use_weighted_loss = trial.suggest_categorical('use_weighted_loss', [True, False])
        lr_decay = trial.suggest_float('lr_decay', 0.5, 0.9)
        initial_online_lr = trial.suggest_float('initial_online_lr', 1e-5, initial_model_lr, log=True)
                
        params['learning_rate'] = initial_online_lr
        
        tmp_checkpoint_dir = os.path.join(output_dir, f'trial_{trial.number}')
        os.makedirs(tmp_checkpoint_dir)


        ctr_merge_policy = trial.suggest_categorical('ctr_merge_policy', ['LeaveMostDiversifiedTable', 'FailIfCtrIntersects', 'IntersectingCountersAverage', 'KeepAllTables'])
        weight_type = trial.suggest_categorical('weight_type', ['default', 'exp_decay', 'linear_decay'])
        
    
        
        y_hat = []
        y = []
        weights = []
        daily_r2 = []
            
        date_idx = 0
        date_ids = online_learning_dataset.select('date_id').collect().to_series().unique().sort().to_list()
        batch_size = 500
        
        models = [model]
        
        
        new_dataset: typing.Optional[pl.DataFrame] = None
        old_dataset: pl.DataFrame = online_old_dataset.collect()
        
        for i in range(0, len(date_ids), batch_size):
            batch_online_learning_dataset = online_learning_dataset.filter(pl.col('date_id').is_between(date_ids[i], date_ids[min(i+batch_size, len(date_ids))-1])).collect()
            n_batch_days = batch_online_learning_dataset['date_id'].n_unique()
            for date_id, test in tqdm(batch_online_learning_dataset.group_by('date_id', maintain_order=True), total=n_batch_days, desc=f'Batch {i}: Predicting online'):
                date_id = date_id[0]
            
                if (date_idx + 1) % train_every == 0:                
                    # max_date = new_dataset['date_id'].max()
                    # new_validation_dataset = new_dataset.filter(pl.col('date_id') > max_date - last_n_days_es)
                    # new_training_dataset = new_dataset.filter(pl.col('date_id') <= max_date - last_n_days_es)
                    
                    train_df = new_dataset
                    val_df = new_dataset.clear()
                    if verbose:
                        old_days = old_dataset['date_id'].unique().sort().to_list()
                        train_days = train_df['date_id'].unique().sort().to_list()
                        val_days = val_df['date_id'].unique().sort().to_list()
                        print('Old days: ', old_days)
                        print('Train days: ', train_days)
                        print('Val days: ', val_days)
                    
                    if old_data_fraction > 0:
                        new_training_dataset_len = train_df.shape[0]
                        old_dataset_len = old_dataset.shape[0]
                        old_data_len = min(int(old_data_fraction * new_training_dataset_len / (1 - old_data_fraction)), old_dataset_len)
                        
                        train_df = pl.concat([old_dataset.sample(n=old_data_len), train_df])
                    
                    logging.info(f'Starting fine tuning at date {date_id}')

        
                    params['learning_rate'] = max(params['learning_rate'] * lr_decay, LR_LOWER_BOUND)
                    
                    curr_model = train(
                        init_model= None, 
                        train_df=train_df,
                        val_df=val_df,
                        use_weighted_loss=use_weighted_loss,
                        params=params,
                        features=features,
                        iterations=iterations,
                        task_type='GPU'
                    )
                    models.append(curr_model)
                    
                    if weight_type == 'exp_decay':
                        decay_rate = 0.8
                        num_models = len(models)
                        models_weights = [decay_rate ** (num_models - i - 1) for i in range(num_models)]
                        models_weights = [w / sum(models_weights) for w in models_weights]
                    elif weight_type == 'linear_decay':
                        num_models = len(models)
                        models_weights = [(i + 1) for i in range(num_models)]
                        weights = [w / sum(models_weights) for w in models_weights]
                    elif weight_type == 'default':
                        models_weights = None # Default equal weights
                    print('Models Weights: ', models_weights)
                    
                    model = sum_models(models, weights=models_weights, ctr_merge_policy=ctr_merge_policy)
                                        
                    max_old_dataset_date = train_df['date_id'].max()
                    old_dataset = pl.concat([
                        old_dataset,
                        train_df
                    ]).filter(
                        pl.col('date_id').is_between(max_old_dataset_date-OLD_DATASET_MAX_HISTORY, max_old_dataset_date)
                    )
                    new_dataset = val_df


                date_idx += 1
                test_ = test.select(AUX_COLS + features)
                new_dataset = test_ if new_dataset is None else new_dataset.vstack(test_)
                
    
                
                preds = model.predict(test_.select(features).to_numpy()).flatten()
                y_hat.append(preds)
                y.append(test_.select(['responder_6']).to_numpy().flatten())
                weights.append(test_.select(['weight']).to_numpy().flatten())
                daily_score = r2_score(y_pred=preds, y_true=y[-1], sample_weight=weights[-1])
                daily_r2.append(daily_score)
            
                                            
        y_hat = np.concatenate(y_hat)
        y = np.concatenate(y)
        weights = np.concatenate(weights)
        daily_r2 = np.array(daily_r2)
        
        partitions = online_learning_dataset.select('partition_id').unique().collect().to_series().sort().to_list()
        partition_index_df = online_learning_dataset.select('partition_id', 'date_id').with_row_index('index').collect()
        
        partition_scores = []
        partition_sharpes = []
        last_date_idx = 0
        for partition_id in partitions:
            _partition_df = partition_index_df.filter(pl.col('partition_id') == partition_id)
            index_min = _partition_df['index'].min()
            index_max = _partition_df['index'].max() + 1
            partition_n_dates = _partition_df['date_id'].n_unique()
            
            y_hat_p = y_hat[index_min:index_max]
            y_p = y[index_min:index_max]
            w_p = weights[index_min:index_max]
            daily_r2_p = daily_r2[last_date_idx:last_date_idx+partition_n_dates]
            y_hat_offline_p = y_pred_offline[index_min:index_max]
            
            last_date_idx += partition_n_dates
            
            score = r2_score(y_true=y_p, y_pred=y_hat_p, sample_weight=w_p)
            score_offline = r2_score(y_true=y_p, y_pred=y_hat_offline_p, sample_weight=w_p)
            sharpe = np.mean(daily_r2_p) / np.std(daily_r2_p)            
            stability_index = np.sum(daily_r2_p > 0) / daily_r2_p.shape[0]
            
            partition_scores.append(score)
            partition_sharpes.append(sharpe)
            
            trial.set_user_attr(f"partition_{partition_id}_score", score)
            trial.set_user_attr(f"partition_{partition_id}_sharpe", sharpe)
            trial.set_user_attr(f"partition_{partition_id}_stability_index", stability_index)
            trial.set_user_attr(f"partition_{partition_id}_score_offline", score_offline)
            
            logging.info(f'Partition {partition_id} -> [offline_r2={score_offline:.4f}, r2={score:.4f}, sharpe_ratio={sharpe:.4f}, stability={stability_index:.4f}]')
        
        assert last_date_idx == len(daily_r2)
        
        total_score = r2_score(y_true=y, y_pred=y_hat, sample_weight=weights)
        total_sharpe = np.mean(daily_r2) / np.std(daily_r2)
        total_stability_index = np.sum(daily_r2 > 0) / daily_r2.shape[0]
        total_score_offline = r2_score(y_true=y, y_pred=y_pred_offline, sample_weight=weights)
        
        trial.set_user_attr('total_score', total_score)
        trial.set_user_attr('total_sharpe', total_sharpe)
        trial.set_user_attr('total_stability_index', total_stability_index)
        trial.set_user_attr('total_score_offline', total_score_offline)
                
        logging.info(f'Total -> [offline_r2={total_score_offline:.4f}, r2={total_score:.4f}, sharpe_ratio={total_sharpe:.4f}, stability={total_stability_index:.4f}, gain={(total_score - total_score_offline):.4f}]')
        
        gain_r2 = total_score - total_score_offline
        
        return gain_r2, total_sharpe
    
        
    study = optuna.create_study(directions=['maximize', 'maximize'], study_name=study_name, storage=storage, load_if_exists=True)    
    study.optimize(obj_function, n_trials=n_trials)
    return study.trials_dataframe()
        

def main(dataset_path, output_dir, study_name, n_trials, storage):
    data_args = {'include_time_id': True, 'include_intrastock_norm_temporal': True}
    config = DataConfig(**data_args)
    loader = DataLoader(data_dir=dataset_path, config=config)
    
    start_p, end_p = 8, 9
    complete_ds = loader.load(PARTITIONS_DATE_INFO[start_p]['min_date']-OLD_DATASET_MAX_HISTORY-1, PARTITIONS_DATE_INFO[end_p]['max_date'])
    
    split_point = PARTITIONS_DATE_INFO[start_p]['min_date']
    old_dataset = complete_ds.filter(
        pl.col('date_id').lt(split_point)
    )
    online_learning_dataset = complete_ds.filter(
        pl.col('date_id').ge(split_point)
    )
    
    # complete_ds = loader.load(1300, 1400)
    
    # old_dataset = complete_ds.filter(
    #     pl.col('date_id').lt(1330)
    # )
    # online_learning_dataset = complete_ds.filter(
    #     pl.col('date_id').ge(1330)
    # )
    

    features = loader.features
    print(f'Loaded features: {features}')
    old_dataset = old_dataset.select(AUX_COLS + features)
    online_learning_dataset = online_learning_dataset.select(AUX_COLS + features)
    
    model = CatBoostRegressor()
    model_file_path = DATA_DIR / 'models' / 'catboost' / 'catboost_4_7_w.cbm'
    model.load_model(model_file_path)
    print(model.get_params())
    
    # Offline scores
    X_online = online_learning_dataset.select(features).collect().to_numpy()
    
    y_pred_offline = []
    pred_batch_size = 1000000
    for i in tqdm(range(0, X_online.shape[0], pred_batch_size), desc='Predicting offline'):
        y_pred_offline.append(model.predict(X_online[i:i+pred_batch_size]).clip(-5, 5).flatten())
    
    y_pred_offline = np.concatenate(y_pred_offline)
    
    # y_pred_offline = np.zeros(X_online.shape[0])
    
    del X_online
    gc.collect()
        


    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    trials_df = optimize_parameters(model_file_path=model_file_path, y_pred_offline=y_pred_offline, output_dir=output_dir, online_old_dataset=old_dataset, online_learning_dataset=online_learning_dataset, features=features, study_name=study_name, n_trials=n_trials, 
                                    storage=storage)

        
    trials_file_path = os.path.join(output_dir, 'trials_dataframe.csv')
    logging.info(f'Saving the trials dataframe at: {trials_file_path}')
    trials_df.to_csv(trials_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script")
    parser.add_argument("-output_dir", default=None, type=str, required=False,
                        help="The directory where the models will be placed")
    
    parser.add_argument("-dataset_path", default=None, type=str, required=False,
                        help="Parquet file where the training dataset is placed")
    parser.add_argument("-n_trials", default=100, type=int, required=False,
                        help="Number of optuna trials to perform")
    parser.add_argument("-study_name", default=None, type=str, required=False,
                        help="Optional name of the study. Should be used if a storage is provided")
    parser.add_argument("-storage", default=None, type=str, required=False,
                        help="Optional storage url for saving the trials")

    
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir or EXP_DIR / 'catboost_online_sum_models'
    DATASET_DIR = args.dataset_path or DATA_DIR 
    N_TRIALS = args.n_trials
    STUDY_NAME = args.study_name
    STORAGE = args.storage
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'catboost_online_tuning_sum_models_{timestamp}' if STUDY_NAME is None else STUDY_NAME
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if STUDY_NAME is None:
        STUDY_NAME = model_name
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(Path(DATASET_DIR), output_dir, STUDY_NAME, N_TRIALS, STORAGE)
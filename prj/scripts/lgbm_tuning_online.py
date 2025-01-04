
import time
import typing
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
LR_LOWER_BOUND=1e-5
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


def train_with_es(init_model: lgb.Booster, params: dict, train_df: pl.DataFrame, val_df: pl.DataFrame, use_weighted_loss, metric, output_dir, es_patience, features):
    start_time = time.time()
    _params = params.copy()
    _params.pop('num_iterations', None)
    
    if metric is not None:
        _params['metric'] = metric
    
    X_train, y_train, w_train = build_splits(train_df, features)
    retrain_data = lgb.Dataset(data=X_train, label=y_train, weight=w_train if use_weighted_loss else None)
    
    del X_train, y_train, w_train
    gc.collect()
    
    X_val, y_val, w_val = build_splits(val_df, features)
    val_retrain_data = lgb.Dataset(data=X_val, label=y_val, weight=w_val if use_weighted_loss else None, reference=retrain_data)

    callbacks = [LGBMEarlyStoppingCallbackWithTimeout(es_patience, timeout_seconds=50), lgb.log_evaluation(period=LOG_PERIOD)]
    
        
    print(f"Learning rate: {_params['learning_rate']:e}")
    model = lgb.train(
        _params, 
        train_set=retrain_data, 
        num_boost_round=MAX_BOOST_ROUNDS, 
        init_model=init_model, 
        valid_sets=[val_retrain_data], 
        callbacks=callbacks,
        feval=METRIC_FN_DICT[metric] if metric in METRIC_FN_DICT else None
    )
    logging.info(f'Train completed in {((time.time() - start_time)/60):.3f} minutes')
     
    save_path = os.path.join(output_dir, 'best_model.txt')
    model.save_model(save_path)
    
    return model


def optimize_parameters(model_file_path: str, y_pred_offline, output_dir, online_old_dataset: pl.LazyFrame, online_learning_dataset: pl.LazyFrame, features: list, study_name, n_trials, storage):   
    def obj_function(trial):
        logging.info(f'Trial {trial.number}')
        
        train_every = trial.suggest_int('train_every', 20, 50)
        last_n_days_es = trial.suggest_int('last_n_days_es', 5, 14)
        old_data_fraction = trial.suggest_float('old_data_fraction', 0.01, 0.5, step=0.01)
        
        
        use_weighted_loss = trial.suggest_categorical('use_weighted_loss', [True, False])
        metric = None
        es_patience=20
        decay_lr_once = trial.suggest_categorical('decay_lr_once', [True, False])
        lr_decay = trial.suggest_float('lr_decay', 0.1, 0.9)

        
        tmp_checkpoint_dir = os.path.join(output_dir, f'trial_{trial.number}')
        os.makedirs(tmp_checkpoint_dir)

        model = lgb.Booster(model_file=model_file_path)
        params = model.params.copy()
        initial_lr = params['learning_rate']
    
        
        y_hat = []
        y = []
        weights = []
        daily_r2 = []
            
        date_idx = 0
        date_ids = online_learning_dataset.select('date_id').collect().to_series().unique().sort().to_list()
        batch_size = 500
        
        
        new_dataset: typing.Optional[pl.DataFrame] = None
        old_dataset: pl.DataFrame = online_old_dataset.collect()
        
        for i in range(0, len(date_ids), batch_size):
            batch_online_learning_dataset = online_learning_dataset.filter(pl.col('date_id').is_between(date_ids[i], date_ids[min(i+batch_size, len(date_ids))-1])).collect()
            n_batch_days = batch_online_learning_dataset['date_id'].n_unique()
            for date_id, test in tqdm(batch_online_learning_dataset.group_by('date_id', maintain_order=True), total=n_batch_days, desc=f'Batch {i}: Predicting online'):
                date_id = date_id[0]
            
                if (date_idx + 1) % train_every == 0:                
                    max_date = new_dataset['date_id'].max()
                    new_validation_dataset = new_dataset.filter(pl.col('date_id') > max_date - last_n_days_es)
                    new_training_dataset = new_dataset.filter(pl.col('date_id') <= max_date - last_n_days_es)
                    
                    
                    if verbose:
                        old_days = old_dataset['date_id'].unique().sort().to_list()
                        train_days = new_training_dataset['date_id'].unique().sort().to_list()
                        val_days = new_validation_dataset['date_id'].unique().sort().to_list()
                        print('Old days: ', old_days)
                        print('Train days: ', train_days)
                        print('Val days: ', val_days)
                    
                    new_training_dataset_len = new_training_dataset.shape[0]
                    old_dataset_len = old_dataset.shape[0]
                    old_data_len = min(int(old_data_fraction * new_training_dataset_len / (1 - old_data_fraction)), old_dataset_len)
                    
                    train_df = pl.concat([old_dataset.sample(n=old_data_len), new_training_dataset])
                    val_df = new_validation_dataset
                    
                    logging.info(f'Starting fine tuning at date {date_id}')

                    if decay_lr_once:
                        params['learning_rate'] = max(initial_lr * lr_decay, LR_LOWER_BOUND)
                    else:
                        params['learning_rate'] = max(params['learning_rate'] * lr_decay, LR_LOWER_BOUND)
                    
                    model = train_with_es(
                        init_model= model, 
                        train_df=train_df,
                        val_df=val_df,
                        use_weighted_loss=use_weighted_loss,
                        metric=metric,
                        es_patience=es_patience,
                        params=params,
                        output_dir=tmp_checkpoint_dir,
                        features=features
                    )
                                        
                    max_old_dataset_date = new_training_dataset['date_id'].max()
                    old_dataset = pl.concat([
                        old_dataset,
                        new_training_dataset
                    ]).filter(
                        pl.col('date_id').is_between(max_old_dataset_date-OLD_DATASET_MAX_HISTORY, max_old_dataset_date)
                    )
                    new_dataset = new_validation_dataset


                date_idx += 1
                test_ = test.select(AUX_COLS + features)
                new_dataset = test_ if new_dataset is None else new_dataset.vstack(test_)
                
    
                
                preds = model.predict(test_.select(features).to_numpy()).flatten()
                y_hat.append(preds)
                y.append(test_.select(['responder_6']).to_numpy().flatten())
                weights.append(test_.select(['weight']).to_numpy().flatten())
                daily_r2.append(r2_score(y_pred=preds, y_true=y[-1], sample_weight=weights[-1]))
            
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
        
        return total_score, total_sharpe
    
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
    
    model_file_path = "/home/lorecampa/projects/jane_street_forecasting/experiments/lgbm_offline/lgbm_offline_2025-01-04_16-04-39/trial_0/best_model.txt"
    model = lgb.Booster(model_file=model_file_path)
    
    # Offline scores
    X_online = online_learning_dataset.select(features).collect().to_numpy()
    y_pred_offline = []
    pred_batch_size = 1000000
    for i in tqdm(range(0, X_online.shape[0], pred_batch_size), desc='Predicting offline'):
        y_pred_offline.append(model.predict(X_online[i:i+pred_batch_size]).clip(-5, 5).flatten())
    
    y_pred_offline = np.concatenate(y_pred_offline)
    
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
    OUTPUT_DIR = args.output_dir or EXP_DIR / 'lgbm_online'
    DATASET_DIR = args.dataset_path or DATA_DIR 
    N_TRIALS = args.n_trials
    STUDY_NAME = args.study_name
    STORAGE = args.storage
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'lgbm_online_tuning_{timestamp}' if STUDY_NAME is None else STUDY_NAME
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    if STUDY_NAME is None:
        STUDY_NAME = model_name
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(Path(DATASET_DIR), output_dir, STUDY_NAME, N_TRIALS, STORAGE)
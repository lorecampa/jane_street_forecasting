
import time
import typing
from line_profiler import profile
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
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

from prj.utils import LGBMEarlyStoppingCallbackWithTimeout, timeout

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
AUX_COLS = ['date_id', 'time_id', 'symbol_id', 'weight', 'responder_6', 'partition_id']

MAX_BOOST_ROUNDS = 500
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
    _params = params.copy()
    _params.pop('num_iterations', None)
    
    early_stopping = val_df.shape[0] > 0
    
    if metric is not None:
        _params['metric'] = metric
    
    X_train, y_train, w_train = build_splits(train_df, features)
    retrain_data = lgb.Dataset(data=X_train, label=y_train, weight=w_train if use_weighted_loss else None)
    
    del X_train, y_train, w_train
    
    
    callbacks = []
    if early_stopping:
        X_val, y_val, w_val = build_splits(val_df, features)
        val_retrain_data = lgb.Dataset(data=X_val, label=y_val, weight=w_val if use_weighted_loss else None, reference=retrain_data)
        callbacks += [LGBMEarlyStoppingCallbackWithTimeout(es_patience, timeout_seconds=50), lgb.log_evaluation(period=LOG_PERIOD)]
        del X_val, y_val, w_val
    
        
    print(f"Learning rate: {_params['learning_rate']:e}")
    model = lgb.train(
        _params, 
        train_set=retrain_data, 
        num_boost_round=MAX_BOOST_ROUNDS if early_stopping else 200, 
        init_model=init_model, 
        valid_sets=[val_retrain_data] if early_stopping else None, 
        callbacks=callbacks,
        keep_training_booster=True,
        feval=METRIC_FN_DICT[metric] if metric in METRIC_FN_DICT else None
    )
     
    save_path = os.path.join(output_dir, 'best_model.txt')
    model.save_model(save_path)
    
    return model

def refit(model: lgb.Booster, train_df: pl.DataFrame, use_weighted_loss, features, decay_rate=0.9):
    start_time = time.time()
    X_train, y_train, w_train = build_splits(train_df, features)
    
    model = model.refit(
        data=X_train,
        label=y_train,
        weight=w_train if use_weighted_loss else None,
        decay_rate=decay_rate,
    )
    
    print(f'Refit completed in {((time.time() - start_time)/60):.3f} minutes')
    
    return model


def optimize_parameters(model_file_path: str, y_pred_offline, output_dir, online_old_dataset: pl.LazyFrame, online_learning_dataset: pl.LazyFrame, features: list, study_name, n_trials, storage):   
    
    def obj_function(trial: optuna.Trial):
        logging.info(f'Trial {trial.number}')
        model = lgb.Booster(model_file=model_file_path)
        params = model.params.copy()
        # initial_model_lr = params['learning_rate']

        
        refit_every = trial.suggest_int('train_every', 5, 40)
        refit_decay_rate = trial.suggest_float('refit_decay_rate', 0.8, 0.95, step=0.01)
        retrain_every_n_refit = trial.suggest_int('retrain_every_n_refit', 5, 10, step=1)
        retrain_every = int(retrain_every_n_refit * refit_every)
        
        old_data_fraction = trial.suggest_float('old_data_fraction', 0., 0.7, step=0.05)
        # train_val_fraction = trial.suggest_float('train_val_fraction', 0.2, 0.8, step=0.05)
        # train_val_fraction = 0.2
        use_weighted_loss = trial.suggest_categorical('use_weighted_loss', [True, False])
        metric = None
        es_patience=20
        lr_decay = 0.8
        # lr_decay = trial.suggest_float('lr_decay', 0.1, 0.99, step=0.05)
        # initial_lr = trial.suggest_float('initial_learning_rate', LR_LOWER_BOUND, initial_model_lr, log=True)
        
        # params['learning_rate'] = initial_model_lr
        
        # es_mode = trial.suggest_categorical('es_mode', ['holdout_first', 'holdout_last', 'random_days', 'random_samples', 'None'])
        es_mode = 'holdout_first'
        es_ratio = 0.1
        # if es_mode != 'None':
        #     es_ratio = trial.suggest_float('es_ratio', 0.1, 0.4, step=0.05)
        
        tmp_checkpoint_dir = os.path.join(output_dir, f'trial_{trial.number}')
        os.makedirs(tmp_checkpoint_dir)

        y_hat = []
        y = []
        weights = []
        daily_r2 = []
            
        date_idx = 0
        online_dataset = online_learning_dataset.clone()
        date_ids = online_dataset.select('date_id').collect().to_series().unique().sort().to_list()
        batch_size = 500
        
        
        new_dataset: typing.Optional[pl.DataFrame] = None
        old_dataset: pl.DataFrame = online_old_dataset.collect()
                
        n_timeout_reached = 0        
        
        step = 0
        for i in range(0, len(date_ids), batch_size):
            batch_online_learning_dataset = online_dataset.filter(pl.col('date_id').is_between(date_ids[i], date_ids[min(i+batch_size, len(date_ids))-1])).collect()
            n_batch_days = batch_online_learning_dataset['date_id'].n_unique()
            for date_id, test in tqdm(batch_online_learning_dataset.group_by('date_id', maintain_order=True), total=n_batch_days, desc=f'Batch {i}: Predicting online'):
                start_pred_time = time.time()
                
                date_id = date_id[0]
                should_retrain = (date_idx + 1) % retrain_every == 0
                should_refit = (date_idx + 1) % refit_every == 0
                if should_retrain or should_refit:
                    train_val_df = new_dataset
                    # train_val_df = new_dataset.group_by('date_id', 'symbol_id').agg(
                    #     pl.all().sample(fraction=train_val_fraction)
                    # )
                    # train_val_df = train_val_df\
                    #     .explode(train_val_df.columns[2:])\
                    #     .sort('date_id', 'time_id', 'symbol_id')\
                    #     .select(new_dataset.columns)
                    
                    
                    train_val_days = train_val_df['date_id'].unique().sort().to_numpy()     
                    len_train_val_days = len(train_val_days) 
                    
                    if es_mode in ['random_days', 'holdout_first', 'holdout_last']:
                        if es_mode == 'random_days':
                            train_days, val_days = train_test_split(train_val_days, test_size=es_ratio)
                        elif es_mode == 'holdout_first':
                            split_point = max(int(len_train_val_days * es_ratio), 1)
                            val_days = train_val_days[:split_point]
                            train_days = train_val_days[split_point:]
                        elif es_mode == 'holdout_last':
                            split_point = max(int(len_train_val_days * es_ratio), 1)
                            val_days = train_val_days[-split_point:]
                            train_days = train_val_days[:-split_point]
                            
                        val_df = train_val_df.filter(pl.col('date_id').is_in(val_days))
                        train_df = train_val_df.filter(pl.col('date_id').is_in(train_days))
                    elif es_mode == 'random_samples':
                        np.random.seed()
                        shuffled_indices = np.random.permutation(len(train_val_df))
                        split_index = int(len(train_val_df) * (1 - es_ratio))
                        train_indices = shuffled_indices[:split_index]
                        val_indices = shuffled_indices[split_index:]
                        
                        val_df = train_val_df[val_indices]
                        train_df = train_val_df[train_indices]
                        
                    elif es_mode == 'None':
                        train_df = train_val_df
                        val_df = train_val_df.clear()
                    
                    else:
                        raise ValueError(f'Unknown es_mode: {es_mode}')
                    
                    
                    if verbose:
                        old_days = old_dataset['date_id'].unique().sort().to_list()
                        train_days = train_df['date_id'].unique().sort().to_list()
                        val_days = val_df['date_id'].unique().sort().to_list()
                        print('Old days: ', old_days)
                        print('Train days: ', train_days)
                        print('Val days: ', val_days)
                        print(train_df.shape, val_df.shape)
                    
                    if old_data_fraction > 0:
                        unique_train_val_symbols = train_val_df['symbol_id'].unique().to_list()
            
                        train_df_len = train_df.shape[0]
                        old_dataset_len = old_dataset.shape[0]
                        old_data_len = min(int(old_data_fraction * train_df_len / (1 - old_data_fraction)), old_dataset_len)
                        if verbose:
                            print(f"Adding {old_data_len} old data samples to training set, {old_data_fraction * 100:.2f}% of the current training set")
                        
                        train_df = old_dataset.filter(pl.col('symbol_id').is_in(unique_train_val_symbols))\
                            .sample(n=old_data_len)\
                            .vstack(train_df)
                        
                    
                    if should_retrain:
                        logging.info(f'Starting RETRAIN at date {date_id}')

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
                        logging.info(f'RETRAIN completed in {(time.time() - start_pred_time)/60:.4f} minutes')
                    else:
                        logging.info(f'Starting REFIT at date {date_id}')
                        with timeout(52):
                            try:
                                model = refit(model, train_val_df, use_weighted_loss, features, decay_rate=refit_decay_rate)
                            except TimeoutError as e:
                                print('Timeout reached during refit')
                                n_timeout_reached += 1
                                
                        logging.info(f'REFIT completed in {(time.time() - start_pred_time)/60:.4f} minutes')

                                     
                    max_old_dataset_date = new_dataset['date_id'].max()
                    old_dataset = pl.concat([
                        old_dataset,
                        new_dataset
                    ]).filter(
                        pl.col('date_id').is_between(max_old_dataset_date-OLD_DATASET_MAX_HISTORY, max_old_dataset_date)
                    )
                    new_dataset = None
                    
                    
                date_idx += 1
                test_ = test.select(AUX_COLS + features)
                new_dataset = test_ if new_dataset is None else new_dataset.vstack(test_)
                
    
                
                preds = model.predict(test_.select(features).to_numpy()).flatten()
                y_hat.append(preds)
                y.append(test_.select(['responder_6']).to_numpy().flatten())
                weights.append(test_.select(['weight']).to_numpy().flatten())
                daily_score = r2_score(y_pred=preds, y_true=y[-1], sample_weight=weights[-1])
                daily_r2.append(daily_score)
                
                step += 1
                
                # total_pred_time = time.time() - start_pred_time
                # logging.info(f'Predict completed in {total_pred_time/60:.4f} minutes')
                # if total_pred_time > 60:
                #     n_timeout_reached += 1
                #     logging.info(f'Timeout reached {n_timeout_reached}/{n_online_retrain} times')
                #     if n_timeout_reached > int(n_online_retrain * 0.3):
                #         logging.info(f'Timeout reached {n_timeout_reached} times. Pruning trial...')
                #         raise optuna.TrialPruned()
        
        logging.info(f'Online learning completed, reached {n_timeout_reached} timeouts')
        trial.set_user_attr('n_timeout_reached', n_timeout_reached)
         
        y_hat = np.concatenate(y_hat)
        y = np.concatenate(y)
        weights = np.concatenate(weights)
        daily_r2 = np.array(daily_r2)
        
        partitions = online_dataset.select('partition_id').unique().collect().to_series().sort().to_list()
        partition_index_df = online_dataset.select('partition_id', 'date_id').with_row_index('index').collect()
        
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
        
        model.save_model(os.path.join(tmp_checkpoint_dir, 'model.txt'))
        
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
    
    model_study_name = "lgbm_offline_2025-01-04_19-42-20"
    model_storage = "mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet"
    study = optuna.load_study(study_name=model_study_name, storage=model_storage)
    best_trial = max(study.best_trials, key=lambda t: t.values[0]) # highest r2

    print(f'Loaded study {model_study_name} from {model_storage} with {len(study.trials)} trials. Best trial number: {best_trial.number}')


    model_file_path = f'/home/lorecampa/projects/jane_street_forecasting/dataset/models/lgbm_maxbin_63_0_7_w.txt'
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
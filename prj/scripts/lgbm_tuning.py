
import time
from sklearn.metrics import r2_score
import torch
from tqdm import tqdm

from prj.config import DATA_DIR, EXP_DIR
from prj.data.data_loader import PARTITIONS_DATE_INFO, DataConfig, DataLoader
from prj.hyperparameters_opt import sample_lgbm_params
from prj.model.torch.metrics import weighted_r2_score
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

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"
AUX_COLS = ['date_id', 'time_id', 'symbol_id', 'weight', 'responder_6', 'partition_id']

MAX_BOOST_ROUNDS = 1000
LOG_PERIOD=50
verbose=False
EARLY_STOPPING = False


def weighted_r2_metric(y_pred, dataset):
    y_true = dataset.get_label()
    sample_weight = dataset.get_weight()
    
    r2 = r2_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    return 'weighted_r2', r2, True


def evaluate_model(model, X, y, w, include_predictions = False):
    y_pred = model.predict(X).clip(-5, 5).flatten()
    score = r2_score(y_true=y, y_pred=y_pred, sample_weight=w)
    if include_predictions:
        return score, y_pred
    return score

def build_splits(df: pl.LazyFrame, features: list, add_dates=False):
    X = df.select(features).collect().to_numpy()
    y = df.select(['responder_6']).collect().to_numpy().flatten()
    w = df.select(['weight']).collect().to_numpy().flatten()
    if add_dates:
        dates = df.select(['date_id']).collect().to_numpy().flatten()
        return X, y, w, dates
    
    return X, y, w

METRIC_FN_DICT = {
    'weighted_r2': weighted_r2_metric
}



def _sample_lgbm_params(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    use_gpu = additional_args.get("use_gpu", False)
    params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "num_leaves": trial.suggest_int("num_leaves", 4, 256),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.1, 0.7),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.8),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1000, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1000, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-6, 1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-7, 1e-1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 10000, log=True),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            "device": "gpu" if use_gpu else "cpu",
            "verbose": -1
        }
    
    if EARLY_STOPPING:
        params['n_estimators'] = MAX_BOOST_ROUNDS
    else:
        params['n_estimators'] = trial.suggest_int("num_boost_round", 100, 800, log=True)
    
    if use_gpu:
        params['max_bin'] = 63
        params['gpu_use_dp'] = False
        
    else:
        max_bin = additional_args.get('max_bin', None)
        if max_bin is not None:
            params['max_bin'] = max_bin
        else:
            params['max_bin'] = trial.suggest_int('max_bin', 8, 256, log=True)
    
    return params

def train(params: dict, train_dl: pl.LazyFrame, val_dl: pl.LazyFrame, use_weighted_loss, metric, output_dir, es_patience, features):
    start_time = time.time()
    _params = params.copy()
    if metric is not None:
        _params['metric'] = metric
    
    X_train, y_train, w_train = build_splits(train_dl, features)
    train_data = lgb.Dataset(data=X_train, label=y_train, weight=w_train if use_weighted_loss else None)
    del X_train, y_train, w_train
    gc.collect()
    
    # print('Loading binary file')
    # binary_name = 'lgbm_maxbin_63_0_7'
    # if use_weighted_loss:
    #     binary_name += '_w'
    
    # binary_path = f"/home/lorecampa/projects/jane_street_forecasting/dataset/binary/{binary_name}.bin"
    # train_data =lgb.Dataset(data=binary_path, params={
    #     'feature_pre_filter': False,
    #     'device': 'cpu'
    # })
    # logging.info('Binary file loaded')

    
    callbacks = []
    if EARLY_STOPPING:
        X_val, y_val, w_val = build_splits(val_dl, features)
        callbacks += [lgb.early_stopping(stopping_rounds=es_patience), lgb.log_evaluation(period=LOG_PERIOD)]
    else:
        # Dummy eval set to log the training progress, cannot find other way to log the training status
        n_dummy_samples = 100        
        X_val = np.random.rand(n_dummy_samples, len(features))
        y_val = np.random.rand(n_dummy_samples)
        w_val = np.random.rand(n_dummy_samples)
        
        callbacks += [lgb.log_evaluation(period=LOG_PERIOD)]
    
    val_data = lgb.Dataset(data=X_val, label=y_val, weight=w_val if use_weighted_loss else None, reference=train_data)
    del X_val, y_val, w_val
    gc.collect()
        
    print(f"Learning rate: {_params['learning_rate']:.4f}")
    num_boost_round = _params.pop('n_estimators')
    
    model = lgb.train(
        _params, 
        train_data, 
        num_boost_round=num_boost_round, 
        valid_sets=[val_data],
        valid_names=['val' if EARLY_STOPPING else 'dummy_val'], 
        callbacks=callbacks,
        feval=METRIC_FN_DICT[metric] if metric in METRIC_FN_DICT else None
    )
    logging.info(f'Train completed in {((time.time() - start_time)/60):.3f} minutes')
    
    score = None
    if EARLY_STOPPING:
        score = evaluate_model(model, X_val, y_val, w_val)
        
    save_path = os.path.join(output_dir, 'best_model.txt')
    model.save_model(save_path)
    
    return model, score


def optimize_parameters(output_dir, pretrain_dataset: pl.LazyFrame, pretrain_es_dataset: pl.LazyFrame, evaluation_dataset: pl.LazyFrame, features: list, study_name, n_trials, storage):   
    def obj_function(trial):
        
        logging.info(f'Trial {trial.number}')
        
        use_weighted_loss = trial.suggest_categorical('use_weighted_loss', [True, False])
        metric = None
        
        use_gpu = torch.cuda.is_available()
        
        additional_args = {'use_gpu': use_gpu, 'max_bin': 63}
        params = _sample_lgbm_params(trial, additional_args=additional_args)
        
        tmp_checkpoint_dir = os.path.join(output_dir, f'trial_{trial.number}')
        os.makedirs(tmp_checkpoint_dir)
        
        logging.info('Starting pretrain')
        
        if verbose:
            train_days = pretrain_dataset.select('date_id').collect().to_series().unique().sort().to_list()
            if EARLY_STOPPING:
                val_days = pretrain_es_dataset.select('date_id').collect().to_series().unique().sort().to_list()
                print('Train days: ', train_days)
                print('Val days: ', val_days)
            else:
                print('Train days: ', train_days)
        
         
        model, initial_wr2_score = train(
            train_dl=pretrain_dataset,
            val_dl=pretrain_es_dataset if EARLY_STOPPING else evaluation_dataset,
            use_weighted_loss=use_weighted_loss,
            metric=metric,
            es_patience=50,
            params=params,
            output_dir=tmp_checkpoint_dir,
            features=features,
        )
        gc.collect()
        
        if initial_wr2_score is not None:
            logging.info(f'Initial wr2 score: {initial_wr2_score}')
            trial.set_user_attr("initial_wr2_score", initial_wr2_score)
        
    
        partitions = evaluation_dataset.select('partition_id').collect().to_series().unique().sort().to_list()
        partition_scores = []
        partition_sharpes = []
        for partition_id in partitions:
            partition_df = evaluation_dataset.filter(pl.col('partition_id') == partition_id)
            X_test_p, y_test_p, w_test_p, dates_test_p = build_splits(partition_df, features, add_dates=True)
            partition_score, y_hat_test_p = evaluate_model(model, X_test_p, y_test_p, w_test_p, include_predictions=True)
            partition_scores.append(partition_score)
            
            daily_r2 = []
            unique_days = np.unique(dates_test_p)
            
            for day in tqdm(unique_days, total=len(unique_days), desc=f'Partition {partition_id}'):
                indices = np.where(dates_test_p == day)[0]
                y_hat_test_p_day = y_hat_test_p[indices]
                y_test_p_day = y_test_p[indices]
                w_test_p_day = w_test_p[indices]
                daily_r2.append(r2_score(y_true=y_test_p_day, y_pred=y_hat_test_p_day, sample_weight=w_test_p_day))
                
            daily_r2 = np.array(daily_r2)
            sharpe = np.mean(daily_r2) / np.std(daily_r2)
            partition_sharpes.append(sharpe)

            trial.set_user_attr(f"partition_{partition_id}_score", partition_score)
            trial.set_user_attr(f"partition_{partition_id}_sharpe", sharpe)
            logging.info(f'Partition {partition_id} score: {partition_score} sharpe: {sharpe}')
            
            
            del X_test_p, y_test_p, w_test_p
            gc.collect()
        
        return np.mean(partition_scores), np.mean(partition_sharpes)
            
    
    study = optuna.create_study(directions=['maximize', 'maximize'], study_name=study_name, storage=storage, load_if_exists=True)    
    study.optimize(obj_function, n_trials=n_trials)
    return study.trials_dataframe()
        

def main(dataset_path, output_dir, study_name, n_trials, storage):
    data_args = {'include_time_id': True, 'include_intrastock_norm_temporal': True}
    config = DataConfig(**data_args)
    loader = DataLoader(data_dir=dataset_path, config=config)
    
    pretraining_dataset = loader.load_with_partition(0, 7)
    # pretraining_dataset = loader.load(1100, 1150)
    features = loader.features
    
    pretraining_dataset = pretraining_dataset.select(AUX_COLS + features)
    
    
    pretrain_es_dataset = None
    if EARLY_STOPPING:
        max_date = pretraining_dataset.select(pl.col('date_id').max()).collect().item()
        pretrain_es_dataset = pretraining_dataset.filter(pl.col('date_id') > max_date - 14)
        pretraining_dataset = pretraining_dataset.filter(pl.col('date_id') <= max_date - 14)

    evaluation_dataset = loader.load_with_partition(8, 9)
    # evaluation_dataset = loader.load(1151, 1200)
    
    # features = loader.features
    print(f'Loaded features: {features}')
    
    evaluation_dataset = evaluation_dataset.select(AUX_COLS + features)

    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    trials_df = optimize_parameters(output_dir, pretraining_dataset, pretrain_es_dataset, evaluation_dataset, features, study_name, n_trials, 
                                    storage)
    
        
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
    OUTPUT_DIR = args.output_dir or EXP_DIR / 'lgbm_offline'
    DATASET_DIR = args.dataset_path or DATA_DIR 
    N_TRIALS = args.n_trials
    STUDY_NAME = args.study_name
    STORAGE = args.storage
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'lgbm_offline_{timestamp}' if STUDY_NAME is None else STUDY_NAME
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
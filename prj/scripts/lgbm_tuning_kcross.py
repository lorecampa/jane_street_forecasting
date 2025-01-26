
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

from prj.utils import BlockingTimeSeriesSplit, CombinatorialPurgedKFold

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

def train(params: dict, train_dl: pl.LazyFrame, test_dl: pl.LazyFrame, use_weighted_loss, metric, features):
    start_time = time.time()
    _params = params.copy()
    if metric is not None:
        _params['metric'] = metric
    
    X_train, y_train, w_train = build_splits(train_dl, features)
    train_data = lgb.Dataset(data=X_train, label=y_train, weight=w_train if use_weighted_loss else None)
    del X_train, y_train, w_train
    gc.collect()
    
    
    callbacks = []

    # Dummy eval set to log the training progress, cannot find other way to log the training status
    n_dummy_samples = 100        
    X_val = np.random.rand(n_dummy_samples, len(features))
    y_val = np.random.rand(n_dummy_samples)
    w_val = np.random.rand(n_dummy_samples)
    val_data = lgb.Dataset(data=X_val, label=y_val, weight=w_val if use_weighted_loss else None, reference=train_data)

    callbacks += [lgb.log_evaluation(period=LOG_PERIOD)]
    
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
    
    X_test, y_test, w_test = build_splits(test_dl, features)
    score = evaluate_model(model, X_test, y_test, w_test)
    
    return model, score


def optimize_parameters(output_dir, train_dataset: pl.LazyFrame, features: list, study_name, n_trials, storage):
    train_dataset = train_dataset.sort('date_id', 'time_id', 'symbol_id')
    train_days = train_dataset.select('date_id').unique().collect().to_series().sort().to_numpy()
    def obj_function(trial):
        
        logging.info(f'Trial {trial.number}')
        
        use_weighted_loss = trial.suggest_categorical('use_weighted_loss', [True, False])
        metric = None
        
        use_gpu = torch.cuda.is_available()
        
        additional_args = {'use_gpu': use_gpu, 'max_bin': 63}
        params = _sample_lgbm_params(trial, additional_args=additional_args)
        
        tmp_checkpoint_dir = os.path.join(output_dir, f'trial_{trial.number}')
        os.makedirs(tmp_checkpoint_dir)
        
        
        
        scores = []
        kf = BlockingTimeSeriesSplit(n_splits=4, val_ratio=0.2)
        for i, (train_index, test_index) in enumerate(kf.split(len(train_days))):  
            model, score = train(
                train_dl=train_dataset.filter(pl.col('date_id').is_in(train_days[train_index])),
                test_dl=train_dataset.filter(pl.col('date_id').is_in(train_days[test_index])),
                use_weighted_loss=use_weighted_loss,
                metric=metric,
                params=params,
                features=features,
            )
            model.save_model(os.path.join(tmp_checkpoint_dir, f'model_{i}.txt'))
            scores.append(score)
            
        return np.mean(scores)
            
    
    study = optuna.create_study(directions=['maximize'], study_name=study_name, storage=storage, load_if_exists=True)    
    study.optimize(obj_function, n_trials=n_trials)
    return study.trials_dataframe()
        

def main(dataset_path, output_dir, study_name, n_trials, storage):
    data_args = {'include_time_id': True, 'include_intrastock_norm_temporal': True}
    config = DataConfig(**data_args)
    loader = DataLoader(data_dir=dataset_path, config=config)
    
    train_dataset = loader.load_with_partition(4, 9)
    # train_dataset = loader.load(1100, 1150)
    features = loader.features
    
    train_dataset = train_dataset.select(AUX_COLS + features)
    
    print(f'Loaded features: {features}')
    
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    trials_df = optimize_parameters(output_dir, train_dataset, features, study_name, n_trials, storage)
    
        
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

import time
from sklearn.metrics import r2_score
import torch
from tqdm import tqdm

from prj.config import DATA_DIR, EXP_DIR
from prj.data.data_loader import PARTITIONS_DATE_INFO, DataConfig, DataLoader
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
AUX_COLS = ['date_id', 'time_id', 'symbol_id', 'weight', 'responder_6']

MAX_BOOST_ROUNDS = 1000
OLD_DATASET_MAX_HISTORY = 30
LOG_PERIOD=50
verbose=True


def weighted_r2_metric(y_pred, dataset):
    y_true = dataset.get_label()
    sample_weight = dataset.get_weight()
    
    r2 = r2_score(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    return 'weighted_r2', r2, True


def evaluate_model(model, X, y, w) -> float:
    y_pred = model.predict(X).clip(-5, 5).flatten()
    return r2_score(y_true=y, y_pred=y_pred, sample_weight=w)

def build_splits(df: pl.LazyFrame, features: list):
    X = df.select(features).collect().to_numpy()
    y = df.select(['responder_6']).collect().to_numpy().flatten()
    w = df.select(['weight']).collect().to_numpy().flatten()
    return X, y, w

METRIC_FN_DICT = {
    'weighted_r2': weighted_r2_metric
}

def sample_lgbm_params_online(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    use_gpu = additional_args.get("use_gpu", False)
    params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "num_leaves": trial.suggest_int("num_leaves", 4, 512),
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
    
    if use_gpu:
        params['max_bin'] = 63
        params['gpu_use_dp'] = False
        
    else:
        max_bin = additional_args.get('max_bin', None)
        if max_bin is not None:
            params['max_bin'] = max_bin
        else:
            params['max_bin'] = trial.suggest_int('max_bin', 8, 512, log=True)
    
    return params

def train_with_es(init_model: lgb.Booster, params: dict, train_dl: pl.LazyFrame, val_dl: pl.LazyFrame, use_weighted_loss, metric, output_dir, es_patience):
    start_time = time.time()
    _params = params.copy()
    if metric is not None:
        _params['metric'] = metric
    
    X_train, y_train, w_train = train_dl
    X_val, y_val, w_val = val_dl

    retrain_data = lgb.Dataset(data=X_train, label=y_train, weight=w_train if use_weighted_loss else None)
    val_retrain_data = lgb.Dataset(data=X_val, label=y_val, weight=w_val if use_weighted_loss else None)
    callbacks = [lgb.early_stopping(stopping_rounds=es_patience)]
    if verbose:
        callbacks += [lgb.log_evaluation(period=LOG_PERIOD)]
        
    print(f"Learning rate: {_params['learning_rate']:.4f}")
    model = lgb.train(
        _params, 
        retrain_data, 
        num_boost_round=MAX_BOOST_ROUNDS, 
        init_model=init_model, 
        valid_sets=[val_retrain_data], 
        callbacks=callbacks,
        feval=METRIC_FN_DICT[metric] if metric in METRIC_FN_DICT else None
    )
    logging.info(f'Train completed in {((time.time() - start_time)/60):.3f} minutes')
     
    score = evaluate_model(model, X_val, y_val, w_val)
    save_path = os.path.join(output_dir, 'best_model.txt')
    model.save_model(save_path)
    
    return model, score


def optimize_parameters(output_dir, pretrain_dataset: pl.LazyFrame, pretrain_val_dataset: pl.LazyFrame, online_learning_dataset: pl.LazyFrame, features: list, study_name, n_trials, storage, start_eval_from):   
    def obj_function(trial):
        
        logging.info(f'Trial {trial.number}')
        
        train_every = trial.suggest_int('train_every', 20, 50)
        last_n_days_es = trial.suggest_int('last_n_days_es', 5, 14)
        old_data_fraction = trial.suggest_float('old_data_fraction', 0.01, 0.5, step=0.01)
        
        
        use_weighted_loss = trial.suggest_categorical('use_weighted_loss', [True, False])
        lr_decay = trial.suggest_float('lr_decay', 0.1, 0.9)
        metric = None
        es_patience=20
        
        params = sample_lgbm_params_online(trial, additional_args={'use_gpu': torch.cuda.is_available()})

        
        tmp_checkpoint_dir = os.path.join(output_dir, f'trial_{trial.number}')
        os.makedirs(tmp_checkpoint_dir)
        
        logging.info('Starting pretrain')
        train_days = pretrain_dataset.select('date_id').collect().to_series().unique().sort().to_list()
        val_days = pretrain_val_dataset.select('date_id').collect().to_series().unique().sort().to_list()
        if verbose:
            print('Train days: ', train_days)
            print('Val days: ', val_days)
        
         
        model, initial_wr2_score = train_with_es(
            init_model=None, 
            train_dl=build_splits(pretrain_dataset, features),
            val_dl=build_splits(pretrain_val_dataset, features),
            use_weighted_loss=use_weighted_loss,
            metric=metric,
            es_patience=50,
            params=params,
            output_dir=tmp_checkpoint_dir
        )
        
        logging.info(f'Initial wr2 score: {initial_wr2_score}')
        trial.set_user_attr("initial_wr2_score", initial_wr2_score)
        
    
        X_val, y_val, w_val = build_splits(online_learning_dataset.filter(pl.col('date_id') >= start_eval_from), features)
        offline_score = evaluate_model(model, X_val, y_val, w_val)
        trial.set_user_attr("offline_score", offline_score)
        logging.info(f'Offline evaluation: {offline_score:.3f}')
        logging.info(f'Initial wr2_score: {initial_wr2_score:.3f}')
        
        del X_val, y_val, w_val
        gc.collect()
   

        max_old_dataset_date = pretrain_dataset.select('date_id').max().collect().item()
        old_dataset = pretrain_dataset.filter(pl.col('date_id').is_between(max_old_dataset_date-OLD_DATASET_MAX_HISTORY, max_old_dataset_date)).clone()  
        new_dataset = pretrain_val_dataset.clone()
        
        y_hat = []
        y = []
        weights = []
        daily_r2 = []
        partition_9_id_start = None
            
        date_idx = 0
        date_ids = online_learning_dataset.select('date_id').collect().to_series().unique().sort().to_list()
        batch_size = 100
        for i in range(0, len(date_ids), batch_size):
            batch_online_learning_dataset = online_learning_dataset.filter(pl.col('date_id').is_between(date_ids[i], date_ids[min(i+batch_size, len(date_ids))-1])).collect()
            n_batch_days = batch_online_learning_dataset['date_id'].n_unique()
            for date_id, test in tqdm(batch_online_learning_dataset.group_by('date_id', maintain_order=True), total=n_batch_days):
                date_id = date_id[0]
            
                if (date_idx + 1) % train_every == 0:                
                    max_date = new_dataset.select('date_id').max().collect().item()
                    new_validation_dataset = new_dataset.filter(pl.col('date_id') > max_date - last_n_days_es)
                    new_training_dataset = new_dataset.filter(pl.col('date_id') <= max_date - last_n_days_es)
                    
                    old_days = old_dataset.select('date_id').unique().collect().to_series().sort().to_list()
                    train_days = new_training_dataset.select('date_id').unique().collect().to_series().sort().to_list()
                    val_days = new_validation_dataset.select('date_id').unique().collect().to_series().sort().to_list()
                    if verbose:
                        print('Old days: ', old_days)
                        print('Train days: ', train_days)
                        print('Val days: ', val_days)
                    
                    new_training_dataset_len = new_training_dataset.select(pl.len()).collect().item()
                    old_dataset_len = old_dataset.select(pl.len()).collect().item()
                    old_data_len = min(int(old_data_fraction * new_training_dataset_len / (1 - old_data_fraction)), old_dataset_len)
                    print(new_training_dataset_len, old_data_len, old_data_fraction)
                    
                    train_dl = build_splits(pl.concat([old_dataset.collect().sample(n=old_data_len).lazy(), new_training_dataset]), features)
                    val_dl = build_splits(new_validation_dataset, features)
                    
                    logging.info(f'Starting fine tuning at date {date_id}')

                    params['learning_rate'] = max(params['learning_rate'] * lr_decay, 1e-6)
                    
                    model, _ = train_with_es(
                        init_model= model, 
                        train_dl=train_dl,
                        val_dl=val_dl,
                        use_weighted_loss=use_weighted_loss,
                        metric=metric,
                        es_patience=es_patience,
                        params=params,
                        output_dir=tmp_checkpoint_dir
                    )
                    
                    del train_dl, val_dl
                    gc.collect()
                    
                    max_old_dataset_date = new_training_dataset.select('date_id').max().collect().item()
                    old_dataset = pl.concat([
                        old_dataset,
                        new_training_dataset
                    ]).filter(
                        pl.col('date_id').is_between(max_old_dataset_date-OLD_DATASET_MAX_HISTORY, max_old_dataset_date)
                    )
                    new_dataset = new_validation_dataset


                date_idx += 1
                test_ = test.select(AUX_COLS + features)
                new_dataset = test_.lazy() if new_dataset is None else pl.concat([new_dataset, test_.lazy()])
                
                if date_id == PARTITIONS_DATE_INFO[9]['min_date']:
                    partition_9_id_start = len(y_hat)
                
                if date_id >= start_eval_from:
                    preds = model.predict(test_.select(features).to_numpy()).flatten()
                    y_hat.append(preds)
                    y.append(test_.select(['responder_6']).to_numpy().flatten())
                    weights.append(test_.select(['weight']).to_numpy().flatten())
                    daily_r2.append(r2_score(y_pred=preds, y_true=y[-1], sample_weight=weights[-1]))
            
            
            
        
        score = weighted_r2_score(np.concatenate(y_hat), np.concatenate(y), np.concatenate(weights))
        partition_9_score = weighted_r2_score(np.concatenate(y_hat[partition_9_id_start:]), np.concatenate(y[partition_9_id_start:]), np.concatenate(weights[partition_9_id_start:])) if partition_9_id_start else None
        daily_r2 = np.array(daily_r2)
        sharpe = np.mean(daily_r2) / np.std(daily_r2)
        stability_index = np.sum(daily_r2 > 0) / daily_r2.shape[0]
        trial.set_user_attr("partition_9_score", partition_9_score)
        trial.set_user_attr("stability_index", stability_index)
        
        logging.info(f'r2={score}, sharpe_ratio={sharpe}, partition_9_r2={partition_9_score}, stability={stability_index}')
        
        del model, y, y_hat, weights, old_dataset, new_dataset, new_training_dataset, new_validation_dataset
        gc.collect()
        
        return score, sharpe
    
    study = optuna.create_study(directions=['maximize', 'maximize'], study_name=study_name, storage=storage, load_if_exists=True)    
    study.optimize(obj_function, n_trials=n_trials)
    return study.trials_dataframe()
        

def main(dataset_path, output_dir, study_name, n_trials, storage):
    data_args = {'include_time_id': True, 'include_intrastock_norm_temporal': True}
    config = DataConfig(**data_args)
    loader = DataLoader(data_dir=dataset_path, config=config)
    
    pretraining_dataset = loader.load_with_partition(6, 8)
    # pretraining_dataset = loader.load(1000, 1150)
    features = loader.features
    print(f'Loaded features: {features}')
    pretraining_dataset = pretraining_dataset.select(AUX_COLS + features)
        
        
    max_date = pretraining_dataset.select(pl.col('date_id').max()).collect().item()
    pretraining_val_dataset = pretraining_dataset.filter(pl.col('date_id') > max_date - 14)
    pretraining_dataset = pretraining_dataset.filter(pl.col('date_id') <= max_date - 14)

    online_learning_dataset = loader.load_with_partition(9, 9)
    start_eval_from = PARTITIONS_DATE_INFO[9]['min_date']
    
    # online_learning_dataset = loader.load(1151, 1300)
    # start_eval_from = 1151

    online_learning_dataset = online_learning_dataset.select(AUX_COLS + features)
    
        
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    trials_df = optimize_parameters(output_dir, pretraining_dataset, pretraining_val_dataset, online_learning_dataset, features, study_name, n_trials, 
                                    storage, start_eval_from=start_eval_from)
    
    # params_file_path = os.path.join(output_dir, 'best_params.json')
    # logging.info(f'Best parameters: {best_params}')
    # logging.info(f'Saving the best parameters at: {params_file_path}')
    # with open(params_file_path, 'w') as params_file:
    #     json.dump(best_params, params_file)    
        
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
    model_name = f'lgbm_online_tuning_{timestamp}'
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(Path(DATASET_DIR), output_dir, STUDY_NAME, N_TRIALS, STORAGE)

from sklearn.metrics import r2_score
from tqdm import tqdm

from prj.config import DATA_DIR, EXP_DIR
from prj.data.data_loader import PARTITIONS_DATE_INFO, DataConfig, DataLoader
import optuna
import argparse
from datetime import datetime
import os
import logging
import polars as pl
import numpy as np
import gc
from pathlib import Path
import lightgbm as lgb
from prj.metrics import weighted_mae
from prj.oamp.oamp import OAMP
from prj.oamp.oamp_config import ConfigOAMP
from prj.metrics import weighted_mae, weighted_mse, weighted_rmse


AUX_COLS = ['date_id', 'time_id', 'symbol_id', 'weight', 'responder_6']

LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def squared_weighted_error_loss_fn(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w.reshape(-1, 1) * ((y_true.reshape(-1, 1) - y_pred) ** 2)

def absolute_weighted_error_loss_fn(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w.reshape(-1, 1) * np.abs(y_true.reshape(-1, 1) - y_pred)

def log_cosh_weighted_loss_fn(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> np.ndarray:    
    return w.reshape(-1, 1) * np.log(np.cosh(y_true.reshape(-1, 1) - y_pred))

LOSSES_DICT = {
    'mae': absolute_weighted_error_loss_fn,
    'mse': squared_weighted_error_loss_fn,
    'log_cosh': log_cosh_weighted_loss_fn
}

def build_splits(df: pl.LazyFrame, features: list[str]):
    X = df.select(features).cast(pl.Float32).collect().to_numpy()
    y = df.select('responder_6').cast(pl.Float32).collect().to_series().to_numpy()
    w = df.select('weight').cast(pl.Float32).collect().to_series().to_numpy()
    info = df.select(['date_id', 'time_id', 'symbol_id']).collect().to_numpy()
    return X, y, w, info


def metrics(y_true, y_pred, weights):
    return {
        'r2_w': r2_score(y_true, y_pred, sample_weight=weights),
        'mae_w': weighted_mae(y_true, y_pred, weights=weights),
        'mse_w': weighted_mse(y_true, y_pred, weights=weights),
        'rmse_w': weighted_rmse(y_true, y_pred, weights=weights),
    }


def _sample_oamp_params(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    params = {
        "agents_weights_upd_freq": trial.suggest_int("agents_weights_upd_freq", 1, 50, step=1),
        "loss_fn_window": trial.suggest_int("loss_fn_window", 100, 1e6, step=1e4),
        "loss_function": trial.suggest_categorical("loss_function", ["mse", "mae", "log_cosh"]),
        "agg_type": trial.suggest_categorical("agg_type", ["mean", "median", "max"]),
    }
                
    return params

def optimize_parameters(output_dir, agents: list, evaluation_dataset: pl.LazyFrame, features: list, study_name, n_trials, storage):
    
    evaluation_dataset = evaluation_dataset.sort('date_id', 'symbol_id', 'time_id')
        
    X, y, w, info = build_splits(evaluation_dataset, features)
    
    
    batch_size = 500000
    n_agents = len(agents)
    n_samples = X.shape[0]
    agents_predictions = np.empty((n_samples, n_agents))

    for start in tqdm(range(0, n_samples, batch_size)):
        end = min(start + batch_size, n_samples)
        for i, agent in enumerate(agents):
            agents_predictions[start:end, i] = agent.predict(X[start:end])
    
    del X
    gc.collect()
    
    date_idx_info = pl.DataFrame(info[:, 0], schema=['date_id']).with_row_index().group_by('date_id').agg(
        pl.col('index').min().alias('start'),
        pl.col('index').max().alias('end'),
    ).sort('date_id').to_numpy()

    offline_scores = {}
    for i in range(len(agents)):
        offline_scores[f'agent_{i}'] = r2_score(y, agents_predictions[:, i], sample_weight=w)
    
    offline_scores['mean'] = r2_score(y, agents_predictions.mean(axis=1), sample_weight=w)
    offline_scores['median'] = r2_score(y, np.median(agents_predictions, axis=1), sample_weight=w)
    max_agent_score = np.max([offline_scores[f'agent_{i}'] for i in range(n_agents)])
    
    baseline = offline_scores['mean']
    print(f'Offline scores: {offline_scores}')
    
    def obj_function(trial):
        
        logging.info(f'Trial {trial.number}')
        
        additional_args = {}
        oamp_params = _sample_oamp_params(trial, additional_args=additional_args)
        
        loss_fn = oamp_params.pop('loss_function')
        
        n_agents = len(agents)
                   
        agents_losses = LOSSES_DICT[loss_fn](y, agents_predictions, w)

        y_hat_ens = []
        daily_r2 = []
        multi_stock_oamp = trial.suggest_categorical("multi_stock_oamp", [True, False])
        
        if multi_stock_oamp:
            oamp_symbol_dict = {} 
            for i, (_, start, end) in enumerate(tqdm(date_idx_info)):
                if i > 0:                
                    _, prev_start, prev_end = date_idx_info[i-1]
                    daily_prev_agents_losses = agents_losses[prev_start:prev_end+1, :]
                                    
                    unique_symbols, symbols_count = np.unique(info[prev_start:prev_end+1, 2], return_counts=True)
                    for i, (s, c) in enumerate(zip(unique_symbols, np.cumsum(symbols_count))):
                        l = symbols_count[i]
                        s_s = c - l
                        s_e = c
                        oamp_symbol_dict[s].step(daily_prev_agents_losses[s_s:s_e, :])
                        
                    
                unique_symbols, symbols_count = np.unique(info[start:end+1, 2] , return_counts=True)
                curr_agents_predictions = agents_predictions[start:end+1, :]
                
                if not set(unique_symbols).issubset(oamp_symbol_dict.keys()):
                    missing_symbols = set(unique_symbols) - set(oamp_symbol_dict.keys())
                    for symbol in missing_symbols:
                        oamp_config = ConfigOAMP(oamp_params.copy())
                        oamp_symbol_dict[symbol] = OAMP(n_agents, oamp_config)
                    
                
                y_hat_ens_daily = []
                for i, (s, c) in enumerate(zip(unique_symbols, np.cumsum(symbols_count))):
                    l = symbols_count[i]
                    s_s = c - l
                    s_e = c
                    y_hat_ens_daily.append(oamp_symbol_dict[s].compute_prediction(curr_agents_predictions[s_s:s_e, :]))
                y_hat_ens_daily = np.concatenate(y_hat_ens_daily)
                
                daily_r2.append(r2_score(y[start:end+1], y_hat_ens_daily, sample_weight=w[start:end+1]))
                y_hat_ens.append(y_hat_ens_daily)
        else:
            oamp = OAMP(n_agents, ConfigOAMP(oamp_params.copy()))
            
            for i, (_, start, end) in enumerate(tqdm(date_idx_info)):
                if i > 0:                
                    _, prev_start, prev_end = date_idx_info[i-1]
                    daily_prev_agents_losses = agents_losses[prev_start:prev_end+1, :]
                    oamp.step(daily_prev_agents_losses[prev_start:prev_end+1, :])

                y_hat_ens_daily = oamp.compute_prediction(agents_predictions[start:end+1, :])
                daily_r2.append(r2_score(y[start:end+1], y_hat_ens_daily, sample_weight=w[start:end+1]))
                y_hat_ens.append(y_hat_ens_daily)

            
                
        y_hat_ens = np.concatenate(y_hat_ens)
        
        trial.set_user_attr('mean_score', offline_scores['mean'])
        trial.set_user_attr('max_agent_score', max_agent_score)
        
        oamp_score = r2_score(y, y_hat_ens, sample_weight=w)
        gain = oamp_score - baseline
        trial.set_user_attr('oamp_score', oamp_score)
        logging.info(f'Trial {trial.number} -> Gain: {gain}, OAMP score: {oamp_score}, mean: {offline_scores["mean"]}, max: {max_agent_score}')
        
        
        return gain
            
    sampler = optuna.samplers.TPESampler(multivariate=True)
    study = optuna.create_study(directions=['maximize'], study_name=study_name, storage=storage, load_if_exists=True, sampler=sampler)    
    study.optimize(obj_function, n_trials=n_trials)
    return study.trials_dataframe()
        

def main(dataset_path, output_dir, study_name, n_trials, storage):
    data_args = {'include_time_id': True, 'include_intrastock_norm_temporal': True}
    config = DataConfig(**data_args)
    loader = DataLoader(data_dir=dataset_path, config=config)
    
    evaluation_dataset = loader.load_with_partition(8, 9)
    
    # evaluation_dataset = loader.load(1151, 1155)
    
    features = loader.features
    print(f'Loaded features: {features}')
    
    evaluation_dataset = evaluation_dataset.select(AUX_COLS + features)

    lgbm_files = [
        "/home/lorecampa/projects/jane_street_forecasting/dataset/models/lgbm/lgbm_maxbin_63_0_7_324272949.txt",
        "/home/lorecampa/projects/jane_street_forecasting/dataset/models/lgbm/lgbm_maxbin_63_0_7_917304356.txt",
        "/home/lorecampa/projects/jane_street_forecasting/dataset/models/lgbm/lgbm_maxbin_63_0_7_3234493111.txt",
        "/home/lorecampa/projects/jane_street_forecasting/dataset/models/lgbm/lgbm_maxbin_63_0_7_3729223622.txt"
    ]

    agents = []
    
    for lgbm_file in lgbm_files:
        agents.append(lgb.Booster(model_file=lgbm_file))
  
  
    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    optuna.logging.disable_default_handler() # Stop showing logs in sys.stderr (prevents double logs)

    trials_df = optimize_parameters(output_dir, agents, evaluation_dataset, features, study_name, n_trials, storage)
    
        
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
    OUTPUT_DIR = args.output_dir or EXP_DIR / 'oamp'
    DATASET_DIR = args.dataset_path or DATA_DIR 
    N_TRIALS = args.n_trials
    STUDY_NAME = args.study_name
    STORAGE = args.storage
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f'oamp_{timestamp}' if STUDY_NAME is None else STUDY_NAME
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
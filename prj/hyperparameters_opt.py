import optuna
from lightgbm import LGBMRegressor

def sample_oamp_params(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    params = {
        "agents_weights_upd_freq": trial.suggest_int("agents_weights_upd_freq", 0, 50, step=1),
        "loss_fn_window": trial.suggest_int("loss_fn_window", 1, 100000, step=1000),
        "loss_function": trial.suggest_categorical("loss_function", ["mse", "mae", "log_cosh"]),
        "agg_type": trial.suggest_categorical("agg_type", ["mean", "median", "max"]),
    }
    
    params['agents_weights_upd_freq'] = max(params['agents_weights_upd_freq'], 1)
    params['loss_fn_window'] = max(params['loss_fn_window'], 1)
            
    return params
    
def sample_lgbm_params(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    use_gpu = additional_args.get("use_gpu", False)
    params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "num_leaves": trial.suggest_int("num_leaves", 4, 1024),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.1, 0.7),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.8),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1000, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1000, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-6, 1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-7, 1e-1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 10000, log=True),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            "device_type": "gpu" if use_gpu else "cpu",
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

def sample_catboost_params(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    use_gpu = additional_args.get("use_gpu", False)
    params = {
            'iterations': trial.suggest_int('iterations', 100, 1000, step=50),
            'learning_rate': trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 1e5, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'MVS']),
            'subsample': trial.suggest_float("subsample", 0.05, 0.7),
            'fold_permutation_block': trial.suggest_int('fold_permutation_block', 1, 100),
            'border_count': trial.suggest_int('border_count', 8, 512, log=True),
            # 'has_time': trial.suggest_categorical('has_time', [True, False]),
            'cat_features': additional_args.get('cat_features', []),
            'task_type': 'GPU' if use_gpu else 'CPU',
        }
        
    if params['grow_policy'] == 'Lossguide':
        params['max_leaves'] = trial.suggest_int("max_leaves", 8, 64, log=True)
        params['depth'] = trial.suggest_int("depth", 2, 14)
        if not use_gpu:
            params['langevin'] = trial.suggest_categorical("langevin", [True, False])
            if params['langevin']:
                params['diffusion_temperature'] = trial.suggest_float('diffusion_temperature', 1e2, 1e6, log=True)
    else: # for Lossguide, Cosine is not supported. Newton and NewtonL2 are only supported in GPU
        if not use_gpu:
            params['sampling_frequency'] = trial.suggest_categorical('sampling_frequency', ['PerTree', 'PerTreeLevel'])
            params['score_function'] = trial.suggest_categorical('score_function', ['Cosine', 'L2'])
        
        params['depth'] = trial.suggest_int("depth", 2, 10)

    if params['grow_policy'] != 'SymmetricTree':
        params['min_data_in_leaf'] = trial.suggest_float('min_data_in_leaf', 10, 1000)     
            
    if params['bootstrap_type'] == 'MVS' and not use_gpu:
        params['mvs_reg'] = trial.suggest_float('mvs_reg', 1e-4, 1e4, log=True)
    
    if not use_gpu:
        params['rsm'] = trial.suggest_float("rsm", 0.05, 0.8, log=True)

    if use_gpu:     
        params['border_count'] = trial.suggest_int('border_count', 8, 254, log=True) # suggested to be small on GPU
    else:
        params['random_strength'] = trial.suggest_float('random_strength', 1e-6, 1e2, log=True)
    
    return params
        
        
def sample_xgb_params(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    use_gpu = additional_args.get("use_gpu", False)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 5000, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1000, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1000, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'max_leaves': trial.suggest_int('max_leaves', 8, 1024),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'max_bin': trial.suggest_int('max_bin', 8, 512),
        'gamma': trial.suggest_float('gamma', 1e-7, 10, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-7, 10, log=True),
        'subsample': trial.suggest_float('subsample', 0.05, 0.5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
        "enable_categorical": True, #TODO: should it be false?
        "device": "gpu" if use_gpu else "cpu",
    }    
    return params

def _sample_base_neural_params(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    params = {
        'use_gaussian_noise': trial.suggest_categorical('use_gaussian_noise', [True, False]),
        'learning_rate': trial.suggest_float('learning_rate', 5e-5, 1e-3, log=True),
        'l1_lambda': trial.suggest_float('l1_lambda', 1e-7, 1e-4, log=True),
        'l2_lambda': trial.suggest_float('l2_lambda', 1e-7, 1e-4, log=True),
        'use_scheduler': trial.suggest_categorical('use_scheduler', [True, False]),
    }
    
    if params['use_gaussian_noise']:
        params['gaussian_noise_std'] = trial.suggest_float('gaussian_noise_std', 1e-3, 1)
        
    if params['use_scheduler']:
        params['scheduling_rate'] = trial.suggest_float('scheduling_rate', 1e-3, 0.1)
        
    return params

def sample_mlp_params(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    params = _sample_base_neural_params(trial=trial, additional_args=additional_args)
    
    archs = {
        'tiny': [64, 32],
        'medium': [128, 64, 32],
        'large': [256, 128, 64],
        'xlarge': [512, 256, 128],
        'xxlarge': [1024, 512, 256, 128],
    }
    
    arch_type = trial.suggest_categorical('arch_type', sorted(list(archs.keys())))
    
    params.update({
        "hidden_dims": archs[arch_type],
        "use_dropout": trial.suggest_categorical("use_dropout", [True, False]),
        "use_bn": trial.suggest_categorical("use_bn", [True, False]),
    })
    if params['use_dropout']:
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05)
        

    return params


def sample_cnn_resnet_params(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    params = _sample_base_neural_params(trial, additional_args)
    params.update({
        'conv_filters': trial.suggest_categorical('conv_filters', [16, 32]),
        'kernel_size': trial.suggest_categorical('kernel_size', [2, 4, 8]),
        'rnn_neurons': trial.suggest_int('rnn_neurons', 32, 64, step=16),
        # Uncomment if needed:
        # 'rnn_decay_factor': trial.suggest_categorical('rnn_decay_factor', [1, 1.5, 2, 2.5, 3]),
        'kernel_initializer': trial.suggest_categorical('kernel_initializer', ['he_uniform', 'he_normal', 'glorot_uniform']),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, log=True),
        'layers': trial.suggest_int('layers', 4, 6),
    })
    return params

def sample_tcn_params(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    params = _sample_base_neural_params(trial, additional_args)
    params.update({
        'conv_filters': trial.suggest_categorical('conv_filters', [16, 32, 64]),
        'kernel_size': trial.suggest_categorical('kernel_size', [2, 4, 8, 16, 24, 32]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, log=True),
        'layers': trial.suggest_int('layers', 4, 6),
        'kernel_initializer': trial.suggest_categorical('kernel_initializer', ['he_uniform', 'he_normal', 'glorot_uniform']),
    })
    return params



def sample_rnn_params(trial: optuna.Trial, additional_args: dict = {}) -> dict:
    params = _sample_base_neural_params(trial, additional_args)
    params.update({
        'layers': trial.suggest_int('layers', 1, 4),
        'use_lstm': trial.suggest_categorical('use_lstm', [False, True]),
        'use_batch_norm': trial.suggest_categorical('use_batch_norm', [False, True]),
        'start_neurons': trial.suggest_int('start_neurons', 32, 128, step=32),
        'decay_factor': trial.suggest_categorical('decay_factor', [1, 1.5, 2, 2.5, 3]),
    })
    if params['use_batch_norm']:
        params.update({
            'momentum': trial.suggest_float('momentum', 0.1, 0.99, log=True),
        })
    return params

SAMPLER = {
    "oamp": sample_oamp_params,
    "lgbm": sample_lgbm_params,
    "catboost": sample_catboost_params,
    "xgb": sample_xgb_params,
    "mlp": sample_mlp_params,
    "cnn_resnet": sample_cnn_resnet_params,
    "tcn": sample_tcn_params,
    "rnn": sample_rnn_params
}
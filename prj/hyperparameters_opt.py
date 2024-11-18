import optuna

def sample_oamp_params(trial: optuna.Trial, additional_args: dict) -> dict:
    agents_weights_upd_freq = trial.suggest_int("agents_weights_upd_freq", 1, 10)
    loss_fn_window = trial.suggest_int("loss_fn_window", 1, 10)
    action_thresh = trial.suggest_float("action_thresh", 0.5, 1, step=0.05)
    return {
        "agents_weights_upd_freq": agents_weights_upd_freq,
        "loss_fn_window": loss_fn_window,
        "action_thresh": action_thresh,
    }
    
    

SAMPLER = {
    "oamp": sample_oamp_params,
}
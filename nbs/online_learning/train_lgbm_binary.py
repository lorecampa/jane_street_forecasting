import optuna
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

from prj.config import DATA_DIR
from prj.utils import set_random_seed

# study_name = "lgbm_offline_2025-01-04_19-42-20"
# storage = "mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet"
# study = optuna.load_study(study_name=study_name, storage=storage)

# trial_with_highest_r2 = max(study.best_trials, key=lambda t: t.values[0])
# trial_with_highest_r2, trial_with_highest_r2.number


train_params = {
    "use_weighted_loss": False,
    "max_depth": 8,
    "num_leaves": 171,
    "subsample_freq": 11,
    "subsample": 0.571590696990308,
    "learning_rate": 0.07316257046174213,
    "colsample_bytree": 0.2455489011034214,
    "colsample_bynode": 0.8745192679905138,
    "reg_lambda": 54.62908352171549,
    "reg_alpha": 0.5403041261001198,
    "min_split_gain": 0.17117093723499863,
    "min_child_weight": 0.008267229603730341,
    "min_child_samples": 38,
    "extra_trees": True,
    "num_boost_round": 246
}

num_boost_round = train_params.pop("num_boost_round")
use_weighted_loss = train_params.pop("use_weighted_loss")


train_data = lgb.Dataset(
    data='/home/lorecampa/projects/jane_street_forecasting/dataset/binary/lgbm_maxbin_63_0_7.bin',
    params={
        'feature_pre_filter': False,
        'max_bin': 63,
        # 'device': 'cpu',
        'verbosity': -1,
        'device': 'gpu',
    }
)



X_val = np.random.rand(10, 134)
y_val = np.random.rand(10)
w_val = np.random.rand(10)
val_data = lgb.Dataset(data=X_val, label=y_val, weight=w_val if use_weighted_loss else None, reference=train_data)


print(f'Num boost round: {num_boost_round}, weighted loss: {use_weighted_loss}')


seeds = sorted([np.random.randint(2**32 - 1, dtype="int64").item() for _ in range(4)])

for seed in tqdm(seeds):
    seed_params = train_params.copy()
    seed_params['seed'] = seed
    set_random_seed(seed)

    print(f"Seed: {seed}")
    model = lgb.train(
        params=seed_params,
        train_set=train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(period=10)],
    )

    model.save_model(DATA_DIR / 'models' / 'lgbm' / f'lgbm_maxbin_63_0_7_{seed}.txt')

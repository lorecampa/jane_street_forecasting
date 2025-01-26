import optuna
import numpy as np
import lightgbm as lgb
from prj.config import DATA_DIR
from prj.data.data_loader import DataConfig, DataLoader
from prj.utils import set_random_seed
import gc


study_name = "lgbm_offline_2025-01-11_11-45-06"
storage = "mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet"
study = optuna.load_study(study_name=study_name, storage=storage)

best_trial = study.best_trial


params = best_trial.params
print(params)



train_params = params.copy()
num_boost_round = train_params.pop("num_boost_round")
use_weighted_loss = train_params.pop("use_weighted_loss")


# train_data = lgb.Dataset(
#     data='/home/lorecampa/projects/jane_street_forecasting/dataset/binary/lgbm_maxbin_63_0_9.bin',
#     params={
#         'feature_pre_filter': False,
#         'max_bin': 63,
#         'device': 'cpu',
#         'verbosity': -1,
#         # 'device': 'gpu',
#     }
# )

data_args = {'include_time_id': True, 'include_intrastock_norm_temporal': True}
config = DataConfig(**data_args)
loader = DataLoader(data_dir=DATA_DIR, config=config)

train_dataset = loader.load_with_partition(6, 9)
X_train, y_train, w_train, info_train = loader._build_splits(train_dataset)
train_data = lgb.Dataset(data=X_train, label=y_train, weight=w_train if use_weighted_loss else None)

del X_train, y_train, w_train, info_train
gc.collect()

X_val = np.random.rand(10, 134)
y_val = np.random.rand(10)
w_val = np.random.rand(10)
val_data = lgb.Dataset(data=X_val, label=y_val, weight=w_val if use_weighted_loss else None, reference=train_data)


print(f'Num boost round: {num_boost_round}, weighted loss: {use_weighted_loss}')

model = lgb.train(
    params=train_params,
    train_set=train_data,
    num_boost_round=num_boost_round,
    valid_sets=[val_data],
    callbacks=[lgb.log_evaluation(period=10)],
)

model.save_model(DATA_DIR / 'models' / 'lgbm' / f'lgbm_maxbin_63_6_9_kcross.txt')

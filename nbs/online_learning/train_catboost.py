import gc
from catboost import CatBoostRegressor, Pool
from prj.config import DATA_DIR
from prj.data.data_loader import DataConfig, DataLoader


use_weighted_loss = False
params = {
    'verbose': 50,
    'iterations': 717,
    'learning_rate': 0.019678599283449602,
    'depth': 8,
    'has_time': False,
    'bootstrap_type': 'Bernoulli',
    'reg_lambda': 0.00924440304487912,
    'min_data_in_leaf': 72,
    'subsample': 0.63603957073985,
    'task_type': 'CPU',
}

model = CatBoostRegressor(**params)

config = DataConfig(**{'include_intrastock_norm_temporal': True, 'include_time_id': True})
loader = DataLoader(data_dir=DATA_DIR, config=config)

start_p, end_p = 5, 7

train_df = loader.load_with_partition(start_p, end_p)
X_train, y_train, w_train, _ = loader._build_splits(train_df)

train_pool = Pool(data=X_train, label=y_train, weight=w_train if use_weighted_loss else None)

del X_train, y_train, w_train
gc.collect()

model.fit(
    train_pool,
    verbose=50
)

model_name = f'catboost_{start_p}_{end_p}'
if use_weighted_loss:
    model_name += '_w'
    
model.save_model(DATA_DIR / 'catboost' / f'{model_name}.cbm')
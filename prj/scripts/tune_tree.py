from datetime import datetime
import argparse
import gc
from logging import Logger
import os
from prj.agents.AgentNeuralRegressor import NEURAL_NAME_MODEL_CLASS_DICT
from prj.agents.AgentTreeRegressor import TREE_NAME_MODEL_CLASS_DICT
from prj.agents.factory import AgentsFactory
from prj.config import DATA_DIR, EXP_DIR
from prj.data import DATA_ARGS_CONFIG
from prj.data.data_loader import PARTITIONS_DATE_INFO, DataConfig, DataLoader
from prj.hyperparameters_opt import SAMPLER
from prj.logger import setup_logger
from prj.tuner import Tuner
from prj.utils import BlockingTimeSeriesSplit, CombinatorialPurgedKFold, str_to_dict_arg
import polars as pl
import numpy as np

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        default="lgbm",
        help="Model name"
    )
    
    parser.add_argument(
        '--start_dt',
        type=int,
        default=PARTITIONS_DATE_INFO[5],
    )
    parser.add_argument(
        '--end_dt',
        type=int,
        default=PARTITIONS_DATE_INFO[9],
    )
    parser.add_argument(
        '--end_val_dt',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help="number of iterations of optuna"
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=1
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--storage',
        type=str,
        default=None
    )
    parser.add_argument(
        '--study_name',
        type=str,
        default=None
    )
    parser.add_argument(
        '--verbose',
        type=int,
        help='Enable verbose',
        default=0
    )
    parser.add_argument(
        '--custom_model_args',
        type=str_to_dict_arg,
        default='{}',
        help="Custom arguments in dictionary format"
    )
    parser.add_argument(
        '--custom_learn_args',
        type=str_to_dict_arg,
        default='{}',
        help="Custom arguments in dictionary format"
    )
    parser.add_argument(
        '--custom_data_args',
        type=str_to_dict_arg,
        default='{}',
        help="Custom arguments in dictionary format"
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        default=False,
        help="Run only training, no optimization"
    )
    parser.add_argument(
        '--kcross',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=False,
        help="Run only training with gpu"
    )


    return parser.parse_args()


class TreeTuner(Tuner):
    def __init__(
        self,
        model_type: str,
        data_dir: str = DATA_DIR,
        start_dt: int = PARTITIONS_DATE_INFO[5],
        end_dt: int = PARTITIONS_DATE_INFO[9],
        end_val_dt: int = None,
        val_ratio: float = 0.15,
        out_dir: str = '.',
        n_seeds: int = None,
        storage: str = None,
        use_gpu: bool = False,
        study_name: str = None,
        n_trials: int = 50,
        verbose: int = 0,
        kcross: bool = False,
        custom_model_args: dict = {},
        custom_learn_args: dict = {},
        custom_data_args: dict = {},
        logger: Logger = None,
    ):
        super().__init__(
            model_type=model_type,
            data_dir=data_dir,
            out_dir=out_dir,
            n_seeds=n_seeds,
            storage=storage,
            study_name=study_name,
            use_gpu=use_gpu,
            n_trials=n_trials,
            verbose=verbose,
            custom_model_args=custom_model_args,
            custom_learn_args=custom_learn_args,
            custom_data_args=custom_data_args,
            logger=logger,
            kcross=kcross
        )
        
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.end_val_dt = end_val_dt
        self.val_ratio = val_ratio
        self.logger = logger
        
        self.logger.info(f'Using model: {model_type}, start_dt: {start_dt}, end_dt: {end_dt}, val_ratio: {val_ratio}')
        
        model_dict = TREE_NAME_MODEL_CLASS_DICT
        
        self.model_class = model_dict[self.model_type]
        self.model = AgentsFactory.build_agent({'agent_type': self.model_type, 'seeds': self.seeds})

        data_args = {}
        # data_args = {'include_intrastock_norm_temporal': True, 'include_time_id': True}
        data_args.update(self.custom_data_args)
        config = DataConfig(**data_args)
        self.loader = DataLoader(data_dir=data_dir, config=config)
        self.logger.info(f'Loading data from {data_dir} with config args {data_args}...')
        # train_df, val_df = self.loader.load_train_and_val(self.start_dt, self.end_dt, self.val_ratio)
        if self.end_val_dt is not None:
            complete_df = self.loader.load(self.start_dt, self.end_val_dt)
            train_df = complete_df.filter(pl.col('date_id').le(self.end_dt))
            val_df = complete_df.filter(pl.col('date_id').gt(self.end_dt))
        else:
            train_df, val_df = self.loader.load_train_and_val(self.start_dt, self.end_dt, self.val_ratio)
            complete_df = pl.concat([train_df, val_df])
    
        min_train_date = train_df.select('date_id').min().collect().item()
        max_train_date = train_df.select('date_id').max().collect().item()
        min_val_date = val_df.select('date_id').min().collect().item()
        max_val_date = val_df.select('date_id').max().collect().item()
        self.logger.info(f'Train date range: {min_train_date} - {max_train_date}, Val date range: {min_val_date} - {max_val_date}')
        n_dates_train = train_df.select('date_id').collect().n_unique()
        n_dates_val = val_df.select('date_id').collect().n_unique()
        self.logger.info(f'N dates train: {n_dates_train}, N dates val: {n_dates_val}')
            
        n_rows_train = train_df.select(pl.len()).collect().item()
        self.data = self.loader._build_splits(complete_df)
        self.train_data = tuple(data[:n_rows_train] for data in self.data)
        self.val_data = tuple(data[n_rows_train:] for data in self.data)
        
        self.logger.info(f'Using features: {self.loader.features}. N features: {len(self.loader.features)}')
                
        self.logger.info(f'Train: {self.train_data[0].shape}, VAL: {self.val_data[0].shape}')
        
        self.model_args = {}
        self.learn_args = {}

        # cat_features_idx = [self.loader.features.index(f) for f in self.loader.categorical_features]
        cat_features_idx = []
        if model_type == 'lgbm':
            self.model_args['verbose'] = -1
            if len(cat_features_idx) > 0:
                self.learn_args['categorical_feature'] = ','.join([str(c) for c in cat_features_idx])  
        elif model_type == 'catboost':
            self.learn_args['verbose'] = 100
        elif model_type == 'xgb':
            if len(cat_features_idx) > 0:
                self.model_args.update({
                    'enable_categorical': True,
                    'categorical_feature': cat_features_idx
                })
                
    def train(self, model_args:dict, learn_args: dict):
        X, y, w, _ = self.train_data
            
        self.model.train(
            X, y, w,
            model_args=model_args,
            learn_args=learn_args,
        )
        X_val, y_val, w_val, _ = self.val_data
        batch_size = None
        if self.use_gpu:
            batch_size = X_val.shape[0] // 5
        val_metrics = self.model.evaluate(X_val, y_val, w_val, batch_size=batch_size)
        
        return val_metrics
        
    def train_kcross(self, model_args:dict, learn_args: dict):
        kcross_type = 'blocking' # 'comb  
        n_splits = 3
        X, y, w, info = self.data
        if kcross_type == 'comb':
            kf = CombinatorialPurgedKFold(n_splits=n_splits + 1)
        elif kcross_type == 'blocking':
         kf = BlockingTimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f'Invalid kcross type: {kcross_type}')
        
        val_metrics = {}
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            if kcross_type == 'blocking':
                # Do like that to avoid creating copy of memory in case of continuous indexes
                min_train_idx, max_train_idx = train_index.min(), train_index.max()
                min_test_idx, max_test_idx = test_index.min(), test_index.max()
                self.logger.info(f'Fold {i}: {min_train_idx} - {max_train_idx} - {min_test_idx} - {max_test_idx}')
                X_k_train, X_k_test = X[min_train_idx:max_train_idx + 1], X[min_test_idx:max_test_idx + 1]
                y_k_train, y_k_test = y[min_train_idx:max_train_idx + 1], y[min_test_idx:max_test_idx + 1]
                w_k_train, w_k_test = w[min_train_idx:max_train_idx + 1], w[min_test_idx:max_test_idx + 1]
                dates_k_train, dates_k_test = info[min_train_idx:max_train_idx + 1][:, 0], info[min_test_idx:max_test_idx + 1][:, 0]
            else:
                X_k_train, X_k_test = X[train_index], X[test_index]
                y_k_train, y_k_test = y[train_index], y[test_index]
                w_k_train, w_k_test = w[train_index], w[test_index]
                dates_k_train, dates_k_test = info[train_index][:, 0], info[test_index][:, 0]

            # self.logger.info(np.unique(dates_k_train), np.unique(dates_k_test)) 
            self.logger.info(f'Fold {i}: {np.min(dates_k_train)} - {np.max(dates_k_train)} - {np.min(dates_k_test)} - {np.max(dates_k_test)}')              
            self.model.train(
                X_k_train, y_k_train, w_k_train,
                model_args=model_args,
                learn_args=learn_args,
            )
            self.logger.info(f'Fold {i}: Evaluating...')
            batch_size = None
            if self.use_gpu:
                batch_size = X_k_test.shape[0] // 5
            val_k_metrics = self.model.evaluate(X_k_test, y_k_test, w_k_test, batch_size=batch_size)
            for k, v in val_k_metrics.items():
                if k in val_metrics:
                    val_metrics[k].append(v)
                else:
                    val_metrics[k] = [v]
            self.logger.info(f"Fold {i}: {val_k_metrics['r2_w']:.3f}")
            
                    
        return val_metrics
            
            
            
                        
if __name__ == "__main__":
    args = get_cli_args()
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        print('Using GPU!')

    data_dir = args.data_dir if args.data_dir is not None else DATA_DIR    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    study_name = args.study_name if args.study_name is not None else \
        f'{args.model}_{args.n_seeds}seeds_{args.start_dt}_{args.end_dt}-{(args.end_val_dt) if args.end_val_dt is not None else args.val_ratio}_{timestamp}'

    
    val_ratio = args.val_ratio
    if args.train:
        out_dir = args.out_dir if args.out_dir is not None else str(EXP_DIR / 'train' / str(args.model) / f'{study_name}_{timestamp}')
        val_ratio = 0.
    else:
        out_dir = args.out_dir if args.out_dir is not None else str(EXP_DIR / 'tuning' / str(args.model) / study_name)
    storage = f'sqlite:///{out_dir}/optuna_study.db' if args.storage is None else args.storage
    
    logger = setup_logger(out_dir)
    logger.info(f'Tuning model: {args.model}')
    

    optimizer = TreeTuner(
        model_type=args.model,
        start_dt=args.start_dt,
        end_dt=args.end_dt,
        end_val_dt=args.end_val_dt,
        val_ratio=val_ratio,
        data_dir=data_dir,
        out_dir=out_dir,
        n_seeds=args.n_seeds,
        verbose=args.verbose,
        storage=storage,
        study_name=study_name,
        n_trials=args.n_trials,
        use_gpu=args.gpu,
        custom_model_args=args.custom_model_args,
        custom_learn_args=args.custom_learn_args,
        custom_data_args=args.custom_data_args,
        logger=logger,
        kcross=args.kcross
    )
    optimizer.create_study()

    if args.train:
        print('Training best trial')
        optimizer.train_best_trial()
        save_path = f'{out_dir}/train/model'
        os.makedirs(save_path, exist_ok=True)
        optimizer.model.save(save_path)
    else:
        optimizer.run()
    
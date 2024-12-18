from datetime import datetime
import argparse
import gc
from logging import Logger
import os
from prj.agents.AgentNeuralRegressor import NEURAL_NAME_MODEL_CLASS_DICT
from prj.agents.factory import AgentsFactory
from prj.config import DATA_DIR, EXP_DIR
from prj.data import DATA_ARGS_CONFIG
from prj.data.data_loader import DataConfig, DataLoader as BaseDataLoader
from prj.logger import setup_logger
from prj.model.torch.datasets.base import JaneStreetBaseDataset
from prj.tuner import Tuner
from prj.utils import str_to_dict_arg
from torch.utils.data import DataLoader
import polars as pl

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        default="mlp",
        help="Model name"
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
        '--start_dt',
        type=int,
        default=1100,
    )
    parser.add_argument(
        '--end_dt',
        type=int,
        default=1698,
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
        '--gpu',
        action='store_true',
        default=False,
        help="Run only training with gpu"
    )


    return parser.parse_args()


class NeuralTuner(Tuner):
    def __init__(
        self,
        model_type: str,
        start_dt: int = 1100,
        end_dt: int = None,
        data_dir: str = DATA_DIR,
        out_dir: str = '.',
        n_seeds: int = None,
        storage: str = None,
        use_gpu: bool = False,
        study_name: str = None,
        n_trials: int = 50,
        verbose: int = 0,
        early_stopping: bool = True,
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
            logger=logger
        )
        self.early_stopping = early_stopping
        self.start_dt = start_dt
        self.end_dt = end_dt
        print(f'Start date: {start_dt}, End date: {end_dt}')
        
        model_dict = NEURAL_NAME_MODEL_CLASS_DICT
        self.model_class = model_dict[self.model_type]
        self.model = AgentsFactory.build_agent({'agent_type': self.model_type, 'seeds': self.seeds})
        num_workers = 0
        
        data_args = {'zero_fill': True}
        data_args.update(self.custom_data_args)
        config = DataConfig(data_dir=data_dir, **data_args)
        self.loader = BaseDataLoader(config=config)
        self.features = self.loader.features
        train_ds, val_ds = self.loader.load_train_and_val(start_dt=self.start_dt, end_dt=self.end_dt, val_ratio=0.15)        
        es_ds = None
        es_ratio = 0.10
        if self.early_stopping:
            train_dates = train_ds.select('date_id').unique().collect().to_series().sort()
            split_point = int(len(train_dates) * (1 - es_ratio))
            split_date = train_dates[split_point]
            es_ds = train_ds.filter(pl.col('date_id').ge(split_date))
            train_ds = train_ds.filter(pl.col('date_id').lt(split_date))
        
        n_rows_train = train_ds.select(pl.len()).collect().item()
        n_dates_train = train_ds.select('date_id').unique().collect().count().item()
        n_rows_es = es_ds.select(pl.len()).collect().item() if self.early_stopping else 0
        n_dates_es = es_ds.select('date_id').unique().collect().count().item() if self.early_stopping else 0
        n_rows_val = val_ds.select(pl.len()).collect().item()
        n_dates_val = val_ds.select('date_id').unique().collect().count().item()
        print(f'N rows train: {n_rows_train}, ES: {n_rows_es}, VAL: {n_rows_val}')
        print(f'N dates train: {n_dates_train}, ES: {n_dates_es}, VAL: {n_dates_val}')
                    
        train_ds = JaneStreetBaseDataset(train_ds, features=self.features)
        if self.early_stopping:
            es_ds = JaneStreetBaseDataset(es_ds, features=self.features)
        self.val_ds = JaneStreetBaseDataset(val_ds, features=self.features)
        
        
        batch_size = 1024
        self.train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.es_dataloader = None
        if self.early_stopping:
            self.es_dataloader = DataLoader(es_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
            
        self.model_args = {
            'input_dim': (len(self.loader.features),),
            'output_dim': 1,
        }
        self.learn_args = {
            'max_epochs': 50,
            'gradient_clip_val': 10,
            'precision': '32-true',
            'use_model_ckpt': False,
        }

    
    def train(self, model_args:dict, learn_args: dict):
        self.model.train(
            train_dataloader=self.train_dataloader,
            val_dataloader=self.es_dataloader,
            model_args=model_args,
            learn_args=learn_args,
        )
    
    def evaluate(self):
        return self.model.evaluate(
            X=self.val_ds.X,
            y=self.val_ds.y,
            weights=self.val_ds.weights
        )

        
        
                        
if __name__ == "__main__":
    args = get_cli_args()
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    data_dir = args.data_dir if args.data_dir is not None else DATA_DIR    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    study_name = args.study_name if args.study_name is not None else \
        f'{args.model}_{args.n_seeds}seeds_{args.start_dt}-{args.end_dt}-{timestamp}'
        
    out_dir = args.out_dir if args.out_dir is not None else str(EXP_DIR / 'tuning' / 'tmp' / str(args.model) / study_name)
    storage = f'sqlite:///{out_dir}/optuna_study.db' if args.storage is None else args.storage
    
    logger = setup_logger(out_dir)
    logger.info(f'Tuning model: {args.model}')

    optimizer = NeuralTuner(
        model_type=args.model,
        start_dt=args.start_dt,
        end_dt=args.end_dt,
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
        logger=logger
    )
    optimizer.create_study()

    if args.train:
        optimizer.train_best_trial()
        save_path = f'{out_dir}/train/model'
        os.makedirs(save_path, exist_ok=True)
        optimizer.model.save(save_path)
    else:
        optimizer.run()
    
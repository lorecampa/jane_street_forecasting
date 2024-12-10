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
from prj.data.data_loader import DataLoader
from prj.hyperparameters_opt import SAMPLER
from prj.logger import setup_logger
from prj.tuner import Tuner
from prj.utils import str_to_dict_arg



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
        '--start_partition',
        type=int,
        default=5,
        help="start partition (included) "
    )

    parser.add_argument(
        '--end_partition',
        type=int,
        default=5,
        help="end partition (included) "
    )
    
    parser.add_argument(
        '--start_val_partition',
        type=int,
        default=6,
        help="starting val partition(included) "
    )
    
    parser.add_argument(
        '--end_val_partition',
        type=int,
        default=6,
        help="ending val partition(included) "
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


class MultiTuner(Tuner):
    def __init__(
        self,
        model_type: str,
        start_partition: int,
        end_partition: int,
        start_val_partition: int,
        end_val_partition: int,
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
            logger=logger
        )
        self.start_partition = start_partition
        self.end_partition = end_partition
        self.start_val_partition = start_val_partition
        self.end_val_partition = end_val_partition
        
        model_dict = TREE_NAME_MODEL_CLASS_DICT | NEURAL_NAME_MODEL_CLASS_DICT
        self.is_neural = model_type in NEURAL_NAME_MODEL_CLASS_DICT.keys()
        self.early_stopping = early_stopping
        
        self.model_class = model_dict[self.model_type]
        self.model = AgentsFactory.build_agent({'agent_type': self.model_type, 'seeds': self.seeds})
      
        data_args = DATA_ARGS_CONFIG[self.model_type]
        self.loader = DataLoader(data_dir=self.data_dir, **data_args)
        self.train_data = self.loader.load_partitions(self.start_partition, self.end_partition)
        self.es_data = None
        if self.is_neural and self.early_stopping:
            _X, _y, _w, _info = self.train_data
            split_point = int(0.8 * len(_X))
            self.train_data = (_X[:split_point], _y[:split_point], _w[:split_point], _info[:split_point])
            self.es_data = (_X[split_point:], _y[split_point:], _w[split_point:], _info[split_point:])
        
        self.val_data = self.loader.load_partitions(self.start_val_partition, self.end_val_partition)
    
        if self.is_neural:
            self.model_args = {'input_dim': self.train_data[0].shape[1:]}
            self.learn_args = {
                'validation_data': self.es_data[:-1],
                'epochs': 1,
                'early_stopping_rounds': 5,
                'scheduler_type': 'simple_decay'
            }
        else:
            self.model_args = {'verbose': self.verbose}
            self.learn_args = {}
            
    
    def train(self, model_args:dict, learn_args: dict):
        X, y, w, _ = self.train_data
            
        self.model.train(
            X, y, w,
            model_args=model_args,
            learn_args=learn_args,
        )
        gc.collect()
        
    def train_best_trial(self):
        best_trial = self.study.best_trial
        model_args = self.model_args.copy()
        model_args.update(self.custom_model_args)
        model_args.update(SAMPLER[self.model_type](best_trial, additional_args=self.sampler_args))
                
        learn_args = self.learn_args.copy()
        learn_args.update(self.custom_learn_args)
        
        self.train(model_args=model_args, learn_args=learn_args) 
        
        train_metrics = self.model.evaluate(*self.train_data[:-1])
        val_metrics = self.model.evaluate(*self.val_data[:-1])  
        return train_metrics, val_metrics
                        

if __name__ == "__main__":
    args = get_cli_args()
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    data_dir = args.data_dir if args.data_dir is not None else DATA_DIR    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    study_name = args.study_name if args.study_name is not None else \
        f'{args.model}_{args.n_seeds}seeds_{args.start_partition}_{args.end_partition}-{args.start_val_partition}_{args.end_val_partition}_{timestamp}'
        
    out_dir = args.out_dir if args.out_dir is not None else str(EXP_DIR / 'tuning' /str(args.model) / study_name)
    storage = f'sqlite:///{out_dir}/optuna_study.db' if args.storage is None else args.storage
    
    logger = setup_logger(out_dir)
    logger.info(f'Tuning model: {args.model}')

    optimizer = MultiTuner(
        model_type=args.model,
        start_partition=args.start_partition,
        end_partition=args.end_partition,
        start_val_partition=args.start_val_partition,
        end_val_partition=args.end_val_partition,
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
        logger=logger
    )
    optimizer.create_study()

    if args.train:
        train_res, val_res = optimizer.train_best_trial()
        logger.info(f"Train: {train_res}, Val: {val_res}")
        
        save_path = f'{out_dir}/train/model'
        os.makedirs(save_path, exist_ok=True)
        optimizer.model.save(save_path)
    else:
        optimizer.run()
    
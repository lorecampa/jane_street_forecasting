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
        '--val_ratio',
        type=float,
        default=0.15,
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
        val_ratio: float = 0.15,
        out_dir: str = '.',
        n_seeds: int = None,
        storage: str = None,
        use_gpu: bool = False,
        study_name: str = None,
        n_trials: int = 50,
        verbose: int = 0,
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
        
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.val_ratio = val_ratio
        
        model_dict = TREE_NAME_MODEL_CLASS_DICT
        
        self.model_class = model_dict[self.model_type]
        self.model = AgentsFactory.build_agent({'agent_type': self.model_type, 'seeds': self.seeds})
      
        data_args = {}
        data_args.update(self.custom_data_args)
        config = DataConfig(**data_args)
        self.loader = DataLoader(data_dir=data_dir, config=config)
        
        
        train_df, val_df = self.loader.load_train_and_val(self.start_dt, self.end_dt, self.val_ratio)
        self.train_data = self.loader._build_splits(train_df)
        self.val_data = self.loader._build_splits(val_df)

        self.model_args = {}
        if model_type == 'lgbm':
            self.model_args.update({'verbose': -1})
            
        self.learn_args = {}
            
    
    def train(self, model_args:dict, learn_args: dict):
        X, y, w, _ = self.train_data
            
        self.model.train(
            X, y, w,
            model_args=model_args,
            learn_args=learn_args,
        )

                        
if __name__ == "__main__":
    args = get_cli_args()
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    data_dir = args.data_dir if args.data_dir is not None else DATA_DIR    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    study_name = args.study_name if args.study_name is not None else \
        f'{args.model}_{args.n_seeds}seeds_{args.start_dt}_{args.end_dt}-{args.val_ratio}_{timestamp}'
        
    out_dir = args.out_dir if args.out_dir is not None else str(EXP_DIR / 'tuning' /str(args.model) / study_name)
    storage = f'sqlite:///{out_dir}/optuna_study.db' if args.storage is None else args.storage
    
    logger = setup_logger(out_dir)
    logger.info(f'Tuning model: {args.model}')

    optimizer = TreeTuner(
        model_type=args.model,
        start_dt=args.start_dt,
        end_dt=args.end_dt,
        val_ratio=args.val_ratio,
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
    
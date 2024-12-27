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
from prj.data.data_loader import DataConfig, DataLoader
from prj.hyperparameters_opt import SAMPLER
from prj.logger import setup_logger
from prj.tuner import Tuner
from prj.utils import str_to_dict_arg
import lightgbm as lgb


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
        '--train',
        action='store_true',
        default=False,
        help="Run only training, no optimization"
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
        '--gpu',
        action='store_true',
        default=False,
        help="Run only training with gpu"
    )


    return parser.parse_args()


class TunerLGBMBinary(Tuner):
    def __init__(
        self,
        model_type: str,
        data_dir: str = DATA_DIR,
        out_dir: str = '.',
        n_seeds: int = None,
        storage: str = None,
        use_gpu: bool = False,
        study_name: str = None,
        n_trials: int = 50,
        verbose: int = 0,
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
            logger=logger,
        )
                
        model_dict = TREE_NAME_MODEL_CLASS_DICT | NEURAL_NAME_MODEL_CLASS_DICT
        
        self.model_class = model_dict[self.model_type]
        self.model = AgentsFactory.build_agent({'agent_type': self.model_type, 'seeds': self.seeds})
      
        self.model_args = {'verbose': self.verbose}
        self.learn_args = {}
        max_bin = 305
        self.sampler_args = {'max_bin': max_bin, 'use_gpu': self.use_gpu}
        binary_path_train = '/home/lorecampa/projects/jane_street_forecasting/dataset/binary/lgbm_base_features/lgbm_maxbin_305_0_8.bin'
                
        print(f"Loading binary file: {binary_path_train}")

        self.train_data = lgb.Dataset(data=binary_path_train, params={
            'feature_pre_filter': False, 
            'device': 'gpu' if self.use_gpu else 'cpu'
        })
        
        data_args = {}
        config = DataConfig(**data_args)
        self.loader = DataLoader(data_dir=data_dir, config=config)
        val_ds = self.loader.load_with_partition(9, 9)
        self.val_data = self.loader._build_splits(val_ds)
        
   
    def train(self, model_args:dict, learn_args: dict):
        self.model.train_native(
            self.train_data,
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
        f'{args.model}_{args.n_seeds}seeds_max_bin_128_0_8-9_9_{timestamp}'
        
    out_dir = args.out_dir if args.out_dir is not None else str(EXP_DIR / 'tuning' / 'lgbm_binary' / study_name)
    
    storage = f'sqlite:///{out_dir}/optuna_study.db' if args.storage is None else args.storage
    logger = setup_logger(out_dir)
    logger.info(f'Tuning model: {args.model}')
    optimizer = TunerLGBMBinary(
        model_type=args.model,
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
        optimizer.train_best_trial()
        save_path = f'{out_dir}/train/model'
        os.makedirs(save_path, exist_ok=True)
        optimizer.model.save(save_path)
    else:
        optimizer.run()
    
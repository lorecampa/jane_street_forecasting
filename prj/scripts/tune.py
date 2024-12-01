from datetime import datetime
import argparse
import os
from prj.agents.AgentNeuralRegressor import NEURAL_NAME_MODEL_CLASS_DICT
from prj.agents.AgentTreeRegressor import TREE_NAME_MODEL_CLASS_DICT
from prj.agents.factory import AgentsFactory
from prj.config import DATA_DIR, EXP_DIR
from prj.data_loader import DataLoader
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
        custom_model_args: dict = {},
        custom_learn_args: dict = {},
    ):
        super().__init__(
            model_type=model_type,
            start_partition=start_partition,
            end_partition=end_partition,
            start_val_partition=start_val_partition,
            end_val_partition=end_val_partition,
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
        )
        
        model_dict = TREE_NAME_MODEL_CLASS_DICT | NEURAL_NAME_MODEL_CLASS_DICT
        self.is_neural = model_type in NEURAL_NAME_MODEL_CLASS_DICT.keys()
        
        self.model_class = model_dict[self.model_type]
        self.model = AgentsFactory.build_agent({'agent_type': self.model_type, 'seeds': self.seeds})
      
        if self.is_neural:
            data_args = {
                'ffill': True
            }
        else:
            data_args = {
                'ffill': False
            }
        self.data_loader = DataLoader(data_dir=self.data_dir, **data_args)
        self.train_data = self.data_loader.load_partitions(self.start_partition, self.end_partition)
        self.val_data = self.data_loader.load_partitions(self.start_val_partition, self.end_val_partition)
    
        if self.is_neural:
            self.model_args = {'input_dim': self.train_data[0].shape[1:]}
            self.learn_args = {
                'validation_data': self.val_data,
                'epochs': 20,
                'early_stopping_rounds': 5,
            }
        else:
            self.model_args = {'verbose': self.verbose}
            self.learn_args = {}
                        

if __name__ == "__main__":
    args = get_cli_args()
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    data_dir = args.data_dir if args.data_dir is not None else DATA_DIR
    print(f'Tuning model: {args.model}')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    study_name = args.study_name if args.study_name is not None else \
        f'{args.model}_{args.n_seeds}seeds_{args.start_partition}_{args.end_partition}-{args.start_val_partition}_{args.end_val_partition}_{timestamp}'
        
    out_dir = args.out_dir if args.out_dir is not None else str(EXP_DIR / 'tuning' / study_name)
    
    storage = f'sqlite:///{out_dir}/optuna_study.db' if args.storage is None else args.storage

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
        custom_learn_args=args.custom_learn_args
    )
    optimizer.create_study()

    if args.train:
        train_res, val_res = optimizer.train_best_trial()
        print(f"Train: {train_res}, Val: {val_res}")
        
        save_path = f'{out_dir}/train/model'
        os.makedirs(save_path, exist_ok=True)
        optimizer.model.save(save_path)
    else:
        optimizer.run()
    
from datetime import datetime
import gc
import numpy as np
import optuna
import argparse
import os
import optuna
import polars as pl
from prj.agents.AgentTreeRegressor import TREE_NAME_MODEL_CLASS_DICT, AgentTreeRegressor
from prj.agents.factory import AgentsFactory
from prj.config import DATA_DIR, GLOBAL_SEED
from prj.data_loader import DataLoader
from prj.hyperparameters_opt import SAMPLER
from prj.model.nn.neural import TabularNNModel
from prj.tuner import Tuner



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
        default="."
    )
    parser.add_argument(
        '--verbose',
        type=int,
        help='Enable verbose',
        default=0
    )

    return parser.parse_args()


class TreeTuner(Tuner):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.model_class = TREE_NAME_MODEL_CLASS_DICT[self.model_type]
        self.model: AgentTreeRegressor = AgentsFactory.build_agent({'agent_type': self.model_type, 'seeds': self.seeds})
      
      
        data_args = {
            'ffill': False
        }
        self.data_dir = data_dir
        self.data_loader = DataLoader(data_dir=self.data_dir, **data_args)
        self.train_data = self.data_loader.load_partitions(self.start_partition, self.end_partition)
        self.val_data = self.data_loader.load_partitions(self.start_val_partition, self.end_val_partition)
    
    
        
    def train(self, model_args: dict):
        X, y, w = self.train_data
        model_args.update({
            'verbose': self.verbose,
        })
        
        self.model.train(
            X, y, 
            sample_weight=w,
            model_args=model_args,
            learn_args={}
        )
        gc.collect()
                    

if __name__ == "__main__":
    args = get_cli_args()
    data_dir = args.data_dir if args.data_dir is not None else DATA_DIR
    print(f'Tuning model: {args.model}')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = f'{args.out_dir}_{timestamp}'
    storage = f'sqlite:///{out_dir}/optuna_study.db'

    optimizer = TreeTuner(
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
        n_trials=args.n_trials
    )
    optimizer.create_study()
    optimizer.run()
import argparse
from datetime import datetime
import os
import typing
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import numpy as np
from tqdm import tqdm
from xgboost import XGBRegressor
from prj.agents.AgentTreeRegressor import AgentTreeRegressor
from prj.agents.factory import AgentsFactory
from prj.config import DATA_DIR, EXP_DIR, GLOBAL_SEED, ROOT_DIR
import polars as pl

from prj.plots import plot_partition_heatmap
from prj.utils import check_for_inf, load_json, merge_dicts, save_dict_to_json, save_dict_to_pickle, set_random_seed

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model',
        type=str,
        default='lgbm'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=GLOBAL_SEED,
    )
    parser.add_argument(
        '--start_train_partition_id',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--end_train_partition_id',
        type=int,
        default=5
    )
    parser.add_argument(
        '--n_training_seeds',
        type=int,
        default=1
    )
    parser.add_argument(
        '--storage',
        type=str,
        default=None
    )
    parser.add_argument(
        '--load_model_dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Enable progress output',
        default=False
    )

    return parser.parse_args()

class RegressorTrainer:
    def __init__(self, 
            model_type: str, 
            load_path: typing.Optional[str] = None,
            n_seeds:int = 1,
            verbose: int = 0,
    ):      
        self.model_type = model_type
        self.n_seeds = n_seeds
        self.load_path = load_path
        self.verbose=verbose
        self.agent: AgentTreeRegressor = AgentsFactory.load_agent({
            'agent_type': self.model_type, 
            'n_seeds': self.n_seeds,
            'load_path': self.load_path
        })
            
        
    @staticmethod
    def prepare_dataset(partition_ids: list[int]):
        train_ds = pl.concat([
            pl.scan_parquet(DATA_DIR / 'train' / f'partition_id={i}' / 'part-0.parquet')
            for i in partition_ids
        ]).sort('date_id', 'time_id', 'symbol_id').collect()
        features = [col for col in train_ds.columns if col.startswith('feature_')]
        target_feature = 'responder_6'
        
        # Check for Inf
        #TODO: How should I handle inf values and NaN/Null values?
        # rows_with_inf, cols_with_inf = check_for_inf(train_ds)
        
        X = train_ds.select(features).cast(pl.Float32).to_numpy()
        y = train_ds[target_feature].cast(pl.Float32).to_numpy()
        w = train_ds['weight'].cast(pl.Float32).to_numpy()
        
        return X, y, w
        
    def train(self, partition_ids: list[int], model_args: dict = {}, save_path: typing.Optional[str] = None):
        X, y, _ = self.prepare_dataset(partition_ids)
        self.agent.train(X, y, model_args=model_args)
        print(f'Training completed')
        
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self.agent.save(save_path)
    
    def evaluate(self, partition: int, save_path: typing.Optional[str] = None):
        X, y, w = self.prepare_dataset([partition])
        result = self.agent.evaluate(X, y, weights=w)
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            save_dict_to_pickle(result, os.path.join(save_path, f'result_{partition}.pkl'))
        return result
    
def main():
    args = get_cli_args()
    seed = args.seed
    set_random_seed(seed)
    
    model_type = args.model.lower()
    start_partition_id = args.start_train_partition_id
    end_partition_id = args.end_train_partition_id
    train_partition_ids = list(range(start_partition_id, end_partition_id + 1))
    
    load_path = args.load_model_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EXP_DIR / f'train/{model_type}_{start_partition_id}-{end_partition_id}_{timestamp}' if args.out_dir is None else args.out_dir
    
    trainer = RegressorTrainer(
        model_type=model_type,
        load_path=load_path,
        n_seeds=args.n_training_seeds
    )
    
    if load_path is None and len(trainer.agent.agents) == 0:
        print(f'Training agents with {model_type} model')
        trainer.train(train_partition_ids, 
                      model_args=DEFAULT_TREE_PARAMS[model_type], 
                      save_path=os.path.join(out_dir, 'models')
        )
    
    # evaluations_partition_ids = list(range(train_partition_ids[-1] + 1, train_partition_ids[-1] + 1 + 3))
    range_evaluation = range(0, 10)
    evaluations = []
    for partition_id in tqdm(list(range_evaluation)):
        evaluation = trainer.evaluate(partition_id, save_path=os.path.join(out_dir, 'evaluations'))
        evaluations.append(evaluation)
    
    final_evaluation_dict = merge_dicts(evaluations)
    save_dict_to_pickle(final_evaluation_dict, os.path.join(out_dir, 'evaluations', 'result_aggregate.pkl'))
    
    # Plots
    first_partion_evaluation = list(range_evaluation)[0]
    training_partitions = [
        [(start_partition_id-first_partion_evaluation, end_partition_id-first_partion_evaluation)]
    ]
    x_tick_labels = [f'partition {i}' for i in range_evaluation]
    y_tick_labels = [model_type]
    
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    for metric in final_evaluation_dict.keys():
        plot_partition_heatmap(
            np.array([np.array(final_evaluation_dict[metric])]),
            training_partitions, 
            title=f'{model_type}-{metric.upper()} Heatmap', 
            xticklabels=x_tick_labels, 
            yticklabels=y_tick_labels, 
            save_path=os.path.join(plot_dir, f'{metric}_heatmap.png'),
            show_best_per_partition=False,
            decimal_places=4,
            invert_scale=metric not in ['r2_w'] # Invert scale for loss metrics
        )
    
    
DEFAULT_TREE_PARAMS = {
    'lgbm': {
        'objective': 'regression',
    },
    'xgb': {
        'objective': 'reg:squarederror',
    },
    'catboost': {
        'objective': 'RMSE',
        'verbose': 50,
    }
}
    
if __name__ == "__main__":
    main()
    
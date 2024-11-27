from datetime import datetime
import gc
import numpy as np
import optuna
import argparse
import os
import optuna
import polars as pl
from prj.agents.AgentNeuralRegressor import NEURAL_NAME_MODEL_CLASS_DICT, AgentNeuralRegressor
from prj.agents.factory import AgentsFactory
from prj.config import DATA_DIR, GLOBAL_SEED
from prj.data_loader import DataLoader
from prj.hyperparameters_opt import SAMPLER
from prj.model.nn.neural import TabularNNModel



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

class NeuralTuner:
    def __init__(
        self,
        model_type: str,
        start_partition: int,
        end_partition: int,
        start_val_partition: int,
        end_val_partition: int,
        out_dir: str = '.',
        n_seeds: int = None,
        storage: str = None,
        n_trials: int = 50,
        verbose: int = 0,
    ):
        self.model_type = model_type
        self.model_class = NEURAL_NAME_MODEL_CLASS_DICT[model_type]
        self.start_partition = start_partition
        self.end_partition = end_partition
        self.start_val_partition = start_val_partition
        self.end_val_partition = end_val_partition
        assert self.start_partition <= self.end_partition, "start_partition must be less than end_partition"
        assert self.start_val_partition <= self.end_val_partition, "start_val_partition must be less than end_val_partition"
        assert self.end_partition < self.start_val_partition or self.end_val_partition < self.start_partition, "No overlap between train and val partitions"
        
        self.n_seeds = n_seeds
        if self.n_seeds is not None:
            np.random.seed()
            self.seeds = [np.random.randint(2**32 - 1, dtype="int64").item() for i in range(self.n_seeds)]
        else:
            self.n_seeds = 1
            self.seeds = [GLOBAL_SEED]
            
        # Optuna
        self.storage = storage
        self.n_trials = n_trials
        
        self.out_dir = out_dir 
        
        self.verbose = verbose       
        self.study = None
        
        self._setup_directories()
        
        self.model: AgentNeuralRegressor = AgentsFactory.build_agent({'agent_type': self.model_type, 'seeds': self.seeds})
        
        data_args = {
            'ffill': True
        }
        self.data_loader = DataLoader(**data_args)
        self.train_data = self.data_loader.load_partitions(self.start_partition, self.end_partition)
        self.val_data = self.data_loader.load_partitions(self.start_val_partition, self.end_val_partition)
    
        
    def create_study(self):
        self.study = optuna.create_study(
            study_name=f'{self.model_class.__name__}_tuning_{self.n_seeds}seeds_{self.start_partition}_{self.end_partition}',
            direction="maximize", 
            storage=self.storage
        )
    def _setup_directories(self):
        self.optuna_dir = f'{out_dir}/optuna'
            
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.optuna_dir, exist_ok=True)

        
    def train(self, model_args: dict):
        X, y, w = self.train_data
        model_args.update({
            'input_dim': X.shape[1:],
        })
        
        self.model.train(
            X, y, w,
            model_args=model_args,
            validation_data=self.val_data,
            epochs=5,
            early_stopping_rounds=3
        )
        gc.collect()
                    
                    
    def optimize_hyperparameters(self, metric: str = 'r2_w'):
        def objective(trial):
            model_params = SAMPLER[self.model_type](trial, additional_args={})
            self.train(model_params)
            
            train_metrics = self.model.evaluate(*self.train_data)
            trial.set_user_attr("train_metrics", train_metrics)

            val_metrics = self.model.evaluate(*self.val_data)
            trial.set_user_attr("val_metrics", val_metrics)
            
            
            if trial.number > 1:
                self._plot_results(trial)
            
            return val_metrics[metric]
        
        print(f"Optimizing {self.model_class.__class__} hyperparameters")
        print(f'Using seeds: {self.seeds}')
        self.study.optimize(objective, n_trials=self.n_trials)
    
    def _plot_results(self, trial):
        plots = [
            ("ParamsOptHistory.png", optuna.visualization.plot_optimization_history(self.study)),
            ("ParamsImportance.png", optuna.visualization.plot_param_importances(self.study)),
            ("ParamsContour.png", optuna.visualization.plot_contour(self.study)),
            ("ParamsSlice.png", optuna.visualization.plot_slice(self.study))
        ]
        optuna_plot_dir = f"{self.optuna_dir}/plots"
        os.makedirs(optuna_plot_dir, exist_ok=True)
        for filename, fig in plots:
            fig.write_image(f"{optuna_plot_dir}/{filename}")
    
    def run(self):
        self.optimize_hyperparameters()

if __name__ == "__main__":
    args = get_cli_args()
    
    model_class = NEURAL_NAME_MODEL_CLASS_DICT[args.model]
    
    
    print(f'Tuning model: {model_class.__name__}')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = f'{args.out_dir}_{timestamp}'
    storage = f'sqlite:///{out_dir}/optuna_study.db'

    
    optimizer = NeuralTuner(
        model_type=args.model,
        start_partition=args.start_partition,
        end_partition=args.end_partition,
        start_val_partition=args.start_val_partition,
        end_val_partition=args.end_val_partition,
        out_dir=out_dir,
        n_seeds=args.n_seeds,
        verbose=args.verbose,
        storage=storage,
        n_trials=args.n_trials
    )
    
    
    optimizer.create_study()
    optimizer.run()
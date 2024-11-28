import gc
import os
import numpy as np
import optuna
from prj.config import DATA_DIR, GLOBAL_SEED
from prj.hyperparameters_opt import SAMPLER


class Tuner:
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
        n_trials: int = 50,
        verbose: int = 0,
        custom_args: dict = {}
    ):
        self.model_type = model_type
        self.data_dir = data_dir
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
        
        self.custom_args = custom_args
        # Optuna
        self.storage = storage
        self.n_trials = n_trials
        
        self.out_dir = out_dir 
        
        self.verbose = verbose       
        self.study = None
        
        self._setup_directories()
        
    
        
    def create_study(self):
        self.study = optuna.create_study(
            study_name=f'{self.model_class.__name__}_{self.n_seeds}seeds_{self.start_partition}_{self.end_partition}-{self.start_val_partition}_{self.end_val_partition}',
            direction="maximize", 
            storage=self.storage
        )
    def _setup_directories(self):
        self.optuna_dir = f'{self.out_dir}/optuna'
            
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.optuna_dir, exist_ok=True)

        
                            
    def optimize_hyperparameters(self, metric: str = 'r2_w'):
        def objective(trial):
            model_params = SAMPLER[self.model_type](trial, additional_args={}).copy()
            model_params.update(self.custom_args)
            
            self.train(model_params)
            
            train_metrics = self.model.evaluate(*self.train_data)
            trial.set_user_attr("train_metrics", train_metrics)

            val_metrics = self.model.evaluate(*self.val_data)
            trial.set_user_attr("val_metrics", val_metrics)
            
            
            if trial.number > 1:
                self._plot_results(trial)
            
            return val_metrics[metric]
        
        print(f"Optimizing {self.model_class.__name__} hyperparameters")
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
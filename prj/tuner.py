import gc
import os
import shutil
import numpy as np
import optuna
from prj.agents.AgentRegressor import AgentRegressor
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
        storage: str = None,
        study_name: str = None,
        n_seeds: int = None,
        n_trials: int = 50,
        verbose: int = 0,
        custom_model_args: dict = {},
        custom_learn_args: dict = {}
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
        
        if n_seeds is not None:
            np.random.seed()
            self.seeds = sorted([np.random.randint(2**32 - 1, dtype="int64").item() for i in range(n_seeds)])
        else:
            self.seeds = [GLOBAL_SEED]
        
        self.model: AgentRegressor = None
        self.custom_model_args = custom_model_args
        self.model_args = {}
        self.custom_learn_args = custom_learn_args
        self.learn_args = {}
        
        # Optuna
        self.storage = storage
        self.n_trials = n_trials
        self.study_name = study_name
        self.out_dir = out_dir 
        
        self.verbose = verbose       
        self.study = None
        
        self._setup_directories()
        
    
    def train(self, model_args:dict, learn_args: dict):
        X, y, w = self.train_data
        self.model.train(
            X, y, w,
            model_args=model_args,
            learn_args=learn_args
        )
        gc.collect()
        
    def create_study(self):
        if self.study_name is None:
            timestamp = self.out_dir.split('_')[-1]
            self.study_name = f'{self.model_class.__name__}_{len(self.seeds)}seeds_{self.start_partition}_{self.end_partition}-{self.start_val_partition}_{self.end_val_partition}_{timestamp}'
                 
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize", 
            storage=self.storage,
            load_if_exists=True
        )
        
        is_study_loaded = len(self.study.trials) >= 1
        if is_study_loaded:
            print(f"Study {self.study_name} loaded with {len(self.study.trials)} trials, loading seeds")
            self.seeds = sorted([int(seed) for seed in self.study.get_user_attrs('seeds')])
            # Updating model seeds
            self.model.set_seeds(self.seeds)
        else:
            self.study.set_user_attr('seeds', self.seeds)
            
            
    def _setup_directories(self):
        self.optuna_dir = f'{self.out_dir}/optuna'
            
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.optuna_dir, exist_ok=True)

          
    def optimize_hyperparameters(self, metric: str = 'r2_w'):
        def objective(trial):
            model_args: dict = SAMPLER[self.model_type](trial).copy()
            model_args.update(self.custom_model_args)
            model_args.update(self.model_args)
            model_args.update(self.custom_model_args)
            
            learn_args = self.learn_args.copy()
            learn_args.update(self.custom_learn_args)
                        
            self.train(model_args=model_args, learn_args=learn_args)
            
            train_metrics = self.model.evaluate(*self.train_data)
            trial.set_user_attr("train_metrics", str(train_metrics))

            val_metrics = self.model.evaluate(*self.val_data)
            trial.set_user_attr("val_metrics", str(val_metrics))
            
            
            if trial.number > 1:
                self._plot_results(trial)
            
            return val_metrics[metric]
        
        print(f"Optimizing {self.model_class.__name__} hyperparameters")
        print(f'Using seeds: {self.seeds}')
        self.study.optimize(objective, n_trials=self.n_trials, callbacks=[self._bootstrap_trial])
    
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
            
            
    
    def _bootstrap_trial(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state in [optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL]:
            return None
        
        if trial.number == study.best_trial.number:
            print(f'Best trial found: {trial.number}')
            
            best_dir_path = f'{self.out_dir}/best_trial'
            if os.path.exists(best_dir_path):
                shutil.rmtree(best_dir_path)
            
            best_dir_saved_model_path = f'{best_dir_path}/saved_model'
            os.makedirs(best_dir_saved_model_path, exist_ok=True)
            self.model.save(best_dir_saved_model_path)
    
    def run(self):
        self.optimize_hyperparameters()
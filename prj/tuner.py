import time
from logging import Logger
import os
import shutil
import numpy as np
import optuna
from prj.agents.AgentRegressor import AgentRegressor
from prj.config import DATA_DIR, GLOBAL_SEED
from prj.hyperparameters_opt import SAMPLER
from prj.logger import get_default_logger
from prj.utils import save_dict_to_json, save_dict_to_pickle

class Tuner:
    def __init__(
        self,
        model_type: str,
        data_dir: str = DATA_DIR,
        out_dir: str = '.',
        storage: str = None,
        study_name: str = None,
        n_seeds: int = None,
        n_trials: int = 50,
        verbose: int = 0,
        use_gpu: bool = False,
        custom_model_args: dict = {},
        custom_learn_args: dict = {},
        logger: Logger = None
    ):
        self.model_type = model_type
        self.data_dir = data_dir
        self.loader = None
        if logger is None:
            self.logger = get_default_logger()
        else:
            self.logger = logger
                
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
        
        self.use_gpu = use_gpu
        self.sampler_args = {'use_gpu': use_gpu}
        
        # Optuna
        self.storage = storage
        self.n_trials = n_trials
        self.study_name = study_name
        self.out_dir = out_dir 
        
        self.verbose = verbose       
        self.study = None
        
        self._setup_directories()
        
        
    def create_study(self):          
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize", 
            storage=self.storage,
            load_if_exists=True
        )
        
        is_study_loaded = len(self.study.trials) >= 1
        if is_study_loaded:
            self.logger.info(f"Study {self.study_name} loaded with {len(self.study.trials)} trials, loading seeds")
            self.seeds = sorted([int(seed) for seed in self.study.user_attrs['seeds']])
            # Updating model seeds
            self.model.set_seeds(self.seeds)
        else:
            self.study.set_user_attr('seeds', self.seeds)
    

    def train_best_trial(self):
        best_trial = self.study.best_trial
        model_args = self.model_args.copy()
        model_args.update(self.custom_model_args)
        model_args.update(SAMPLER[self.model_type](best_trial, additional_args=self.sampler_args))
                
        learn_args = self.learn_args.copy()
        learn_args.update(self.custom_learn_args)
        
        self.train(model_args=model_args, learn_args=learn_args) 
        
    def _setup_directories(self):
        self.optuna_dir = f'{self.out_dir}/optuna'
            
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.optuna_dir, exist_ok=True)

    def evaluate(self):
        return self.model.evaluate(*self.val_data[:-1])
    
    def optimize_hyperparameters(self, metric: str = 'r2_w'):
        def objective(trial: optuna.Trial):
            start_time = time.time()
            sampler_args = self.sampler_args.copy()
            model_args = self.model_args.copy()
            model_args.update(self.custom_model_args)
            model_args.update(SAMPLER[self.model_type](trial, additional_args=sampler_args))
            
            learn_args = self.learn_args.copy()
            learn_args.update(self.custom_learn_args)
                        
            self.train(model_args=model_args, learn_args=learn_args)
                        
            val_metrics = self.evaluate()
            trial.set_user_attr("val_metrics", str(val_metrics))
            
            if trial.number % 5 == 0 or trial.number == self.n_trials:
                self._plot_optuna_results(trial)
            
            self.logger.info(f"Trial {trial.number} finished in {(time.time() - start_time)/60:.2f} minutes")
            return val_metrics[metric]
        
        self.logger.info(f"Optimizing {self.model_class.__name__} hyperparameters")
        self.logger.info(f'Using seeds: {self.seeds}')
        self.study.optimize(objective, n_trials=self.n_trials, callbacks=[self._bootstrap_trial])
    
    def _plot_optuna_results(self, trial):
        plots = [
            ("ParamsOptHistory.png", optuna.visualization.plot_optimization_history),
            ("ParamsImportance.png", optuna.visualization.plot_param_importances),
            ("ParamsContour.png", optuna.visualization.plot_contour),
            ("ParamsSlice.png", optuna.visualization.plot_slice)
        ]
        optuna_plot_dir = f"{self.optuna_dir}/plots"
        os.makedirs(optuna_plot_dir, exist_ok=True)
        for filename, fn in plots:
            try:
                fig = fn(self.study)
                fig.write_image(f"{optuna_plot_dir}/{filename}")
            except Exception as e:
                self.logger.error(f"Error while plotting {filename}: {e}")
                
    def _bootstrap_trial(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state in [optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL]:
            return None
        
        if trial.number == study.best_trial.number:
            self.logger.info(f'Best trial found: {trial.number}')
            
            best_dir_path = f'{self.out_dir}/best_trial'
            if os.path.exists(best_dir_path):
                shutil.rmtree(best_dir_path)
                
            best_dir_saved_model_path = f'{best_dir_path}/saved_model'
            os.makedirs(best_dir_saved_model_path, exist_ok=True)
            self.model.save(best_dir_saved_model_path)
                
            os.makedirs(best_dir_path, exist_ok=True)
            best_params = SAMPLER[self.model_type](trial, additional_args=self.sampler_args)
            save_dict_to_json(best_params, f'{best_dir_path}/best_params.json')
            
            params = dict(
                model_args=self.model_args,
                custom_model_args=self.custom_model_args,
                learn_args=self.learn_args,
                custom_learn_args=self.custom_learn_args,
                sampler_args=self.sampler_args,
                seeds=self.seeds,
                best_params=best_params
            )
            if self.loader is not None:
                params.update(**self.loader.get_info())
            
            save_dict_to_pickle(params, f'{best_dir_path}/params.pkl')
            
            plot_dir = f'{best_dir_path}/plots'
            os.makedirs(plot_dir, exist_ok=True)
            self.model.plot_stats(save_path=plot_dir)
            
    
    def run(self):
        self.optimize_hyperparameters()
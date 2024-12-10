from datetime import datetime
import argparse
import gc
from logging import Logger
import os
import time
import optuna
from tqdm import tqdm
from prj.agents.AgentNeuralRegressor import NEURAL_NAME_MODEL_CLASS_DICT
from prj.agents.AgentTreeRegressor import TREE_NAME_MODEL_CLASS_DICT
from prj.agents.factory import AgentsFactory
from prj.config import DATA_DIR, EXP_DIR
from prj.data import DATA_ARGS_CONFIG
from prj.data.data_loader import DataLoader
from prj.hyperparameters_opt import SAMPLER
from prj.logger import setup_logger
from prj.metrics import absolute_weighted_error_loss_fn, weighted_mae, weighted_rmse, weighted_r2, weighted_mse, squared_weighted_error_loss_fn, log_cosh_weighted_loss_fn
from prj.oamp.oamp import OAMP
from prj.oamp.oamp_config import ConfigOAMP
from prj.tuner import Tuner
from prj.utils import str_to_dict_arg
import numpy as np


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str,
        default=None
    )
    parser.add_argument(
        '--start_partition',
        type=int,
        default=9,
        help="start partition (included) "
    )

    parser.add_argument(
        '--end_partition',
        type=int,
        default=9,
        help="end partition (included) "
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help="number of iterations of optuna"
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
    

    return parser.parse_args()


class TunerOamp(Tuner):
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        start_partition: int = 9,
        end_partition: int = 9,
        out_dir: str = '.',
        storage: str = None,
        study_name: str = None,
        n_trials: int = 50,
        verbose: int = 0,
        custom_model_args: dict = {},
        custom_learn_args: dict = {},
        logger: Logger = None,
    ):
        super().__init__(
            model_type='oamp',
            data_dir=data_dir,
            out_dir=out_dir,
            storage=storage,
            study_name=study_name,
            n_trials=n_trials,
            verbose=verbose,
            custom_model_args=custom_model_args,
            custom_learn_args=custom_learn_args,
            logger=logger,
        )
        self.start_partition = start_partition
        self.end_partition = end_partition
        
        agent_base_dir = EXP_DIR / 'saved'
        self.agents_dict = [
            {'agent_type': 'lgbm', 'load_path': os.path.join(agent_base_dir, "LGBMRegressor_1seeds_4_5-6_6_134802", "best_trial", "saved_model")},
            {'agent_type': 'lgbm', 'load_path': os.path.join(agent_base_dir, "LGBMRegressor_1seeds_5_6-7_7_134811", "best_trial", "saved_model")},
            {'agent_type': 'lgbm', 'load_path': os.path.join(agent_base_dir, "LGBMRegressor_1seeds_6_7-8_8_134900", "best_trial", "saved_model")},
        ]
        self.agents = [AgentsFactory.load_agent(agent_dict) for agent_dict in self.agents_dict]
        
        
        data_args = DATA_ARGS_CONFIG[self.model_type]
        self.loader = DataLoader(data_dir=self.data_dir, data_args=data_args)
        X, self.y, self.w, info = self.loader.load_partitions(self.start_partition, self.end_partition)
        
        # f = 100000
        # X = X[:f]
        # self.y = self.y[:f]
        # self.w = self.w[:f]
        # info = info[:f]
        
        self.dates = info[:, 0]
        self.times = info[:, 1]
        self.symbols = info[:, 2]
        self.agent_predictions = np.concatenate([agent.predict(X).reshape(-1, 1) for agent in tqdm(self.agents)], axis=1)
        
        self.losses_dict = {
            'mae': absolute_weighted_error_loss_fn,
            'mse': squared_weighted_error_loss_fn,
            'log_cosh': log_cosh_weighted_loss_fn
        }
        
        del X
        gc.collect()
        
        self.model_args = {}
        self.learn_args = {}
        self.sampler_args = {}
        
        self.model: OAMP = None
            
    def train(self, model_args:dict, learn_args: dict):
        loss_fn = model_args.get('loss_function', None)
        del model_args['loss_function']
        if loss_fn not in self.losses_dict:
            raise ValueError(f"Loss function {loss_fn} not found in agent losses")

        agent_losses = self.losses_dict[loss_fn](y_true=self.y, y_pred=self.agent_predictions, w=self.w)  
        print(agent_losses.shape)
           
        config = ConfigOAMP(model_args)
        self.model = OAMP(agents_count=len(self.agents), args=config)
        
        preds = []
        last_day = 0
        for i in tqdm(range(self.agent_predictions.shape[0])):
            is_new_day = i > 0 and self.dates[i] != self.dates[i-1]
            if is_new_day:
                # print(f'New day {self.dates[i]}, doing steps of previous day')
                for j in range(last_day, i):
                    is_new_group = j > last_day and self.times[j] != self.times[j-1]
                    self.model.step(agent_losses[j], is_new_group=is_new_group)
                last_day = i
                
            preds.append(self.model.compute_prediction(self.agent_predictions[i]))
                    
        return np.array(preds)
                
    
    @staticmethod
    def metrics(y_true, y_pred, weights):
        return {
            'r2_w': weighted_r2(y_true, y_pred, weights=weights),
            'mae_w': weighted_mae(y_true, y_pred, weights=weights),
            'mse_w': weighted_mse(y_true, y_pred, weights=weights),
            'rmse_w': weighted_rmse(y_true, y_pred, weights=weights),
        }
    
    def optimize_hyperparameters(self, metric: str = 'r2_w'):
        def objective(trial):
            start_time = time.time()
            sampler_args = self.sampler_args.copy()
            model_args = self.model_args.copy()
            model_args.update(self.custom_model_args)
            model_args.update(SAMPLER[self.model_type](trial, additional_args=sampler_args))
            
            learn_args = self.learn_args.copy()
            learn_args.update(self.custom_learn_args)
                        
            y_hat = self.train(model_args=model_args, learn_args=learn_args)
            val_metrics = TunerOamp.metrics(y_true=self.y, y_pred=y_hat, weights=self.w)                        
            trial.set_user_attr("val_metrics", str(val_metrics))
            
            if trial.number % 5 == 0:
                self._plot_optuna_results(trial)
            
            self.logger.info(f"Trial {trial.number} finished in {(time.time() - start_time)/60:.2f} minutes")
            return val_metrics[metric]
        
        self.logger.info(f"Optimizing {self.model.__class__} hyperparameters")
        self.study.optimize(objective, n_trials=self.n_trials, callbacks=[self._bootstrap_trial])
        
                    
if __name__ == "__main__":
    args = get_cli_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    data_dir = args.data_dir if args.data_dir is not None else DATA_DIR
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    study_name = args.study_name if args.study_name is not None else \
        f'oamp_{args.start_partition}_{args.end_partition}_{timestamp}'
        
    out_dir = args.out_dir if args.out_dir is not None else str(EXP_DIR / 'tuning' / 'oamp' / study_name)
    
    storage = f'sqlite:///{out_dir}/optuna_study.db' if args.storage is None else args.storage
    logger = setup_logger(out_dir)
    
    
    optimizer = TunerOamp(
        data_dir=data_dir,
        start_partition=args.start_partition,
        end_partition=args.end_partition,
        out_dir=out_dir,
        verbose=args.verbose,
        storage=storage,
        study_name=study_name,
        n_trials=args.n_trials,
        custom_model_args=args.custom_model_args,
        custom_learn_args=args.custom_learn_args,
        logger=logger
    )
    optimizer.create_study()
    optimizer.run()
    
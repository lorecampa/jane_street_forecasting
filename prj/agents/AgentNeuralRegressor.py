from abc import abstractmethod
import gc
import os
import typing
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, log_evaluation
import numpy as np
from tqdm import tqdm
from xgboost import XGBRegressor
from prj.agents.AgentRegressor import AgentRegressor
import joblib
import polars as pl
from prj.config import DATA_DIR
from prj.model.nn.cnn_resnet import CnnResnet
from prj.model.nn.mlp import Mlp
from prj.model.nn.neural import TabularNNModel
from keras import optimizers as tfko
from keras import metrics as tfkm
from keras import callbacks as tfkc
import keras as tfk
from prj.model.nn.losses import WeightedZeroMeanR2Loss
from prj.model.nn.rnn import Rnn
from prj.model.nn.scheduler import get_simple_decay_scheduler
from prj.model.nn.tcn import Tcn
from prj.utils import set_random_seed 

NEURAL_NAME_MODEL_CLASS_DICT = {
    'mlp': Mlp,
    'tcn': Tcn,
    'rnn': Rnn,
    'cnn_resnet': CnnResnet,
}

class AgentNeuralRegressor(AgentRegressor):
    def __init__(
        self,
        agent_type: str,
        seeds: typing.Optional[list[int]] = None,
        n_seeds: int = 1,
    ):
        super().__init__(agent_type, seeds=seeds, n_seeds=n_seeds)
        
        self.agent_class: TabularNNModel = NEURAL_NAME_MODEL_CLASS_DICT[self.agent_type]
        self.agents: list[TabularNNModel] = []
        

    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        sample_weight: typing.Optional[np.ndarray] = None, 
        model_args: dict = {},
        learn_args: dict = {},
    ):        
        self.agents = []
        for seed in tqdm(self.seeds):
            curr_model_args = model_args.copy()
            
            if 'learning_rate' not in curr_model_args:
                curr_model_args['learning_rate'] = 5e-4
                warnings.warn(f"Learning rate not provided. Using default value {curr_model_args['learning_rate']}")
            learning_rate = curr_model_args.pop('learning_rate')
            
            use_scheduler = curr_model_args.pop('use_scheduler', False)
            scheduling_rate = curr_model_args.pop('scheduling_rate', None)
            if use_scheduler and (scheduling_rate is None):
                scheduling_rate = 0.005
                warnings.warn(f"Scheduling rate not specified, using default {scheduling_rate}")
            
            set_random_seed(seed)
            curr_agent: TabularNNModel = self.agent_class(**curr_model_args, random_seed=seed)
            
            optimizer = tfko.Adam(learning_rate=learning_rate)
            # loss = WeightedZeroMeanR2Loss()
            loss = tfk.losses.MeanSquaredError()
            metrics = [tfkm.R2Score(), tfkm.MeanSquaredError()]
            
            
            lr_scheduler = None
            scheduler_type = learn_args.get('scheduler_type', 'simple_decay')
            if use_scheduler:
                if scheduler_type == 'simple_decay':
                    lr_scheduler = get_simple_decay_scheduler(scheduling_rate, start_epoch=5)
                elif scheduler_type == 'reduce_lr_on_plateau':
                    lr_scheduler = tfkc.ReduceLROnPlateau(
                        monitor='val_loss',
                        patience=5,
                        verbose=1
                    )
                else:
                    raise ValueError(f"Scheduler type {scheduler_type} not recognized")
            
            curr_agent.fit(
                X, 
                y,
                sample_weight=sample_weight,
                validation_data=learn_args.get('validation_data', None),
                loss=loss,
                optimizer=optimizer,
                metrics=metrics,
                lr_scheduler=lr_scheduler,
                epochs=learn_args['epochs'],
                early_stopping_rounds=learn_args['early_stopping_rounds'],
            )
                
            self.agents.append(curr_agent)
                    
        return self.agents
            
    
    def save(self, path: str):
        for i, seed in enumerate(self.seeds):
            seed_path = os.path.join(path, f'seed_{seed}')
            self.agents[i].save(seed_path)
    
    def load(self, path: typing.Optional[str]):
        if path is None:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")
        
        print(f'Loading models, overwriting seeds: {self.seeds}')
        seeds_dir = sorted([f for f in os.listdir(path) if f.startswith('seed_')], key=lambda x: int(x.split('_')[-1]))
        self.seeds = [int(seed_dir.split('_')[-1]) for seed_dir in seeds_dir]
        
        self.agents = []
        for seed in self.seeds:
            seed_path = os.path.join(path, f'seed_{seed}')
            self.agents.append(TabularNNModel.load(seed_path))
        

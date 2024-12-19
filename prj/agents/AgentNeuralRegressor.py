import os
import typing
import warnings
import numpy as np
from tqdm import tqdm
from prj.agents.AgentRegressor import AgentRegressor
from torch.utils.data import DataLoader
from prj.model.keras.mlp import Mlp
from prj.model.keras.neural import TabularNNModel
import torch.nn as nn
import keras
from keras import optimizers as tfko
from keras import callbacks as tfkc
from keras import metrics as tfkm
from prj.model.keras.scheduler import get_simple_decay_scheduler
from prj.utils import set_random_seed

NEURAL_NAME_MODEL_CLASS_DICT = {
    'mlp': Mlp,
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
            curr_learn_args = learn_args.copy()
            
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
            loss = keras.losses.MeanSquaredError()
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
                loss=loss,
                optimizer=optimizer,
                metrics=metrics,
                lr_scheduler=lr_scheduler,
                **curr_learn_args
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
        

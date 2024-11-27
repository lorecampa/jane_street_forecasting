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
from prj.model.nn.mlp import MLP
from prj.model.nn.neural import TabularNNModel
from keras import optimizers as tfko
from keras import metrics as tfkm
from prj.model.nn.losses import WeightedZeroMeanR2Loss
from prj.utils import set_random_seed 

NEURAL_NAME_MODEL_CLASS_DICT = {
    'mlp': MLP,
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
        validation_data: typing.Optional[typing.Tuple[np.ndarray, np.ndarray, np.ndarray]] = None, 
        epochs: int = 10, 
        early_stopping_rounds:int=5
    ):        
        self.agents = []
        for seed in tqdm(self.seeds):
            curr_model_args = model_args.copy()
            
            if 'learning_rate' not in curr_model_args:
                curr_model_args['learning_rate'] = 1e-3
                warnings.warn("Learning rate not provided. Using default value 1e-3")
            learning_rate = curr_model_args.pop('learning_rate')
            
            curr_agent: TabularNNModel = self.agent_class(**model_args, random_seed=seed)
            set_random_seed(seed) #TODO: set cuda determinism
            
            optimizer = tfko.Adam(learning_rate=learning_rate)
            loss = WeightedZeroMeanR2Loss()
            metrics = [tfkm.R2Score(), tfkm.MeanSquaredError()]
            curr_agent.fit(
                X, 
                y,
                sample_weight=sample_weight,
                validation_data=validation_data,
                loss=loss,
                optimizer=optimizer,
                metrics=metrics,
                epochs=epochs,
                early_stopping_rounds=early_stopping_rounds,
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
        

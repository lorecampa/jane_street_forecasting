from abc import abstractmethod
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
from prj.utils import set_random_seed

TREE_NAME_MODEL_CLASS_DICT = {
    'lgbm': LGBMRegressor,
    'xgb': XGBRegressor,
    'catboost': CatBoostRegressor,
}


class AgentTreeRegressor(AgentRegressor):
    def __init__(
        self,
        agent_type: str,
        seeds: typing.Optional[list[int]] = None,
        n_seeds: int = 1,
    ):
        super().__init__(agent_type, seeds=seeds, n_seeds=n_seeds)
        
        self.agent_class: typing.Union[LGBMRegressor, CatBoostRegressor, XGBRegressor] = TREE_NAME_MODEL_CLASS_DICT[self.agent_type]
        self.agents: list[typing.Union[LGBMRegressor, CatBoostRegressor, XGBRegressor]] = []
        
    
    
    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, model_args: dict = {}, learn_args: dict = {}):
        self.agents = []
        for seed in tqdm(self.seeds):
            set_random_seed(seed)
            callbacks = []
            if self.agent_type == 'lgbm':
                callbacks = [log_evaluation(period=20)]
                curr_agent = self.agent_class(**model_args, random_state=seed)    
                curr_agent.fit(X, y, callbacks=callbacks, sample_weight=sample_weight, **learn_args)
            elif self.agent_type == 'xgb':
                curr_agent = self.agent_class(**model_args, random_state=seed)    
                curr_agent.fit(X, y, sample_weight=sample_weight, **learn_args)
            elif self.agent_type == 'catboost':
                curr_agent = self.agent_class(**model_args, random_state=seed)    
                curr_agent.fit(X, y, sample_weight=sample_weight, **learn_args)
                
            self.agents.append(curr_agent)
            
        return self.agents
    
    def save(self, path: str):
        for i, seed in enumerate(self.seeds):
            seed_path = os.path.join(path, f'seed_{seed}')
            os.makedirs(seed_path, exist_ok=True)
            if self.agent_type in ['lgbm', 'xgb', 'catboost']:
                joblib.dump(self.agents[i], os.path.join(seed_path, 'model.joblib'))
                

    def load(self, path: typing.Optional[str]):
        if path is None:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")
        
        seeds_dir = sorted([f for f in os.listdir(path) if f.startswith('seed_')], key=lambda x: int(x.split('_')[-1]))
        self.seeds = [int(seed_dir.split('_')[-1]) for seed_dir in seeds_dir]
        print(f'Loading models, overwriting seeds: {self.seeds}')
        self.agents = []
        for seed in self.seeds:
            seed_path = os.path.join(path, f'seed_{seed}')
            if self.agent_type in ['lgbm', 'xgb', 'catboost']:
                self.agents.append(joblib.load(os.path.join(seed_path, 'model.joblib')))
        

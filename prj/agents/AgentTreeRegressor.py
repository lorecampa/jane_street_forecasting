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

TREE_NAME_MODEL_CLASS_DICT = {
    'lgbm': LGBMRegressor,
    'xgb': XGBRegressor,
    'catboost': CatBoostRegressor,
}


class AgentTreeRegressor(AgentRegressor):
    def __init__(
        self,
        agent_type: str,
        n_seeds: int = 1,
    ):
        self.agent_type = agent_type
        self.agent_class: typing.Union[LGBMRegressor, CatBoostRegressor, XGBRegressor] = TREE_NAME_MODEL_CLASS_DICT[agent_type]
        self.agents = []
        self.n_seeds = n_seeds
        np.random.seed()
        self.seeds = sorted([np.random.randint(2**32 - 1, dtype="int64").item() for i in range(self.n_seeds)])
        
    
    
    def train(self, X: np.ndarray, y: np.ndarray, model_args: dict = {}):
        if len(self.agents) > 0:
            warnings.warn("Agent is already trained. Retraining...")
        self.agents = []
        for seed in tqdm(self.seeds):
            callbacks = []
            if self.agent_type == 'lgbm':
                callbacks = [log_evaluation(period=20)]
                curr_agent = self.agent_class(**model_args, random_state=seed)    
                curr_agent.fit(X, y, callbacks=callbacks)
            elif self.agent_type == 'xgb':
                curr_agent = self.agent_class(**model_args, random_state=seed)    
                curr_agent.fit(X, y)
            elif self.agent_type == 'catboost':
                curr_agent = self.agent_class(**model_args, random_state=seed)    
                curr_agent.fit(X, y)
                
            self.agents.append(curr_agent)
            
        return self.agents

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([agent.predict(X) for agent in self.agents], axis=0)
    
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
        

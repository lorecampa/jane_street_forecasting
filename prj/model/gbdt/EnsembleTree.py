from lightgbm import LGBMRegressor
import numpy as np
from prj.agents.factory import AgentsFactory
import polars as pl

class EnsembleTree:
    def __init__(self, agent_dicts: list[dict], model_args: dict = {}):
        self.agent_dicts = agent_dicts
        self.agent_dict = {'agent_type': 'lgbm', 'load_path': "/kaggle/input/janestreetmodels/model/model"}
        self.models = []
        
        
        self.model_args = model_args
        self.model_class = LGBMRegressor
        self.model = None
        
        self._load_models()
        self.model = self.model_class(**model_args)
        
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, **kwargs):
        self.model.fit(X, y, sample_weight=sample_weight, **kwargs)
        
    def _load_models(self):
        self.models = []
        for agent_dict in self.agent_dicts:
            agent = AgentsFactory.load_agent(agent_dict)
            self.models.append(agent)
    
    
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([model.predict(X) for model in self.models])
        
        df = pl.DataFrame(X).with_columns(
            [pl.Series(f'pred_{i}', values=pred, dtype=pl.Float32) for i, pred in enumerate(preds)]
        )
        return df.to_numpy()
    
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess(X)
        return self.model.predict(X)
    
    
    
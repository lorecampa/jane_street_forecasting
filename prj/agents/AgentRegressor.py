from abc import abstractmethod
import typing
from prj.metrics import weighted_mae, weighted_mse, weighted_r2, weighted_rmse
import numpy as np
from prj.agents.base import AgentBase
from prj.utils import interquartile_mean

class AgentRegressor(AgentBase):
    
    def __init__(
        self, 
        agent_type: str,
        seeds: typing.Optional[list[int]] = None,
        n_seeds: int = 1
    ):
        self.agent_type = agent_type
        self.agents: list[any] = []
        if seeds is not None:
            self.seeds = seeds
            self.n_seeds = len(seeds)
        else:
            self.n_seeds = n_seeds
            np.random.seed()
            self.seeds = sorted([np.random.randint(2**32 - 1, dtype="int64").item() for _ in range(self.n_seeds)])
            
    def set_seeds(self, seeds: list[int]) -> None:
        self.seeds = seeds
        self.n_seeds = len(seeds)
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    # def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
    #     aggregate = kwargs.get('aggregate', 'mean')
    #     preds = np.array([agent.predict(X).clip(-5, 5) for agent in self.agents])
    #     if aggregate == 'mean':
    #         return np.mean(preds, axis=0)
    #     elif aggregate == 'median':
    #         return np.median(preds, axis=0)
    #     elif aggregate == 'iqm':
    #         return interquartile_mean(preds, qmin=0.25, qmax=0.75)
    #     elif aggregate == None:
    #         return np.array(preds)
    #     else:
    #         raise ValueError(f"Unknown aggregation method: {aggregate}")
        
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        aggregate = kwargs.get('aggregate', 'mean')
        batch_size = kwargs.get('batch_size', X.shape[0])
        if batch_size is None:
            batch_size = X.shape[0]
        preds = []

        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = start_idx + batch_size
            X_batch = X[start_idx:end_idx]
            batch_preds = np.array([agent.predict(X_batch).clip(-5, 5) for agent in self.agents])
            preds.append(batch_preds)

        preds = np.concatenate(preds, axis=1)

        if aggregate == 'mean':
            return np.mean(preds, axis=0)
        elif aggregate == 'median':
            return np.median(preds, axis=0)
        elif aggregate == 'iqm':
            return interquartile_mean(preds, qmin=0.25, qmax=0.75)
        elif aggregate is None:
            return preds
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None, batch_size: float = None) -> float:
        y_pred = self.predict(X, batch_size=batch_size)
        return {
            'r2_w': weighted_r2(y, y_pred, weights=weights),
            'mae_w': weighted_mae(y, y_pred, weights=weights),
            'mse_w': weighted_mse(y, y_pred, weights=weights),
            'rmse_w': weighted_rmse(y, y_pred, weights=weights),
        }
        
    def plot_stats(self, save_path: typing.Optional[str] = None) -> None:
        pass
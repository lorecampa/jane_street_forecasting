from abc import abstractmethod
from prj.metrics import weighted_mae, weighted_mse, weighted_r2, weighted_rmse
import numpy as np
from prj.agents.base import AgentBase

class AgentRegressor(AgentBase):
    @abstractmethod
    def load(self, path: str) -> None:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> float:
        y_pred = self.predict(X)
        return {
            'r2_w': weighted_r2(y, y_pred, weights=weights),
            'mae_w': weighted_mae(y, y_pred, weights=weights),
            'mse_w': weighted_mse(y, y_pred, weights=weights),
            'rmse_w': weighted_rmse(y, y_pred, weights=weights),
        }
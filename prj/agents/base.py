from abc import ABC, abstractmethod
import numpy as np

class AgentBase(ABC):
    @abstractmethod
    def predict(self, state: np.ndarray) -> float:
        pass
    def __call__(self, state):
        return self.predict(state)
import gc
import os
import typing
import matplotlib.pyplot as plt
import numpy as np
import joblib
from collections import deque
from prj.oamp.oamp_config import ConfigOAMP
from prj.oamp.oamp_utils import (
    get_m,
    get_p,
    get_r,
    upd_n,
    upd_w,
)

class OAMP:
    def __init__(
        self,
        agents_count: int,
        args: ConfigOAMP,
        agent_labels: typing.Optional[list[str]] = None,
    ):
        assert args.agents_weights_upd_freq > 0, "Agents' weights update frequency should be greater than 0"
        assert args.loss_fn_window > 0, "Loss function window should be greater than 0"
        assert agent_labels is None or len(agent_labels) == agents_count, "Number of agent labels should be equal to the number of agents"
        # Initializing agents
        self.agents_count = agents_count
        self.agents_losses = self.init_agents_losses(args.loss_fn_window)
        self.agents_weights_upd_freq = args.agents_weights_upd_freq # Right now it is measured in groups
        self.agg_type = args.agg_type
        self.agent_labels = agent_labels
        
        # Initializing OAMP
        self.l_tm1 = np.zeros(agents_count)
        self.n_tm1 = np.ones(agents_count) * 0.25
        self.w_tm1 = np.ones(agents_count) / agents_count
        self.p_tm1 = np.ones(agents_count) / agents_count
        self.cum_err = np.zeros(agents_count)
        
        # Group params
        self.group_t = 0
        
        # Initializing OAMP stats
        self.stats = {
            "losses": [],
            "weights": [],
        }
        
    def init_agents_losses(
        self,
        loss_fn_window: int,
    ):
        return deque(maxlen=loss_fn_window)
    

    def step(
        self,
        agents_losses: np.ndarray,
    ):
        # Updating agents' losses
        if agents_losses.ndim == 1:
            agents_losses = agents_losses.reshape(1, -1)
        
        for agents_loss in agents_losses:
            self.agents_losses.append(agents_loss)
        
        self.group_t += 1
        
        # Updating agents' weights
        if self.group_t > 0 and self.group_t == self.agents_weights_upd_freq:
            self.update_agents_weights()
            self.group_t = 0
            
        
    def update_agents_weights(
        self,
    ):
        # Computing agents' losses
        l_t = self.compute_agents_losses()
        # Computing agents' regrets estimates
        m_t = get_m(
            self.l_tm1,
            self.n_tm1,
            self.w_tm1,
            self.agents_count,
        )
        # Computing agents' selection probabilites
        p_t = get_p(m_t, self.w_tm1, self.n_tm1)
        # Computing agents' regrets
        r_t = get_r(l_t, p_t)
        # Computing agents' regrets estimatation error
        self.cum_err += (r_t - m_t) ** 2
        # Updating agents' learning rates
        n_t = upd_n(self.cum_err, self.agents_count)
        # Updating agents' weights
        w_t = upd_w(
            self.w_tm1,
            self.n_tm1,
            n_t,
            r_t,
            m_t,
            self.agents_count,
        )
        self.l_tm1 = l_t
        self.n_tm1 = n_t
        self.w_tm1 = w_t
        self.p_tm1 = p_t
        self.stats["losses"].append(l_t)
        self.stats["weights"].append(p_t)
        return p_t

    def compute_agents_losses(
        self,
    ) -> np.ndarray:
        # Computing agents' losses
        agents_losses: np.ndarray = np.sum(self.agents_losses, axis=0)
        # Normalizing agents' losses
        agents_losses_min = agents_losses.min()
        agents_losses_max = agents_losses.max()
        if agents_losses_min != agents_losses_max:
            agents_losses = (agents_losses - agents_losses_min) / (
                agents_losses_max - agents_losses_min
            )
            
        return agents_losses

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, 'class.joblib'))
    
    @staticmethod
    def load(path: str) -> 'OAMP':
        return joblib.load(os.path.join(path, 'class.joblib'))

        
    def compute_prediction(
        self,
        agent_predictions: np.ndarray,
    ) -> np.ndarray:
        p_tm1 = np.array(self.p_tm1)

        if self.agg_type == "max":
            max_idx = np.argmax(p_tm1)
            return agent_predictions[..., max_idx]

        elif self.agg_type == "mean":
            weighted_sum = np.dot(agent_predictions, p_tm1)
            total_weight = np.sum(p_tm1)
            return weighted_sum / total_weight

        elif self.agg_type == "median":
            sorted_indices = np.argsort(agent_predictions, axis=-1)
            sorted_agent_predictions = np.take_along_axis(agent_predictions, sorted_indices, axis=-1)
            sorted_weights = np.take_along_axis(np.tile(p_tm1, (agent_predictions.shape[0], 1)),
                                                sorted_indices, axis=-1)

            cumulative_weights = np.cumsum(sorted_weights, axis=-1)
            total_weight = np.sum(p_tm1)
            median_indices = np.apply_along_axis(
                lambda x: np.searchsorted(x, total_weight / 2), axis=-1, arr=cumulative_weights
            )
            return np.take_along_axis(sorted_agent_predictions, median_indices[:, None], axis=-1).squeeze()
        else:
            raise ValueError(f"Unknown aggregation type: {self.agg_type}")
            
        
    def plot_stats(
        self,
        save_path: typing.Optional[str] = None,
    ):
        agents = [f"Agent {n}" for n in range(self.agents_count)] if self.agent_labels is None else self.agent_labels
        agents_losses = np.array(self.stats["losses"])
        agents_weights = np.array(self.stats["weights"])
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(agents_losses.cumsum(axis=0))
        axs[0].set_title("Agents' Losses")
        axs[0].grid()
        axs[1].stackplot(np.arange(len(agents_weights)), np.transpose(agents_weights))
        axs[1].grid()
        axs[1].set_title("Agents' Weights")
        fig.legend(labels=agents, loc="center left", bbox_to_anchor=(0.95, 0.5))
        
        if save_path is not None:
            fig.savefig(os.path.join(save_path, "oamp_stats.png"), bbox_inches="tight")
        else:
            plt.tight_layout()
            plt.show()
            
        plt.close(fig)
        gc.collect()
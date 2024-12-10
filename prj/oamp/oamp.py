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
    ):
        assert args.agents_weights_upd_freq > 0, "Agents' weights update frequency should be greater than 0"
        assert args.loss_fn_window > 0, "Loss function window should be greater than 0"
        
        # Initializing agents
        self.agents_count = agents_count
        self.agents_losses = self.init_agents_losses(args.loss_fn_window)
        self.agents_weights_upd_freq = args.agents_weights_upd_freq # Right now it is measured in groups
        self.agg_type = args.agg_type
        
        # Initializing OAMP
        self.t = 0
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
        agents_predictions: np.ndarray,
        is_new_group: bool = False,
    ):
        # Updating agents' losses
        self.agents_losses.append(agents_losses)
        
        if self.t > 0 and is_new_group: # New group
            self.group_t += 1
        
        # Updating agents' weights
        if self.group_t > 0 and self.group_t == self.agents_weights_upd_freq:
            # print(f'Updating agents weights at timestep {self.t} and group {self.group_t}, {self.groups[self.t]} {self.groups[self.t-1]}')
            self.update_agents_weights()
            self.group_t = 0
            
        self.t += 1
        
        return self.compute_prediction(agents_predictions)

    def update_agents_weights(
        self,
    ):
        # print(f'Updating agents weights at timestep {self.t}')
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
        if self.agg_type == "max":
            return agent_predictions[np.argmax(self.p_tm1)]
        elif self.agg_type == "mean":
            return np.sum(agent_predictions * self.p_tm1) / np.sum(self.p_tm1) 
        elif self.agg_type == "median":
            sorted_indices = np.argsort(agent_predictions)
            sorted_agent_predictions = agent_predictions[sorted_indices]
            sorted_weights = self.p_tm1[sorted_indices]
            cumulative_weights = np.cumsum(sorted_weights)
            total_weight = np.sum(self.p_tm1)
            return sorted_agent_predictions[np.searchsorted(cumulative_weights, total_weight / 2)]
        else:
            raise ValueError(f"Unknown aggregation type: {self.agg_type}")
            
        
    def plot_stats(
        self,
        save_path: typing.Optional[str] = None,
    ):
        agents = [f"Agent {n}" for n in range(self.agents_count)]
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
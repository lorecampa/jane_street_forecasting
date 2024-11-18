import os
import matplotlib.pyplot as plt
import numpy as np

from collections import deque

from oamp.oamp_config import ConfigOAMP
from oamp.oamp_utils import (
    get_m,
    get_p,
    get_r,
    upd_n,
    upd_w,
)


class OAMP:
    def __init__(
        self,
        episodes_len: int,
        agents_count: int,
        args: ConfigOAMP,
    ):
        self.max_step = episodes_len
        # Initializing agents
        self.agents_count = agents_count
        self.agents_rewards = []
        self.agents_returns = deque(maxlen=args.loss_fn_window)
        # Initializing OAMP
        self.l_tm1 = np.zeros(agents_count)
        self.n_tm1 = np.ones(agents_count) * 0.25
        self.w_tm1 = np.ones(agents_count) / agents_count
        self.p_tm1 = np.ones(agents_count) / agents_count
        self.cum_err = np.zeros(agents_count)
        # Initializing OAMP stats
        self.stats = {
            "agents_losses": [],
            "agents_rewards": [],
            "agents_weights": [],
            "ensemble_rewards": [],
        }

    def step(
        self,
        agents_rewards: np.ndarray,
        agents_actions: np.ndarray,
        ensemble_reward: float,
    ):
        # Updating agents' rewards
        self.agents_rewards.append(agents_rewards)
        self.stats["agents_losses"].append(self.l_tm1)
        self.stats["agents_weights"].append(self.p_tm1)
        self.stats['agents_rewards'].append(agents_rewards)
        self.stats['ensemble_rewards'].append(ensemble_reward)
        return self.compute_ensemble_action(agents_actions, self.p_tm1)

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
        return self.p_tm1

    def compute_agents_losses(
        self,
    ) -> np.ndarray:
        # Updating agents' returns
        self.agents_returns.append(np.sum(self.agents_rewards, axis=0))
        self.agents_rewards = []
        # Computing agents' losses
        agents_losses: np.ndarray = -np.sum(self.agents_returns, axis=0)
        # Normalizing agents' losses
        agents_losses_min = agents_losses.min()
        agents_losses_max = agents_losses.max()
        if agents_losses_min != agents_losses_max:
            agents_losses = (agents_losses - agents_losses_min) / (
                agents_losses_max - agents_losses_min
            )
        return agents_losses

    def compute_ensemble_action(
        self,
        agents_actions: np.ndarray,
        agents_weights: np.ndarray,
    ) -> np.ndarray:
        return np.argmax(agents_weights), agents_actions[np.argmax(agents_weights)]
    
    def plot_stats(
        self,
        save_path: str,
        agents_names: list[str],
    ):
        agents_rewards = np.array(self.stats["agents_rewards"])
        agents_losses = np.array(self.stats["agents_losses"])
        agents_weights = np.array(self.stats["agents_weights"])
        ensemble_rewards = np.array(self.stats["ensemble_rewards"])
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(agents_rewards.cumsum(axis=0))
        axs[0].plot(ensemble_rewards.cumsum(axis=0))
        axs[0].set_title("Agents' Rewards")
        axs[0].grid()
        axs[1].plot(agents_losses.cumsum(axis=0))
        axs[1].set_title("Agents' Losses")
        axs[1].grid()
        axs[2].stackplot(np.arange(len(agents_weights)), np.transpose(agents_weights))
        axs[2].grid()
        axs[2].set_title("Agents' Weights")
        fig.legend(labels=agents_names+['multi-agent'], loc="center left", bbox_to_anchor=(0.95, 0.5))
        fig.savefig(os.path.join(save_path, "oamp_stats.png"), bbox_inches="tight")
        plt.close(fig)

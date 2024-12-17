import os
import typing
import warnings
from tqdm import tqdm
from prj.agents.AgentRegressor import AgentRegressor
from torch.utils.data import DataLoader
from prj.model.torch.losses import WeightedMSELoss
from prj.model.torch.models.mlp import Mlp
from prj.model.torch.utils import train
from prj.model.torch.wrappers.base import JaneStreetModelWrapper
import torch.nn as nn

NEURAL_NAME_MODEL_CLASS_DICT = {
    'mlp': Mlp,
}

class AgentNeuralRegressor(AgentRegressor):
    def __init__(
        self,
        agent_type: str,
        seeds: typing.Optional[list[int]] = None,
        n_seeds: int = 1,
    ):
        super().__init__(agent_type, seeds=seeds, n_seeds=n_seeds)
        
        self.agent_class: nn.Module = NEURAL_NAME_MODEL_CLASS_DICT[self.agent_type]
        self.agents: list[JaneStreetModelWrapper] = []
        

    
    def _train_with_seed(self, train_dataloader: DataLoader, val_dataloader: DataLoader, model_args: dict, learn_args: dict, seed: int) -> JaneStreetModelWrapper:
        lr = model_args.pop('learning_rate', None)
        if lr is not None:
            warnings.warn('Learning rate not provided, using default value')
            lr = 5e-4
        
        scheduler = 'ReduceLROnPlateau'
        scheduler_cfg = dict(mode='min', factor=0.1, patience=3, verbose=True, min_lr=1e-8)
        
        model = self.agent_class(**model_args)
        
        scheduler, scheduler_cfg = None, {}
        use_early_stopping, early_stopping_cfg = False, {}
        if val_dataloader:
            use_early_stopping = True
            early_stopping_cfg = {'monitor': 'val_wr2', 'min_delta': 0.00, 'patience': 5, 'verbose': True, 'mode': 'max'}
        
        losses = [WeightedMSELoss()]
        loss_weights = [1]
        
        optimizer = 'Adam'
        optimizer_cfg = {
            'lr': lr,
        }
        model = JaneStreetModelWrapper(
            model, 
            losses=losses, 
            loss_weights=loss_weights, 
            scheduler=scheduler, 
            scheduler_cfg=scheduler_cfg,
            optimizer=optimizer,
            optimizer_cfg=optimizer_cfg,
        )
        
        defalut_learn_args = {
            'max_epochs': 50,
            'precision': '32-true',
            'use_model_ckpt': False,
            'gradient_clip_val': 10,
        }
        defalut_learn_args.update(learn_args)
        
        model:JaneStreetModelWrapper = train(
            model, 
            train_dataloader, 
            val_dataloader, 
            seed=seed,
            compile=False,
            use_early_stopping=use_early_stopping,
            early_stopping_cfg=early_stopping_cfg,
            **defalut_learn_args
        )
        return model
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, model_args: dict, learn_args: dict) -> None:
        self.agents = []
        for seed in tqdm(self.seeds):
            model = self._train_with_seed(train_dataloader, val_dataloader, model_args, learn_args, seed)
            self.agents.append(model)
            
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
            self.agents.append(JaneStreetModelWrapper.load(seed_path))
        

from prj.agents.AgentNeuralRegressor import NEURAL_NAME_MODEL_CLASS_DICT
from prj.agents.AgentTreeRegressor import TREE_NAME_MODEL_CLASS_DICT


_TREE_DATA_CONFIG = {
    'ffill': False,
    'include_symbol_id': False,
}


_NEURAL_BASE_CONFIG = {
    'ffill': True,
    'include_symbol_id': False,
}


DATA_ARGS_CONFIG = {}
for k in TREE_NAME_MODEL_CLASS_DICT.keys():
    DATA_ARGS_CONFIG[k] = _TREE_DATA_CONFIG

for k in NEURAL_NAME_MODEL_CLASS_DICT.keys():
    DATA_ARGS_CONFIG[k] = _NEURAL_BASE_CONFIG
    

# DATA_ARGS_CONFIG.update({
    
# })
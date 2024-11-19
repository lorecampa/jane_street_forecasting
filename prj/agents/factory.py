from prj.agents.AgentTreeRegressor import TREE_NAME_MODEL_CLASS_DICT, AgentTreeRegressor
from prj.agents.base import AgentBase

class AgentsFactory:
    
    @staticmethod
    def build_agent(agent_info: dict) -> AgentBase:
        agent_type = agent_info['agent_type']
        n_seeds = agent_info.get('n_seeds', 1)
        
        agent = None
        if agent_type in TREE_NAME_MODEL_CLASS_DICT.keys():
            agent = AgentTreeRegressor(agent_type, n_seeds=n_seeds)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent
    
    
    @staticmethod
    def load_agent(agent_info: dict) -> AgentBase:
        load_path = agent_info.get('load_path', None)
        agent = AgentsFactory.build_agent(agent_info)
        agent.load(load_path)
        return agent

from typing import Optional 
from abc import abstractmethod 

class BaseEnv: 
    """
    see envs/sudoku.py for an example of BaseEnv interface implementation 
    """

    @abstractmethod
    def step(action: dict) -> tuple[dict, Optional[float]]:
        """ 
        apply action in the env -> return env observation and score 
        """
        pass  

    @abstractmethod
    def reset(self, seed: float | None = None, **kwargs): 
        """
        set a new episode 
        """
        pass 

from typing import Optional, Any
from abc import abstractmethod 

class BaseEnv: 
    """
    see envs/sudoku.py for an example of BaseEnv interface implementation 
    """

    @abstractmethod
    def step(action: dict | list) -> tuple[Any | list, Optional[float]]:
        """ 
        apply action in the env -> return env state Any and score Optional[float] 
        """
        pass  

    @abstractmethod
    def reset(self, seed: float | None = None, **kwargs): 
        """
        set a new episode 
        """
        pass 

    @abstractmethod
    def instructions(self,) -> str:
        pass 
    
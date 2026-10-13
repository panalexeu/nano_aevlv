from abc import abstractmethod
from typing import Optional, Any
 
from llama_cpp import CreateChatCompletionResponse

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
    def state(self,) -> str: 
        """
        return env state in a string format that can be passed into a language model
        """
        pass 
    
    @abstractmethod
    def instructions(self,) -> str:
        pass 

    @abstractmethod
    def tools(self,) -> list[dict]: 
        pass

    @abstractmethod
    def tool_choices(self,) -> list[dict]: 
        pass

    @abstractmethod
    def parse_chat_completion(self, cmpl: CreateChatCompletionResponse) -> list | dict:
        """
        parse chat completion into action 
        """
        pass 

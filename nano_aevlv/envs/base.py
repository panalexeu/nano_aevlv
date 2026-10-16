from abc import abstractmethod
 
from llama_cpp import CreateCompletionResponse

class BaseEnv: 
    """
    see envs/sudoku.py for an example of BaseEnv interface implementation 
    """

    @abstractmethod
    def step(action: dict | list) -> float:
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
    def parse_completion(self, cmpl: CreateCompletionResponse) -> list | dict:
        """
        parse completion into action 
        """
        pass 

    @abstractmethod
    def grammar(self) -> str: 
        """
        define GBNF llama.cpp grammar for the output completion  
        """
        pass

    @abstractmethod
    def is_win(self) -> bool: 
        """
        some envs might have a win state, or some thresh to pass 
        """
        pass
    
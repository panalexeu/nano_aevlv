# ref: https://github.com/google-deepmind/funsearch/blob/main/bin_packing/bin_packing.ipynb
from .base import BaseEnv

from llama_cpp import CreateCompletionResponse

class BinPack(BaseEnv): 
    def __init__(self,): 
        pass 

    def step(action: dict | list) -> float: 
        pass 

    def reset(self, seed: float | None = None): 
        pass

    def state(self,) -> str: 
        pass 

    def instructions(self) -> str: 
        pass 

    def parse_completion(self, cmpl: CreateCompletionResponse) -> list | dict:
        """
        parse completion into action 
        """
        pass 

    def grammar(self) -> str: 
        """
        define GBNF llama.cpp grammar for the output completion  
        """
        pass

    def is_win(self) -> bool: 
        """
        some envs might have a win state, or some thresh to pass 
        """
        pass
    
from .base import BaseEnv
from copy import deepcopy

import numpy as np 
from sudoku import Sudoku
from llama_cpp import CreateCompletionResponse

class SudokuEnv(BaseEnv): 
    def __init__(self):
        self.board = None
        self.solution = None 

    def step(self, action: list) -> float:
        """
        action: 
            'row': int 
            'column': int 
            'number': int 
        reward:
            1 - if row, column, value match with the solution matrix;
            0 - if IndexError or does not match.
        """
        row, column, value = action[0], action[1], action[2]  

        try: 
            if self.solution[row][column] == value: 
                self.board[row][column] = value 
                return 1.0 
            else: 
                return 0.0 
        except IndexError: 
            return 0.0   

    def reset(self, seed: float | None = None, **kwargs): 
        width, height, difficulty = kwargs.get('width', 3), kwargs.get('height'), kwargs.get('difficulty', 0.5)

        puzzle = Sudoku(
            seed=seed, 
            width=width, 
            height=height
        ).difficulty(difficulty)
        self.board = np.array(SudokuEnv._replace_nulls(puzzle.board), dtype=np.int8)
        self.solution = np.array(puzzle.solve().board, dtype=np.int8)

    @classmethod
    def _replace_nulls(cls, board: list[list[int]]) -> list[list[int]]:
        board = deepcopy(board)
        for i in range(len(board)): 
            for j in range(len(board[i])): 
                if board[i][j] is None: board[i][j] = 0
        return board

    def state(self) -> str:
        return str(self.board)

    def instructions(self) -> str:
        return '''
Your task is to solve the provided sudoku puzzle iteratively.
You are provided with a board, which is a two-dimensional matrix where 0 means an empty cell.
For each cell you want to fill, return the row, column, and value, delimited by commas.
Note that indexing starts from 0.
You can fill one or more cells at a time. Put each cell on its own line.
Your response should contain only these lines, with no other text:
row,column,value
row,column,value
...
For instance, filling the cell in the first row and first column with 6, and the cell in the first row and second column with 3, would be:
0,0,6
0,1,3
'''.strip()

    def parse_completion(self, cmpl: CreateCompletionResponse) -> list[list[int]]:
        content = cmpl['choices'][0]['text']
        lines = content.splitlines()
        actions = [line.strip().split(',' )for line in lines]
        return actions

    def grammar(self) -> str:
        return r'''
root ::= line{1,10}
line ::= [0-9] "," [0-9] "," [1-9] "\n"
'''

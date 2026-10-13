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
            # agent provided value in a filled slot 
            if self.board[row][column] != 0: 
                return 0.0

            # agent provided new value correctly => update the board, score 1
            if self.solution[row][column] == value: 
                self.board[row][column] = value 
                return 1.0 
            # otherwise 0.0
            else: 
                return 0.0

        except IndexError: 
            return 0.0   

    def reset(self, seed: float | None = None, **kwargs): 
        width, height, difficulty = kwargs.get('width', 2), kwargs.get('height'), kwargs.get('difficulty', 0.5)

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
        return 'sudoku puzzle: \n' + str(self.board)

    def instructions(self) -> str:
        return '''
Your task is to solve the provided sudoku puzzle iteratively.
The board is a two-dimensional matrix where 0 means an empty cell. Indexing starts from 0.

After each step, you receive feedback:
- Proposed changes: your previous changes with their scores. 1 means correct and applied to the board. 0 means the value is incorrect or the cell was already filled, and the change was not applied.
- Failed changes: changes that raised an error, such as an out-of-range index or a malformed line.

Do not repeat changes that scored 0 or failed. Only fill cells that are 0.

Respond only with one or more lines in the format row,column,value. For example:
0,0,6
0,1,3
'''.strip()

    def parse_completion(self, cmpl: CreateCompletionResponse) -> list[list[int]]:
        content = cmpl["choices"][0]["text"]
        actions = [
            [int(x) for x in line.split(",")]
            for line in content.splitlines()
            if line.strip()
        ]
        return actions

    def grammar(self) -> str:
        return r'''
root ::= line{1,10}
line ::= [0-9] "," [0-9] "," [1-9] "\n"
'''

    def is_win(self):
        return np.array_equal(self.board, self.solution)
    
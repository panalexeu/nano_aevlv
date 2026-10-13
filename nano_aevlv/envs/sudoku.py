from .base import BaseEnv
from copy import deepcopy
from typing import Optional

import numpy as np 
from sudoku import Sudoku

class SudokuEnv(BaseEnv): 
    def __init__(self):
        self.board = None
        self.solution = None 

    def step(self, action: list) -> tuple[dict, Optional[float]]:
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
                return 1 
            else: 
                return 0 
        except IndexError: 
            return 0   

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
    def _replace_nulls(cls, board: list[list]):
        board = deepcopy(board)
        for i in range(len(board)): 
            for j in range(len(board[i])): 
                if board[i][j] is None: board[i][j] = 0
        return board

    def instructions(self):
        return '''
Your task is to solve the provided sudoku puzzle iteratively.
You are provided with a board, which is a two-dimensional matrix where 0 means an empty cell.
Return the row and column along with the value to fill the cell.
Note that indexing starts from 0.
'''.strip()
    
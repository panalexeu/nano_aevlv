from .base import BaseEnv
from copy import deepcopy
from typing import Optional

import numpy as np 
from sudoku import Sudoku

class SudokuEnv(BaseEnv): 
    def __init__(self):
        self.board = None
        self.solution = None 

    def step(action: dict) -> tuple[dict, Optional[float]]:
        pass  

    def reset(self, seed: float | None = None, **kwargs): 
        width, height, difficulty = kwargs.get('width', 3), kwargs.get('height'), kwargs.get('difficulty')

        puzzle = Sudoku(
            seed=seed, 
            width=width, 
            height=height,
            difficulty=difficulty
        )
        board = SudokuEnv._replace_nulls(puzzle.board)
        self.board = np.array(board, dtype=np.int8)
        self.solution = np.array(puzzle.solve().board, dtype=np.int8)

    @classmethod
    def _replace_nulls(cls, board: list[list]):
        board = deepcopy(board)
        for i in range(len(board)): 
            for j in range(len(board[i])): 
                if board[i][j] is None: board[i][j] = 0
        return board
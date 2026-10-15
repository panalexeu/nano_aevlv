# ref: https://en.wikipedia.org/wiki/Circle_packing_in_a_square
import re
from typing import Optional

import numpy as np
from llama_cpp import CreateCompletionResponse

from .base import BaseEnv


class CirclePack(BaseEnv):
    '''
    pack n non-overlapping circles inside the unit square to maximize the sum of radii.

    one action = a full packing: n circles given as (x, y, r).
    a valid packing scores sum(r); an invalid one raises ValueError
    with the reason (circle outside the square, or overlapping circles).
    '''

    def __init__(self, tol: float = 1e-6):
        self.tol = tol

    def reset(self, seed: float | None = None, n: int = 26, target: float = 2.6, **kwargs):
        '''
        args:
            n: number of circles.
            target: sum of radii that counts as a win.
        '''
        self.seed = seed
        self.n = n
        self.target = target
        self.best_packing: Optional[list[tuple[float, float, float]]] = None
        self.best_score = 0.0

    def step(self, action: list) -> float:
        circles = np.asarray(action, dtype=float)
        if circles.shape != (self.n, 3):
            raise ValueError(f'expected {self.n} circles, got {len(circles)}')

        x, y, r = circles.T
        if np.any(r <= 0):
            raise ValueError(f'circle {int(np.argmax(r <= 0))} has non-positive radius')

        # every circle fully inside [0, 1]^2
        outside = (x - r < -self.tol) | (x + r > 1 + self.tol) | (y - r < -self.tol) | (y + r > 1 + self.tol)
        if outside.any():
            raise ValueError(f'circle {int(np.argmax(outside))} extends outside the square')

        # no pairwise overlaps: distance between centers >= r_i + r_j
        d = np.linalg.norm(circles[:, None, :2] - circles[None, :, :2], axis=-1)
        overlap = d < (r[:, None] + r[None, :]) - self.tol
        np.fill_diagonal(overlap, False)
        if overlap.any():
            i, j = np.argwhere(overlap)[0]
            raise ValueError(f'circles {i} and {j} overlap')

        score = float(r.sum())
        if score > self.best_score:
            self.best_score = score
            self.best_packing = [tuple(map(float, c)) for c in circles]
        return score

    def state(self) -> str:
        if self.best_packing is None:
            return f'no valid packing yet. pack {self.n} circles.'
        lines = '\n'.join(f'{x:.4f},{y:.4f},{r:.4f}' for x, y, r in self.best_packing)
        return f'best packing so far (sum of radii = {self.best_score:.4f}):\n{lines}'

    def instructions(self) -> str:
        return '''
Your task is to pack {self.n} circles inside a unit square (from 0 to 1 on both axes) so that the sum of their radii is as large as possible.
Each circle must lie fully inside the square, and no two circles may overlap. 
After each step, you receive feedback:
- Proposed packings: your previous packings with their scores (sum of radii). Higher is better.
- Failed packings: packings that were invalid, with the reason.
Improve on the best packing. Do not repeat failed packings.
Respond only with exactly {self.n} lines in the format x,y,r, one circle per line. For example:
0.1000,0.1000,0.1000
0.3000,0.1000,0.1000
'''

    def parse_completion(self, cmpl: CreateCompletionResponse) -> list:
        text = cmpl['choices'][0]['text']
        circles = [
            tuple(float(v) for v in m)
            for m in re.findall(r'^\s*([\d.]+),([\d.]+),([\d.]+)\s*$', text, re.M)
        ]
        return [circles]          # one action per completion: the whole packing

    def grammar(self) -> str:
        return (
            f'root ::= line{{{self.n}}}\n'
            'line ::= num "," num "," num "\\n"\n'
            'num  ::= "0." [0-9]{1,4}'
        )

    def is_win(self) -> bool:
        return self.best_score >= self.target

    def plot(self, path: str | None = None, show: bool = True):
        '''
        plot the best packing found so far.

        args:
            path: if given, save the figure to this file.
            show: display the figure window.
        '''
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        if self.best_packing is None:
            raise ValueError('no valid packing to plot yet')

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, linewidth=1.5))
        for i, (x, y, r) in enumerate(self.best_packing):
            ax.add_patch(Circle((x, y), r, alpha=0.5, edgecolor='black'))
            ax.text(x, y, str(i), ha='center', va='center', fontsize=8)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.set_title(f'n = {self.n}, sum of radii = {self.best_score:.6f}')

        if path:
            fig.savefig(path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
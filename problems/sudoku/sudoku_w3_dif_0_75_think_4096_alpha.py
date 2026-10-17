from rich import print

from nano_aevlv.search import Search
from nano_aevlv.envs import SudokuEnv

if __name__ == '__main__': 
    seed = 2004
    env = SudokuEnv()
    env.reset(seed, difficulty=0.75, width=3)
    
    print(env.state())
    # [[0 0 0 0 2 0 0 9 0]
    #  [0 0 0 0 3 9 0 0 7]
    #  [0 0 4 5 0 0 0 0 0]
    #  [0 0 0 0 0 0 0 0 0]
    #  [0 0 1 0 9 0 0 0 5]
    #  [0 4 0 7 0 8 0 0 0]
    #  [0 3 5 0 8 0 0 0 0]
    #  [0 0 0 2 0 0 4 0 0]
    #  [4 0 0 0 7 1 0 0 0]]


    search = Search(env, seed, think_tokens=4096, total_tokens=5120,  out_tokens=128)
    for log in search.run(iters=256, sample_n=16, sample_e=2,  sample_t=2): 
        print(log)
    
    print(env.state())
    # [[5 1 7 8 2 6 3 9 4]
    #  [2 6 8 4 3 9 1 5 7]
    #  [3 9 4 5 1 7 6 8 2]
    #  [8 5 3 1 4 2 9 7 6]
    #  [7 2 1 6 9 3 8 4 5]
    #  [9 4 6 7 5 8 2 1 3]
    #  [6 3 5 9 8 4 7 2 1]
    #  [1 7 9 2 6 5 4 3 8]
    #  [4 8 2 3 7 1 5 6 9]]

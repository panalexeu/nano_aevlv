from rich import print

from nano_aevlv.search import Search
from nano_aevlv.envs import SudokuEnv

if __name__ == '__main__': 
    seed = 2004
    env = SudokuEnv()
    env.reset(seed, difficulty=0.25, width=3)
    
    print(env.state())
    # [[0 1 7 8 2 6 3 9 4]
    #  [0 6 0 4 3 9 0 5 7]
    #  [0 9 4 5 0 7 0 8 0]
    #  [8 5 0 1 4 2 0 7 6]
    #  [0 2 1 6 9 0 0 4 5]
    #  [9 4 6 7 5 8 2 1 3]
    #  [6 3 5 0 8 4 0 2 0]
    #  [1 0 9 2 6 5 4 0 8]
    #  [4 8 2 0 7 1 5 0 9]]

    search = Search(env, seed, think_tokens=1024, total_tokens=2048,  out_tokens=128)
    for log in search.run(iters=64, sample_n=16, sample_e=5,  sample_t=2): 
        print(log)
    
    print(env.state())

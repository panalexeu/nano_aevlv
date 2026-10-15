from rich import print

from nano_aevlv.search import Search
from nano_aevlv.envs import SudokuEnv

if __name__ == '__main__': 
    seed = 2004
    env = SudokuEnv()
    env.reset(seed, difficulty=0.5, width=2)
    
    print(env.state())
    # [[0 2 1 0]
    #  [0 0 0 4]
    #  [2 4 3 1]
    #  [1 0 0 0]]

    search = Search(env, seed, think_tokens=4096, total_tokens=5120, out_tokens=128, alpha_name=None)
    for log in search.run(iters=32, sample_n=16, sample_e=2,  sample_t=2): 
        print(log)
    
    print(env.state())
    # [[4 2 1 3]
    #  [3 1 2 4]
    #  [2 4 3 1]
    #  [1 3 4 2]]
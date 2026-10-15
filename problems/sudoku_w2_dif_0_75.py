from rich import print

from nano_aevlv.search import Search
from nano_aevlv.envs import SudokuEnv

if __name__ == '__main__': 
    seed = 2004
    env = SudokuEnv()
    env.reset(seed, difficulty=0.25, width=2)
    search = Search(env, seed, think_tokens=0, total_tokens=1024, out_tokens=128, alpha_name=None)
    for log in search.run(iters=32, sample_n=16, sample_e=5,  sample_t=2): 
        print(log)
    

from rich import print

from nano_aevlv.search import Search
from nano_aevlv.envs import BinPackEnv

if __name__ == '__main__': 
    seed = 2004
    env = BinPackEnv()
    env.reset(seed, name='OR3')
    
    search = Search(env, seed, think_tokens=4096, total_tokens=6144, out_tokens=1024, alpha_name=None)
    for log in search.run(iters=32, sample_n=6, sample_e=6,  sample_t=1): 
        print(log)

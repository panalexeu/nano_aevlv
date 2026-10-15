from rich import print

from nano_aevlv.search import Search
from nano_aevlv.envs import CirclePack

if __name__ == '__main__': 
    seed = 2004
    env = CirclePack()
    env.reset(n=3)
    
    print(env.state())

    search = Search(env, seed, think_tokens=2048, total_tokens=3072,  out_tokens=128, alpha_name=None)
    for log in search.run(iters=4, sample_n=3, sample_e=8,  sample_t=2): 
        print(log)
    
    print(env.state())
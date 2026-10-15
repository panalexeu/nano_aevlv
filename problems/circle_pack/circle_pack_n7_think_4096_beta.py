from rich import print

from nano_aevlv.search import Search
from nano_aevlv.envs import CirclePack

if __name__ == '__main__': 
    seed = 2004
    env = CirclePack()
    env.reset(n=7)
    
    print(env.state())

    search = Search(env, seed, think_tokens=4096, total_tokens=5120, out_tokens=128, alpha_name=None)
    for log in search.run(iters=128, sample_n=8, sample_e=8,  sample_t=0.5): 
        print(log)
    
    print(env.state())
    env.plot(path='./logs/circle_pack_n7_think_4096.png')
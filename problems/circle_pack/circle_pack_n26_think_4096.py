from rich import print

from nano_aevlv.search import Search
from nano_aevlv.envs import CirclePack

if __name__ == '__main__': 
    seed = 2004
    env = CirclePack()
    env.reset(n=26)
    
    print(env.state())

    search = Search(env, seed, think_tokens=4096, total_tokens=5120,  out_tokens=256)
    for log in search.run(iters=32, sample_n=4, sample_e=4,  sample_t=1): 
        print(log)
    
    print(env.state())
    env.plot(path='./logs/circle_pack_n26_think_4096.png')
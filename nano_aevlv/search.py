import os 

import numpy as np
from llama_cpp import Llama

from .envs import SudokuEnv

_data_dir = './models'

def _scores2probs(scores: list[float], t: float=1.0):
    s = np.asarray(scores, dtype=float)
    s = (s - s.mean()) / (s.std() + 1e-8)  
    z = s / t
    z -= z.max()                           
    p = np.exp(z)
    return p / p.sum()

def sample_changes(rng, changes: list[dict], n: int, t: float) -> list[int]: 
    if len(changes) == 0: return []
    if len(changes)-1 < n: n = len(changes)
    scores = [ch['score'] for ch in changes]
    probs = _scores2probs(scores, t)
    return rng.choice(len(probs), size=n, replace=False, p=probs)

def sample_fail_changes(rng, changes: list[dict], n: int) -> list[int]: 
    if len(changes) == 0: return []
    if len(changes)-1 < n: n = len(changes)  
    return rng.choice(len(changes), size=n, replace=False) # uniform sampling 

if __name__ == '__main__': 
    # search params 
    iters = 2
    alpha_name = ''
    alpha_path = ''
    beta_name = 'gemma-4-E2B-it-Q4_0.gguf'
    beta_path = os.path.join(_data_dir, beta_name)
    beta_tokens = 1024
    beta_model = Llama(
        beta_path, 
        n_gpu_layers=-1,
        n_ctx=beta_tokens
    )
    sample_n = 16
    sample_t = 1.0
    sample_e = 4
    seed = 2004
    rng = np.random.default_rng(seed)

    # env 
    env = SudokuEnv()
    env.reset(seed, difficulty=0.5)

    # search loop
    scored_changes, failed_changes = [], []
    for i in range(iters): 
        model = beta_model
        sampled_changes_ids = sample_changes(rng, scored_changes, sample_n, sample_t)
        sampled_changes = []
        sampled_failed_changes_ids = sample_fail_changes(rng, failed_changes, sample_e)
        sampled_failed_changes = []

        chat_completion = model.create_chat_completion(
            messages = [
            {
                "role": "system",
                "content": env.instructions()

            },
            {
                "role": "user",
                "content": env.state()
            }, 
            {
                "role": "user", 
                "content": "proposed changes and their scores: " + str(sampled_changes)
            },
            {
                "role": "user", 
                "content": "failed changes: " + str(sampled_failed_changes)
            }
            ],
            tools=env.tools(),
            tool_choice=env.tool_choices()
        )
        changes = env.parse_chat_completion(chat_completion)
        for change in changes: 
            try: 
                score = env.step(change)
                scored_changes.append({'score': score, 'change': str(change)})
            except Exception as e : 
                failed_changes.append({'exception': str(e), 'change': str(change)})
        breakpoint()

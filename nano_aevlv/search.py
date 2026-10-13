import os 

import numpy as np
from rich import print
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

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
    iters = 16
    total_tokens = 2048
    think_tokens = 1024
    out_tokens = 256
    alpha_name = 'gemma-4-E4B-it-Q4_0.gguf'
    alpha_path = os.path.join(_data_dir, alpha_name)    
    alpha_model = Llama(
        alpha_path, 
        n_gpu_layers=0,
        n_ctx=total_tokens,
        verbose=False
    )
    beta_name = 'gemma-4-E2B-it-Q4_0.gguf'
    beta_path = os.path.join(_data_dir, beta_name)
    beta_model = Llama(
        beta_path, 
        n_gpu_layers=-1,
        n_ctx=total_tokens, 
        verbose=False
    )
    model_t = 1.0
    sample_n = 16
    sample_t = 1.0
    sample_e = 4
    seed = 2004
    rng = np.random.default_rng(seed)

    # other 
    formatter = Jinja2ChatFormatter(
        template=beta_model.metadata['tokenizer.chat_template'], 
        eos_token='', bos_token='', 
        add_generation_prompt=True
    )

    # env 
    env = SudokuEnv()
    env.reset(seed, difficulty=0.5, width=2)
    grammar = LlamaGrammar.from_string(env.grammar())

    # search loop
    scored_changes, failed_changes = [], []
    for i in range(iters): 
        model = alpha_model

        sampled_changes_ids = sample_changes(rng, scored_changes, sample_n, sample_t)
        sampled_changes = [scored_changes[id_] for id_ in sampled_changes_ids]
        sampled_failed_changes_ids = sample_fail_changes(rng, failed_changes, sample_e)
        sampled_failed_changes = [failed_changes[id_] for id_ in sampled_failed_changes_ids]

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
        ]

        # output generation in 2 phases (thinking, then output generation)
        # think
        prompt = formatter(messages=messages).prompt
        out = model.create_completion(
            prompt=prompt + '<|channel>thought\n', 
            max_tokens=think_tokens,
            stop=['<channel|>'],
            stream=True
        )
        thought = ''
        for chunk in out:
            token = chunk['choices'][0]['text']
            print(token, end='', flush=True)
            thought += token

        # output completion
        out = model.create_completion(
            prompt=prompt + thought + '\n<channel|>',
            max_tokens=out_tokens,
            grammar=grammar
        )
        changes = env.parse_completion(out)        
        
        # score proposed changes
        for change in changes: 
            try: 
                score = env.step(change)
                scored_changes.append({'score': score, 'change': str(change)})
            except Exception as e : 
                failed_changes.append({'exception': str(e), 'change': str(change)})

        if env.is_win(): 
            print('\n' + '-' * 64)
            print('search finished, win condition is met')
            print(env.state)
            break

        print('\n' + '-' * 64)
        print(f'iter {i}')
        print(env.state())

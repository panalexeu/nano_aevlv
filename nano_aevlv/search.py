import os 

import numpy as np
from rich import print
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

from .envs import SudokuEnv
from .change_map import FailedChangeMap, ChangeMap

_data_dir = './models'

if __name__ == '__main__': 
    # search params 
    iters = 16
    total_tokens = 2048
    think_tokens = 1536
    out_tokens = 128
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
    score_map, fail_map = ChangeMap(seed=seed), FailedChangeMap(seed=seed)
    for i in range(iters): 
        model = beta_model        
        sampled_changes = score_map.sample(sample_n, sample_t)
        sampled_failed_changes = fail_map.sample(sample_e)

        # input 
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
                score_map.add(text=str(change), score=score)
            except Exception as e : 
                fail_map.add(text=str(change), err=str(e))

        if env.is_win(): 
            print('\n' + '-' * 64)
            print('search finished, win condition is met')
            print(env.state)
            break

        print('\n' + '-' * 64)
        print(f'iter {i}')
        print(f'proposed changes: {[score_map.get_hash(str(change)) for change in changes]}')
        print(f'their scores: {[score_map.get_score(str(change)) for change in changes] }')
        print(env.state())

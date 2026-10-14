import os 
from typing import Any

from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

from .envs import SudokuEnv, BaseEnv
from .change_map import FailedChangeMap, ChangeMap

class Search: 
    def __init__(
        self,
        env: BaseEnv, 
        seed: int,
        total_tokens: int = 2048, 
        think_tokens: int = 512, 
        out_tokens: int = 128, 
        alpha_name: str = 'gemma-4-E4B-it-Q4_0.gguf', 
        beta_name: str = 'gemma-4-E2B-it-Q4_0.gguf', 
    ): 
        self._data_dir = './models'
        self.env = env 
        self.env_grammar = LlamaGrammar.from_string(env.grammar())
        self.total_tokens = total_tokens 
        self.think_tokens = think_tokens 
        self.out_tokens = out_tokens 
        self.alpha_name = alpha_name 
        self.beta_name = beta_name
        self.alpha_path = os.path.join(self._data_dir, self.alpha_name)
        self.alpha_model = Llama(
            self.alpha_path, 
            n_gpu_layers=0, # more capable (typically larger) model is offloaded to cpu 
            n_ctx=total_tokens,
            verbose=False
        )
        self.beta_path = os.path.join(self._data_dir, self.beta_name)
        self.beta_model = Llama(
            self.beta_path, 
            n_gpu_layers=-1,
            n_ctx=total_tokens, 
            verbose=False
        )
        self.formatter = Jinja2ChatFormatter(
            template=self.beta_model.metadata['tokenizer.chat_template'], 
            eos_token='', bos_token='', 
            add_generation_prompt=True
        )
        self.seed = seed

    def __call__(
        self, 
        iters: int = 16,
        sample_t: float = 1.0,
        sample_n: int = 8, 
        sample_e: int = 4, 
        map_size: int = 1024,
        alpha: float = 0.25, 
    ) -> tuple[int, Any]:
        '''
        returns iter and env state
        '''
        alpha_i = int(iters / (iters * alpha))
        score_map = ChangeMap(seed=self.seed, size=map_size) 
        fail_map = FailedChangeMap(seed=self.seed, size=map_size) 
        for i in range(iters): 
            # switch model 
            is_alpha = ((i % alpha_i) == 0)
            if is_alpha:
                model = self.alpha_model
            else: 
                model = self.beta_model 
            
            # sample proposed scored and failed changes
            sampled_changes = score_map.sample(sample_n, sample_t) 
            sampled_fail_changes = fail_map.sample(sample_e)
            
            # output generation in 2 phases (thinking, then output generation)
            # think
            msgs = self._form_msg(sampled_changes, sampled_fail_changes)
            prompt = self.formatter(messages=msgs).prompt
            out = model.create_completion(
                prompt=prompt+'<|channel>thought\n',
                max_tokens=self.think_tokens,
                stop=['<channel|>'], 
                echo=True
            )
            # response
            out = model.create_completion(
                prompt=out['choices'][0]['text'], 
                max_tokens=self.out_tokens,
                grammar=self.env_grammar
            ) 

            # score the proposed changes 
            changes = self.env.parse_completion(out)
            for change in changes: 
                try: 
                    score = self.env.step(change)
                    score_map.add(text=str(change), score=score)
                except Exception as e : 
                    fail_map.add(text=str(change), err=str(e))

            if self.env.is_win():
                return i, env.state()

        return i, env.state()
    
    def _form_msg(self, sampled_changes: list, sampled_fail_changes: list) -> list[dict]: 
        return [
            {
                "role": "system",
                "content": self.env.instructions()
            },
            {
                "role": "user",
                "content": self.env.state()
            }, 
            {
                "role": "user", "content": "proposed changes and their scores: " + str(sampled_changes)
            },
            {
                "role": "user", "content": "failed changes: " + str(sampled_fail_changes)
            }
        ]

if __name__ == '__main__': 
    seed = 2004
    env = SudokuEnv()
    env.reset(seed, difficulty=0.33, width=2)
    search = Search(env, seed, think_tokens=1536)
    res = search(sample_t=2)
    print(res)
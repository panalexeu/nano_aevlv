import os 
import json
import time 
from pathlib import Path
from datetime import datetime
from typing import Any, Iterator 

from rich import print 
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

from .envs import SudokuEnv, BaseEnv
from .change_map import FailedChangeMap, ChangeMap

class Search: 
    def __init__(
        self,
        env: BaseEnv, 
        seed: int,
        total_tokens: int = 2536, 
        think_tokens: int = 0, 
        out_tokens: int = 128, 
        alpha_name: str | None = 'gemma-4-E4B-it-Q4_0.gguf', 
        beta_name: str = 'gemma-4-E2B-it-Q4_0.gguf', 
        log_dir: str = 'logs',
    ): 
        self._data_dir = './models'
        self.env = env 
        self.env_grammar = LlamaGrammar.from_string(env.grammar())
        self.total_tokens = total_tokens 
        self.think_tokens = think_tokens 
        self.out_tokens = out_tokens 
        self.alpha_name = alpha_name 
        self.beta_name = beta_name
        if self.alpha_name is not None: 
            self.alpha_path = os.path.join(self._data_dir, self.alpha_name)
            self.alpha_model = Llama(
                self.alpha_path, 
                n_gpu_layers=0, # more capable (typically larger) model is offloaded to cpu 
                n_ctx=total_tokens,
                verbose=False, 
                flash_attn=True, 
            )
        else: 
            self.alpha_model = None
        self.beta_path = os.path.join(self._data_dir, self.beta_name)
        self.beta_model = Llama(
            self.beta_path, 
            n_gpu_layers=-1,
            n_ctx=total_tokens, 
            verbose=False, 
            flash_attn=True, 
        )
        self.formatter = Jinja2ChatFormatter(
            template=self.beta_model.metadata['tokenizer.chat_template'], 
            eos_token='', bos_token='', 
            add_generation_prompt=True
        )
        self.seed = seed
        self.log_dir = Path(log_dir)
        self.log_file: Path | None = None 

    def run(
        self, 
        iters: int = 16,
        sample_t: float = 1.0,
        sample_n: int = 6, 
        sample_e: int = 3, 
        map_size: int = 1024,
        alpha: float = 0.25
    ) -> Iterator[dict]: 
        self._log_init()

        alpha_i = int(iters / (iters * alpha))
        score_map = ChangeMap(seed=self.seed, size=map_size) 
        fail_map = FailedChangeMap(seed=self.seed, size=map_size) 
        for i in range(iters): 
            # switch model 
            is_alpha = ((i % alpha_i) == 0) and self.alpha_model
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
            start_t = time.time()
            if self.think_tokens > 0: 
                out = model.create_completion(
                    prompt=prompt+'<|channel>thought\n',
                    max_tokens=self.think_tokens,
                    stop=['<channel|>'], 
                    echo=True
                )['choices'][0]['text']
                reasoning = out.split('<|channel>thought\n')[-1]
            else: 
                out = prompt 
                reasoning = '[no reasoning]'
            # response
            out = model.create_completion(
                prompt=out, 
                max_tokens=self.out_tokens,
                grammar=self.env_grammar
            ) 
            end_t = time.time()

            # score the proposed changes 
            changes = self.env.parse_completion(out)
            scores = []
            for change in changes: 
                try: 
                    score = self.env.step(change)
                    score_map.add(text=str(change), score=score)
                    scores.append(score)
                except Exception as e : 
                    fail_map.add(text=str(change), err=str(e))

            # logging 
            log = self._log(reasoning, sampled_changes, sampled_fail_changes, changes, scores, i, is_alpha, round(end_t - start_t, 2))
            yield log

            if self.env.is_win():
                return

    def __call__(
        self,
        **kwargs
    ) -> tuple[int, Any]:
        '''run to completion; returns last iter and env state.'''
        for log in self.run(**kwargs): 
            i = log['iter']
        return i, self.env.state()
    
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

    def _log_init(self): 
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f'{datetime.now():%Y-%m-%d_%H-%M-%S}.jsonl'
        self.log_file.touch() 

    def _log(
        self,
        reasoning: str,
        sampled_changes: list,
        sampled_fail_changes: list,
        changes: list,
        scores: list,
        iter: int,
        is_alpha: bool, 
        time: float
    ) -> dict:
        '''append one iteration as a json line and flush it to disk.'''
        log = {
            'iter': iter,
            'is_alpha': is_alpha,
            'sampled_changes': sampled_changes,
            'sampled_fail_changes': sampled_fail_changes,
            'reasoning': reasoning,
            'changes': changes,
            'scores': scores, 
            'time': time
        }
        with self.log_file.open('a', encoding='utf-8') as f:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')
        return log

import numpy as np 

class ChangeMap:
    '''
    fixed-size, lossy hash map that keeps the best-scoring change per slot.

    each text is hashed (fnv-1a) to one of `size` slots; there is no chaining
    or resizing. on a collision (same slot, whether same or different text),
    the entry with the higher score is kept and the other is dropped.
    use it only where higher score = better; negate scores to minimize.

    args:
        size: number of slots; must be a power of two.
    '''
    def __init__(self, seed: int, size: int = 1024):
        # size is a power of two
        self.size = size
        self.seed = seed 
        self.rng = np.random.default_rng(self.seed)
        self._s_arr = np.full(self.size, fill_value=np.nan)
        self._t_arr = np.full(self.size, fill_value='', dtype=object)
        self._col = 0

    def add(self, text: str, score: float) -> int:
        '''
        insert `text`; return its slot index if stored, or -1 if a
        higher-scoring entry already occupies the slot.
        '''
        idx = self._get_idx(text)
        # collision
        if not np.isnan(self._s_arr[idx]):
            self._col += 1
            # reassign if the new change score is higher
            if self._s_arr[idx] < score:
                self._s_arr[idx] = score
                self._t_arr[idx] = text
                return idx
        # no collision
        else:
            self._s_arr[idx] = score
            self._t_arr[idx] = text
            return idx
        return -1

    def get_score(self, text: str) -> float | None: 
        return self._s_arr[self._get_idx(text)]

    def get_hash(self, text: str) -> int: 
        return self._hash(text)

    def _get_probs(self, idxs: list[int], t: float):
        s = self._s_arr[idxs]
        s = (s - s.mean()) / (s.std() + 1e-8)  
        z = s / t
        z -= z.max() # trick to not overflow exp                            
        p = np.exp(z)
        return p / p.sum()

    def sample(self, n: int, t: float) -> list[tuple[str, int]]: 
        idxs = [self._get_idx(t) for t in self._t_arr if t != '']
        probs = self._get_probs(idxs, t)
        sampled_idxs = self.rng.choice(len(idxs), size=n, replace=False, p=probs)
        return [
            (self._t_arr[idxs[idx]], self._s_arr[idxs[idx]]) 
            for idx in sampled_idxs
        ]

    def _hash(self, text: str) -> int:
        # fnv1a, ref: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        bytes_ = text.encode()
        mask64 = (1 << 64) - 1
        h = 0xcbf29ce484222325
        for b in bytes_:
            h ^= b
            h = (h * 0x100000001b3) & mask64
        return h

    def _get_idx(self, text: str) -> int: 
        h = self._hash(text)
        return h & (self.size - 1)

    def __len__(self) -> int:
        return len(self._s_arr)

class FailedChangeMap(ChangeMap): 
    '''
    fixed-size, lossy hash map for failed changes.
    '''
    def __init__(self, seed, size = 1024):
        self.size = size 
        self.seed = seed 
        self.rng = np.random.default_rng(self.seed)
        self._t_arr: list[str] = np.full(self.size, fill_value='', dtype=object)
        self._e_arr: list[str] = np.full(self.size, fill_value='', dtype=object)
        self._col = 0 

    def add(self, text: str, err: str) -> int: 
        idx = self._get_idx(text) 
        if self._t_arr[idx] != '': self._col += 1
        self._t_arr[idx] = text; self._e_arr[idx] = err
        return idx 

    def sample(self, n: int) -> list[tuple[str, str]]: 
        idxs = [self._get_idx(t) for t in self._t_arr if t != '']
        sampled_idxs = self.rng.choice(len(idxs), size=n, replace=False) # uniform sampling 
        return [
            (self._t_arr[idxs[idx]], self._e_arr[idxs[idx]])
            for idx in sampled_idxs
        ]

    def __len__(self):
        return len(self._t_arr)

    def get_score(self, text):
        raise NotImplementedError
    
    def _get_probs(self, idxs, t):
        raise NotImplementedError
    
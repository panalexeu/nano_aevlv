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
    def __init__(self, size: int = 1024):
        # size is a power of two
        self.size = size
        self._s_arr: list[float] = [None] * size
        self._t_arr: list[str] = [None] * size
        self._col = 0

    def add(self, text: str, score: float) -> int:
        '''
        insert `text`; return its slot index if stored, or -1 if a
        higher-scoring entry already occupies the slot.
        '''
        idx = self._get_idx(text)

        # collision
        if self._s_arr[idx] is not None:
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

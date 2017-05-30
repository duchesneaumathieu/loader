import numpy as np
from permutation import RandomPermutation

class BatchIterator(object):
    def __init__(self, loader, infos, rng, batch_size):
        self._loader = loader
        self._infos = infos
        self._rng = rng
        self._batch_size = batch_size
        self._length = int(np.ceil(len(rng)/float(batch_size)))
        self._left = len(rng)
        
    def __len__(self):
        return self._length
    
    def __iter__(self):
        self._rng.__iter__()
        self._left = len(self._rng)
        return self
    
    def __getitem__(self, k):
        batch_info = [self._infos[self._rng[k]]]
        raw_batch = [self._loader.load(batch_info[0])]
        return self._loader.preprocess(raw_batch, batch_info), batch_info
    
    def next(self):
        if self._left <= 0: raise StopIteration()
        r = range(min(self._left, self._batch_size))
        self._left -= self._batch_size
        
        batch_info = [self._infos[next(self._rng)] for _ in r]
        raw_batch = map(self._loader.load, batch_info)
        return self._loader.preprocess(raw_batch, batch_info), batch_info
    
    def tolist(self):
        return [self._infos[n] for n in self._rng]
    
class AbstractLoader(object):
    def __init__(self, name='', which_set='train', **kwargs):
        self.name = name
        self.which_set = which_set
        
    def get_infos(self):
        NotImplementedError()
        
    def load(self, info):
        NotImplementedError()
    
    def preprocess(self, data, batch_info):
        NotImplementedError()
        
    def get_epochs(self, seed=0, batch_size=32, complete_batch=True,
                  partial=1, start=None, stop=None):
        k = batch_size
        infos = self.get_infos()
        rng = RandomPermutation(len(infos), seed=seed)
        
        n = len(infos) // k
        pr = [[(i*n)//partial*k, ((i+1)*n)//partial*k] for i in range(partial)]
        if not complete_batch: pr[-1][1] = len(infos)
        
        start = 0 if start is None else start; start = max(start, 0)
        stop = partial if stop is None else stop; stop = min(stop, partial)
        
        return [BatchIterator(self, infos, rng[pr[i][0]:pr[i][1]], k) for i in range(start, stop)]
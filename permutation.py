import numpy as np

import pyximport
pyximport.install()
from cnumber import get_bad_idx
from number import powermod, next_prime, bmg

class RandomPermutation(object):
    """
    Implement a memory efficient random permutation
    of the first n integer.
    
    Note: slice will return an iterator.
    """     
    def __init__(self, n, seed=0, _p=None, _g=None, _bad_idx=None, _bias=None, _pslc=None, _slc=slice(None)):
        self.n = n
        self.seed = seed
        self.p = next_prime(start=self.n+2, k=16) if _p is None else _p
        self.g = bmg(self.p, start=int(2/3.*self.p), skip=self.seed) if _g is None else _g
        self.bad_idx = get_bad_idx(self.p, self.n, self.g) if _bad_idx is None else _bad_idx
        np.random.seed(self.seed)
        self.bias = np.random.randint(self.p) if _bias is None else _bias
        np.random.seed(None)
        
        
        self._start = 0
        self._stop = self.n
        self._step = 1
       
        pstart = 0 if _pslc is None else _pslc.start
        pstop = len(self) if _pslc is None else _pslc.stop
        pstep = 1 if _pslc is None else _pslc.step
        l = int(np.ceil((pstop - pstart)//float(pstep)))
        
        self._start = 0 if _slc.start is None else _slc.start
        if self._start < 0: self._start = max(l + self._start, 0)
        self._start = self._start*pstep + pstart

        self._stop = l if _slc.stop is None else _slc.stop
        self._stop = min(self._stop, l)
        if self._stop < 0: self._stop = l + self._stop
        self._stop = self._stop*pstep + pstart

        self._step = pstep if _slc.step is None else pstep*_slc.step
        if self._step == 0: raise ValueError('slice step cannot be zero')
        elif self._step < 0: raise NotImplementedError('negative slice step not implemented')
            
        self._slc = slice(self._start, self._stop, self._step)
            
        self._value = self.g
        self._p_idx = self._to_p_space(0)
        self._idx = self._start

    def __len__(self):
        return int(np.ceil((self._stop - self._start)/float(self._step)))
    
    def _to_n_space(self, k):
        return k*self._step + self._start
    
    def _to_p_space(self, k):
        for i in self.bad_idx:
            if k >= i: k += 1
            else: break
        return k
    
    def __getitem__(self, k):
        if isinstance(k, slice):
            return RandomPermutation(self.n, self.seed, self.p, self.g, self.bad_idx, self.bias, self._slc, k)
        
        if not -len(self) < k < len(self):
            raise IndexError()
        
        k = self._to_n_space(k % len(self))
        k = self._to_p_space(k)
        return (powermod(self.g, k+1, self.p) + self.bias) % self.n
    
    def next(self):
        k = self._idx
        if k >= self._stop: raise StopIteration()

        p_idx = self._to_p_space(self._idx)
        diff = p_idx - self._p_idx
        self._value = (self._value*powermod(self.g, diff, self.p)) % self.p

        self._p_idx = p_idx
        self._idx += self._step
        return (self._value + self.bias) % self.n

    def __iter__(self):
        self._value = self.g
        self._p_idx = self._to_p_space(0)
        self._idx = self._start
        return self
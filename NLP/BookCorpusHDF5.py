from loader.AbstractLoader import AbstractLoader
import loader

import h5py
import numpy as np
import os

class BookCorpusHDF5(AbstractLoader):
    def __init__(self, name='BookCorpus',
                 size='big',
                 vocabulary='vocabulary_5.txt',
                 seq_length=32, start=100, stop=100,
                 **kwargs):
        
        super(BookCorpusHDF5, self).__init__(name=name, **kwargs)
        
        sizes = ['mini', 'small', 'medium', 'big']
        assert size in sizes or isinstance(size, int), 'size must be an integer or in {}'.format(sizes)
        
        self.path = os.path.join(loader.__path__[0], 'datasets', 'BookCorpus') + '/'
        with open(self.path + vocabulary, 'r') as f: self.vocabulary = f.read().decode("utf-8-sig").split(u' ')
            
        self.size = size if isinstance(size, int) else 10**(sizes.index(size)+1)
        self.seq_length = seq_length
        self.start = start
        self.stop = stop
        self._encode = dict([(w, n) for n, w in enumerate(self.vocabulary, 1)])
        self._decode = dict([(n, w) for n, w in enumerate(self.vocabulary, 1)])
        
        self.hdf5 = h5py.File('{}tokenized_books.hdf5'.format(self.path), 'r')
        self.file = self.hdf5[self.which_set]
        self.data = self.file['books']
        
        names = self.file['names'][:]
        genres = self.file['genres'][:]
        lengths = self.file['lengths'][:]
        self.infos = dict()
        keys = map(lambda x: int(x[4:]), self.data.keys())
        for k in keys:
            book = 'book{}'.format(k)
            self.infos[book] = dict()
            self.infos[book]['name'] = names[k]
            self.infos[book]['genre'] = genres[k]
            self.infos[book]['length'] = lengths[k]
                
    def __del__(self):
        self.hdf5.close()       
        
    def get_infos(self, seed=0):
        infos = list()
        keys = sorted(map(lambda x: int(x[4:]), self.data.keys()))
        for k in keys[:self.size]:
            beg = self.start
            end = beg + self.seq_length
            maximum = self.infos['book{}'.format(k)]['length'] - self.stop
            while(end < maximum):
                infos += [(k, beg, end)]
                beg = end
                end += self.seq_length
        return infos
    
    def load(self, info):
        k, beg, end = info
        tokens = self.data['book{}'.format(k)][beg:end]
        return tokens
    
    def preprocess(self, seqs, batch_infos):
        return [np.array(seqs, dtype=np.int64)]
    
    def decode(self, a):
        if isinstance(a, np.ndarray): a = a.tolist()
        if isinstance(a[0], list):
            return [self.decode(b) for b in a]
        else: 
            return u' '.join([self._decode.get(w, 'DFT') for w in a])
        
    def encode(self, a):
        if isinstance(a, list):
            return np.array([self.encode(b) for b in a], dtype=np.int64)
        else:
            return np.array([self._encode.get(w, 0) for w in a.split(u' ')], dtype=np.int64)
        
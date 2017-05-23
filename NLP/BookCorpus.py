from loader.AbstractLoader import AbstractLoader
import loader
import numpy as np
import json
import os

class BookCorpus(AbstractLoader):
    def __init__(self, name='BookCorpus', size=None, vocabulary='vocabulary_5.txt', seq_length=32, **kwargs):
        super(BookCorpus, self).__init__(name=name, **kwargs)
        
        sizes = ['big', 'medium', 'small', 'mini']
        assert size in sizes, 'size not in {}'.format(sizes)
        
        self.path = os.path.join(loader.__path__[0], 'datasets', 'BookCorpus') + '/'
        self.size = size
        with open(self.path + vocabulary, 'r') as f: self.vocabulary = f.read().decode("utf-8-sig").split(u' ')
        self.seq_length = seq_length
        self.encode = dict([(w, n) for n, w in enumerate(self.vocabulary, 1)])
        self.decode = dict([(n, w) for n, w in enumerate(self.vocabulary, 1)])
        
        files_id = list()
        if size is None: pass
        elif self.which_set == 'train':
            if size == 'big': files_id = [i for i in range(14)]
            else: files_id = [0]
        elif self.which_set == 'valid':
            if size == 'big': files_id = [14, 15]
            else: files_id = [14]
        elif self.which_set == 'test':
            if size =='big': files_id = [16, 17, 18, 19]
            else: files_id = [16]
        
        full_data = dict()
        sub_paths = ['tokenized_books/tokenized_books_part_{}.json'.format(i) for i in files_id]
        for sub_path in sub_paths:
            with open(self.path + sub_path, 'r') as f: full_data.update(json.load(f))
        
        n_books = 0
        if size == 'big': n_books = len(full_data)
        elif size == 'medium': n_books = 500
        elif size == 'small': n_books = 100
        elif size == 'mini': n_books = 10
            
        self.min_id = int(min(full_data.keys()))
        self.books_id = range(self.min_id, self.min_id + n_books)
            
        self.data = dict([(int(k), v) for k, v in full_data.iteritems() if int(k) in self.books_id])
        
        self.start()
        
    def get_infos(self):
        infos = list()
        for k, book in self.data.iteritems():
            beg = 0
            end = self.seq_length
            tokens = book['tokens']
            while(end < len(tokens)):
                infos += [(k, beg, end)]
                beg = end
                end += self.seq_length
        return infos
    
    def load(self, info, batch_infos=None):
        k, beg, end = info
        tokens = self.data[k]['tokens'][beg:end]
        return [self.encode.get(token, 0) for token in tokens]
    
    def preprocess(self, seqs, batch_infos):
        return seqs
    
    def decode_seq(self, seq):
        return u' '.join([self.decode.get(n, u'DFT') for n in seq])
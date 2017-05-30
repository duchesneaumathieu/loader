import numpy as np
import pickle
import gzip
import os

import loader
from loader.AbstractLoader import AbstractLoader

class Mnist(AbstractLoader):
    def __init__(self, name='Mnist', img=False, **kwargs):
        super(Mnist, self).__init__(name=name, **kwargs)
        
        self.path = os.path.join(loader.__path__[0], 'datasets', 'Mnist')
        f = gzip.open(os.path.join(self.path, 'mnist.pkl.gz'))
        data = pickle.load(f)
        
        if self.which_set == 'train': i = 0
        elif self.which_set == 'valid': i = 1
        elif self.which_set == 'test': i = 2
            
        self.x, self.y = data[i]
        self.img = img
   
    def get_infos(self, seed=0):
        np.random.seed(seed)
        infos = np.random.permutation(range(len(self.y)))
        np.random.seed(None)
        return infos
    
    def load(self, idx):
        return self.x[idx], self.y[idx]
    
    def preprocess(self, imgs, batch_infos=None):
        x, y_ = zip(*imgs)
        y = np.zeros((len(y_),10))
        y[(range(len(y_)), y_)] = 1
        if self.img: x = np.array(x).reshape((len(y_), 28, 28))
        return [np.array(x), np.array(y)]
        

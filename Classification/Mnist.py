import numpy as np
import loader
import pickle
import gzip
import os

class Mnist(loader.AbstractLoader):
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
        
        self.start()
        
        
    def get_infos(self):
        return range(len(self.y))
    
    def load(self, idx, batch_infos=None):
        return self.x[idx], self.y[idx]
    
    def preprocess(self, imgs, batch_infos=None):
        x, y_ = zip(*imgs)
        y = np.zeros((len(y_),10))
        y[(range(len(y_)), y_)] = 1
        if self.img: x = np.array(x).reshape((len(y_), 28, 28))
        return [np.array(x), np.array(y)], 'whatever'
        

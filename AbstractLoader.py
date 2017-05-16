import numpy as np
import time
import threading
from Queue import Queue
from Queue import Empty

class Jobs(object):
    def __init__(self, fn, maxsize, n_threads=1, inputs=None, log_name=''):
        self.fn = fn
        self.maxsize = maxsize
        self.n_threads = n_threads
        self.islog=isinstance(log_name, str)
        self.log_name=log_name
        self.null = object()
        
        self.fetching_from = None
        self.fetching_thread = None
        
        self.inputs = Queue()
        self.outputs = [None for k in range(maxsize)]
        self.ready = [threading.Event() for k in range(maxsize)]
        self.free_idx = Queue()
        self.current = 0
        for k in range(maxsize):
            self.free_idx.put_nowait(k)
        
        self.done = threading.Event()
        self.order_lock = threading.RLock()
        self.get_lock = threading.RLock()
        
        self.threads = []
        for i in range(n_threads):
            t = threading.Thread(target=self.execute)
            t.daemon = True
            self.threads.append(t)
            t.start()
            
        if isinstance(inputs, Jobs):
            self.fetching_from = inputs
            self.fetching_thread = threading.Thread(target=self.fetching_execute)
            self.fetching_thread.daemon = True
            self.fetching_thread.start()
            
    def fetching_execute(self):
        while not self.done.isSet():
            self.put(self.fetching_from.get())
        
    def put(self, job):
        self.inputs.put_nowait(job)
    
    def get(self, timeout=None):
        self.get_lock.acquire()
        if not self.ready[self.current].wait(timeout):
            self.get_lock.release()
            raise Empty()
        if self.done.isSet():
            self.get_lock.release()
            return self.null
        self.ready[self.current].clear()
        out = self.outputs[self.current]
        self.outputs[self.current] = None
        self.free_idx.put(self.current)
        self.current = (1+self.current)%self.maxsize
        self.get_lock.release()
        return out
    
    def exit(self):
        self.done.set()
        for e in self.ready:
            #free threads from get_lock
            e.set()
        while any(map(lambda x: x.isAlive(), self.threads)):
            #free threads from order_lock
            self.free_idx.put(self.null)
            self.inputs.put(self.null)
    
    def execute(self):
        while 0<1:
            if self.done.isSet(): break
            self.order_lock.acquire()
            idx = self.free_idx.get()
            inp = self.inputs.get()
            self.order_lock.release()
            if self.done.isSet(): break
            
            try:
                out = self.fn(*inp)
            except Exception as e:
                out = self.null
                if self.islog:
                    with open('{}_loader.log'.format(self.log_name), 'a') as f:
                        f.write(e.__str__())
            self.outputs[idx] = out
            self.ready[idx].set()
            
    def isnull(self, obj):
        return obj==self.null

'''
class AbstractLoader(object):
    def get_infos(self): #example
        return list(np.arange(-95,200))
    
    def load(self, info, batch_info): #example
        return info
    
    def preprocess(self, items, batch_info): #example
        if any(np.array(items)<=0): 
            raise ValueError('Batch error. negative number(s):\n{}\n\n'.format(batch_info))
        return np.log(items), batch_info
    
    def __init__(self,
                 which_set='train',
                 batch_size=32, n_threads=1,
                 queue_size=1, name='',
                 **kwargs):
        sets = ['train', 'valid', 'test']
        assert which_set in sets, 'which_set not in {}'.format(sets)
        assert isinstance(name, str), 'name must be a string'
        self.which_set=which_set
        self.batch_size=batch_size
        self.n_threads=n_threads
        self.queue_size=queue_size
        self.name=name
                 
        if kwargs: print 'warning: unused keyword(s): {}'.format(kwargs.keys())
        
        self.length = None
        self.current = None
        self.loader = None
        self.preprocessors = None
        
    def load_batch(self, batch_info):
        batch = []
        for info in batch_info:
            batch.append(self.load(info, batch_info))
        return batch, batch_info
        
    def get_length(self):
        bs = self.batch_size
        infos = self.get_infos()
        return len(infos)//bs
        
    def __len__(self):
        return self.length
    
    def put_batch_infos(self):
        bs = self.batch_size
        infos = self.get_infos()
        for k in range(len(self)):
            self.loader.put([infos[bs*k:bs*(k+1)]])
            
    def start(self):
        if self.length is None:
            self.length = self.get_length()
        self.loader = Jobs(self.load_batch, 1, n_threads=1, log_name=self.name)
        self.preprocessors = Jobs(
            self.preprocess, self.queue_size, n_threads=self.n_threads, inputs=self.loader, log_name=self.name)
        
        self.put_batch_infos()
        self.current = 0
    
    def restart(self):
        self.preprocessors.exit()
        self.loader.exit()
        self.start()
    
    def __iter__(self):
        self.restart()
        return self
    
    def next(self):
        r=self.preprocessors.null
        while(self.preprocessors.isnull(r)):
            if self.current >= len(self):
                raise StopIteration()
            self.current += 1
            r = self.preprocessors.get(5)
        return r
'''
    
class AbstractLoader(object):
    def get_infos(self): #example
        return list(np.arange(-95,200))
    
    def load(self, info, batch_info): #example
        return info
    
    def preprocess(self, items, batch_info): #example
        if any(np.array(items)<=0): 
            raise ValueError('Batch error. negative number(s):\n{}\n\n'.format(batch_info))
        return np.log(items), batch_info
    
    def __init__(self,
                 which_set='train',
                 batch_size=32, name='',
                 **kwargs):
        sets = ['train', 'valid', 'test']
        assert which_set in sets, 'which_set not in {}'.format(sets)
        assert isinstance(name, str), 'name must be a string'
        self.which_set=which_set
        self.batch_size=batch_size
        self.name=name
                 
        if kwargs: print 'warning: unused keyword(s): {}'.format(kwargs.keys())
        
        self.batches_infos = []
        self.current = 0
        
    def load_batch(self, batch_info):
        batch = []
        for info in batch_info:
            batch.append(self.load(info, batch_info))
        return batch, batch_info
    
    def get_batches(self, infos):
        batches = []
        num_batches = int(np.ceil(len(infos)/self.batch_size))
        for k in range(num_batches):
            batches.append(infos[k*self.batch_size:(k+1)*self.batch_size])
        self.batches_infos = batches
        
    def __len__(self):
        return len(self.batches_infos)
            
    def start(self):
        self.get_batches(self.get_infos())
        self.current = 0
    
    def restart(self):
        self.start()
    
    def __iter__(self):
        self.restart()
        return self
    
    def next(self):
        if self.current >= len(self):
            raise StopIteration()
        batch_infos = self.batches_infos[self.current]
        self.current += 1
        return self.preprocess(*self.load_batch(batch_infos))
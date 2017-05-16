import os
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from skimage.transform import resize
import json
import pickle
import loader

class MsCoco(loader.AbstractLoader):
    def __init__(self, name='MsCoco', valid_split_ratio=0.1, **kwargs):
        super(MsCoco, self).__init__(name=name, **kwargs)
        
        self.path = os.path.join(loader.__path__[0], 'datasets', 'MsCoco')
        with open(os.path.join(self.path, 'annotations', 'dictio.json'), 'r') as f:
                self.dictio = dict(json.load(f))
        
        if self.which_set == 'train':
            self.image_path = os.path.join(self.path, 'images', 'train2014')
            with open(os.path.join(self.path, 'annotations', 'MsCoco_Captioning_train2014.pkl'), 'r') as f:
                self.data = pickle.load(f)
            self.img_ids = self.data.keys()
            
        elif self.which_set == 'valid' or 'test':
            self.image_path = os.path.join(self.path, 'images', 'val2014')
            with open(os.path.join(self.path, 'annotations', 'MsCoco_Captioning_valid2014.pkl'), 'r') as f:
                self.data = pickle.load(f)
            self.img_ids = self.data.keys()
            
            split = int(valid_split_ratio*len(self.img_ids))
            if self.which_set == "valid":
                self.img_ids = self.img_ids[:split]
            elif self.which_set == "test":
                self.img_ids = self.img_ids[split:]
                
        self.start()
        
    def get_infos(self):
        #remove a corrupted image
        if 167126 in self.data: del self.data[167126]
                
        #Finding a good order
        flat_batch = list()
        order = dict()
        mix = np.random.permutation(self.data.keys())
        for img_id in mix:
            data = self.data[img_id]
            cap_id = np.random.choice(data['captions'].keys())
            length = data['captions'][cap_id]['length']
            if length not in order:
                order[length] = [(img_id, cap_id)]
            else:
                order[length].append((img_id, cap_id))
                
            if len(order[length]) == self.batch_size:
                flat_batch.extend(order[length])
                del order[length]
                
        return flat_batch
    
    def load(self, info, batch_infos=None):
        path = os.path.join(self.image_path, self.data[info[0]]['file_name'])
        if not os.path.exists(path):
            raise RuntimeError('Image {} is missing'.format(self.data[info[0]]['file_name']))
        
        im = plt.imread(path)
        if im.ndim==2: im = np.repeat(im.reshape(im.shape+(1,)), 3, axis=2)
            
        return im
    
    def prep_img(self, im):
        a, b, _ = im.shape
        d = int(256*max(a,b)/float(min(a, b)))
        new_size = (256, d) if a < b else (d, 256)
        im = resize(im, new_size)
        
        a, b = new_size
        ra = np.random.randint(0, a-223)
        rb = np.random.randint(0, b-223)
        im = im[ra:ra+224,rb:rb+224]
        
        return im.transpose(2,0,1).astype('float32')*255
    
    def prep_caption(self, cap):
        words = ['BEG']+cap.split()+['END']
        one_hots = [[1 if word in self.dictio and i==self.dictio[word] else 0
                     for i in range(8848)]
                    for word in words]
        return np.array(one_hots, dtype='float32')
    
    def preprocess(self, raw_imgs, batch_infos):
        imgs, captions = list(), list()
        for raw_img, info in zip(raw_imgs, batch_infos):
            img_id, cap_id = info
            
            img = self.prep_img(raw_img)
            imgs.append(img)
            
            raw_caption = self.data[img_id]['captions'][cap_id]['caption']
            caption = self.prep_caption(raw_caption)
            captions.append(caption)
            
        return map(lambda x: np.array(x), [imgs, captions]), batch_infos
    
    def convert(self, cap):
        return ' '.join([self.dictio[np.argmax(c)] for c in cap.tolist()[1:-1]])
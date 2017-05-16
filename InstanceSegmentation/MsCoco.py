import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from matplotlib.path import Path
from loader.InstanceSegmentation.InstanceSegmentation import InstanceSegmentation
from PIL import Image
import pickle
import loader

class MsCoco(InstanceSegmentation):
    def __init__(self, name='MsCoco', valid_split_ratio=0.1, **kwargs):
        super(MsCoco, self).__init__(name=name, **kwargs)
        
        self.path = os.path.join(loader.__path__[0], 'datasets', 'MsCoco')
        #COCO = lambda x: l.coco
        if self.which_set == 'train':
            self.image_path = os.path.join(self.path, 'images', 'train2014')
            self.coco = COCO(os.path.join(self.path, 'annotations', 'instances_train2014.json'))
            
            crops_file = '{}_{}_{}_crops_instances_train2014.pkl'.format(25,self.crop[0],self.crop[1])
            with open(os.path.join(self.path, 'annotations', crops_file), 'r') as f:
                self.crops = pickle.load(f)
            self.img_ids = self.crops.keys()
            
        elif self.which_set == 'valid' or 'test':
            self.image_path = os.path.join(self.path, 'images', 'val2014')
            self.coco = COCO(os.path.join(self.path, 'annotations', 'instances_val2014.json'))
            
            crops_file = '{}_{}_{}_crops_instances_val2014.pkl'.format(25,self.crop[0],self.crop[1])
            with open(os.path.join(self.path, 'annotations', crops_file), 'r') as f:
                self.crops = pickle.load(f)
            self.img_ids = self.crops.keys()
            
            split = int(valid_split_ratio*len(self.img_ids))
            if self.which_set == "valid":
                self.img_ids = self.img_ids[:split]
            elif self.which_set == "test":
                self.img_ids = self.img_ids[split:]
                
        self.catIds = self.coco.getCatIds()
        
        self.to_mscoco = dict()
        self.to_reduce = dict()
        for ms, re in zip(self.catIds, range(len(self.catIds))):
            self.to_mscoco[re] = ms
            self.to_reduce[ms] = re
        
        self.start()
        
    def classes_names(self, ids, is_reduce=True):
        #reduce to class names
        if is_reduce: ids = self.to_mscoco_ids(ids)
        names = map(lambda x: str(x['name']), self.coco.loadCats(ids))
        return names if isinstance(ids, list) else names[0]
        
    def to_mscoco_ids(self, ids):
        if isinstance(ids, int):
            return self.to_mscoco[ids]
        
        ms_ids = []
        for i in ids:
            ms_ids.append(self.to_mscoco[i])
        return ms_ids
    
    def to_reduce_ids(self, ids):
        if isinstance(ids, int):
            return self.to_reduce[ids]
        
        re_ids = []
        for i in ids:
            re_ids.append(self.to_reduce[i])
        return re_ids
        
    def get_infos(self):
        #return self.coco.loadImgs(self.img_ids)
        imgs = self.coco.loadImgs(self.img_ids)
        #remove a corrupted image
        for img in imgs:
            if img['file_name'] == 'COCO_train2014_000000167126.jpg':
                imgs.remove(img)
                
        #Finding a good order
        flat_batch = list()
        order = dict()
        for img_id, img_info in zip(map(lambda x: x['id'], imgs), imgs):
            keys = self.crops[img_id].keys()
            img_info['location'] = keys[np.random.randint(len(keys))]
            img_info['ann_ids'] = self.crops[img_id][img_info['location']]
            
            l = len(img_info['ann_ids'])
            if l in order: order[l].append(img_info)
            else: order[l] = [img_info]
            
            if len(order[l]) == self.batch_size:
                flat_batch.extend(order[l])
                order[l] = []
        return flat_batch
    
    def load(self, img, batch_infos=None):
        if not os.path.exists('%s/%s' % (self.image_path, img['file_name'])):
            raise RuntimeError('Image %s is missing' % img['file_name'])
        
        im = Image.open('%s/%s' % (self.image_path, img['file_name'])).copy()
        if im.mode == 'L': im = im.convert('RGB')
        im = np.asarray(im)
        
        return im
    
    def preprocess(self, imgs, batch_infos):
        pimgs, categss, bboxess, segmss = [], [], [], []
        for img, info in zip(imgs, batch_infos):
            img = img/255.
            pimg = self.crop_image(img, self.crop, info['location']).transpose(2,0,1)
            cates, bboxs, segms = self.extract_anns(info)
            pimgs.append(pimg)
            categss.append(cates)
            bboxess.append(bboxs)
            segmss.append(segms)
            
        na = np.asarray
        return [na(pimgs), na(categss), na(bboxess), na(segmss), len(batch_infos[0]['ann_ids'])+3], batch_infos
    
    def crop_image(self, img, crop, location):
        h, w = img.shape[:2]
        dh, dw = crop
        x, y = location
        assert y+dh <= h and x+dw <= w, 'crop is outside the image'

        return img[y:y+dh, x:x+dw]
    
    def extract_anns(self, info):
        shape = info['height'], info['width']
        anns = self.coco.loadAnns(info['ann_ids'])
        cates, bboxs, segms = [self.get_onehot('BEG')], [], []
        for ann in anns:
            cates.append(self.get_onehot(self.to_reduce_ids(ann['category_id'])))
            bboxs.append(self.crop_image(self.get_bbox(ann['bbox'], shape), self.crop, info['location']))
            segms.append(self.crop_image(self.get_mask(ann, shape), self.crop, info['location']))
        cates.append(self.get_onehot('END'))
            
        return cates, bboxs, segms
            
    def get_onehot(self, k):
        k = 80 if k=='BEG' else 81 if k=='END' else k
        onehot = np.zeros((82,))
        onehot[k] = 1
        return onehot
    
    def get_bbox(self, bbox, shape):
        w, h, dw, dh = map(int, bbox)
        mask = np.zeros(shape)
        mask[h:h+dh, w:w+dw] = 1
        return mask
    
    def get_mask(self, ann, shape):
        h, w = shape
        mask = np.zeros(shape)
        if type(ann['segmentation']) == list:
            # polygon
            for seg in ann['segmentation']:
                # xy vertex of the polygon
                poly = np.array(seg).reshape((len(seg)/2, 2))
                closed_path = Path(poly)
                nx, ny = w, h
                x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x, y)).T
                grid = closed_path.contains_points(points)
                grid = grid.reshape((ny, nx))
                mask[grid] = 1
        else:
            # mask
            if type(ann['segmentation']['counts']) == list:
                rle = cocomask.frPyObjects([ann['segmentation']],h,w)
            else:
                rle = [ann['segmentation']]
            grid = cocomask.decode(rle)[:, :, 0]
            grid = grid.astype('bool')
            mask[grid] = 1
        return mask
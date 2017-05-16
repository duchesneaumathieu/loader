import numpy as np
import pickle
import sys, os
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from matplotlib.path import Path

if len(sys.argv) > 1 and sys.argv[1] == 'help':
    print 'python k_precrops_instances.py <k> <crops> "<# images>"'
    sys.exit(0)
    
k = int(sys.argv[1])
c = tuple(map(lambda x: int(x), sys.argv[2][1:-1].split(',')))
if len(sys.argv) > 3: i = int(sys.argv[3])
else: i = sys.maxint



def random_crop_location(im, crop):
    h, w = im['height'], im['width']
    dh, dw = crop
    x = np.random.randint(w-dw+1)
    y = np.random.randint(h-dh+1)
    return (x, y)

def crop(im, crop, location):
    h, w = im.shape[:2]
    dh, dw = crop
    x, y = location
    assert y+dh <= h and x+dw <= w, 'crop is outside the image'
    
    return im[y:y+dh, x:x+dw]

def get_mask(ann, shape):
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

def good_anns(img, c, loc, coco):
    anns = coco.loadAnns(coco.getAnnIds(img['id'], iscrowd=None))
    shape = img['height'], img['width']
    ganns = []
    for ann in anns:
        if any(crop(get_mask(ann, shape), c, loc).flatten()):
            ganns.append(ann)
    return ganns

def get_n_random_crops_length(img_info, n, c, coco):
    shape = img_info['height'], img_info['width']
    anns = coco.loadAnns(coco.getAnnIds(img_info['id'], iscrowd=None))
    locs = [random_crop_location(img_info, c) for i in range(n)]
    loc_to_length = dict([(loc, list()) for loc in locs])
    for ann in anns:
        mask = get_mask(ann, shape)
        for loc in locs:
            cmask = crop(mask, c, loc)
            if any(cmask.flatten()):
                loc_to_length[loc].append(ann['id'])
    return loc_to_length

def get_all_n_random_crops_length(img_ids, n, c, coco):
    img_infos = coco.loadImgs(img_ids)
    mapping = dict()
    for k, img_info in enumerate(img_infos):
        sys.stdout.write('\r{}/{}%     '.format(k, len(img_infos)))
        sys.stdout.flush()
        if c[0] > img_info['height'] or c[1] > img_info['width']: continue
        mapping[img_info['id']] = get_n_random_crops_length(img_info, n, c, coco)
    return mapping


here_path = os.path.dirname(os.path.realpath(__file__))
mscoco_path = os.path.join(here_path, 'datasets', 'MsCoco')
sets = ['instances_train2014.json', 'instances_val2014.json']

for s in sets:
    coco = COCO(os.path.join(mscoco_path, 'annotations', s))

    img_ids = coco.getImgIds()[:i]
        
    mapping = get_all_n_random_crops_length(img_ids, k, c, coco)
    
    del coco
    
    print 'dumping'
    dump_path = os.path.join(mscoco_path, 'annotations', '{}_{}_{}_crops_{}'.format(k,c[0],c[1],s[:-4]+'pkl'))
    with open(dump_path, 'w') as f:
        pickle.dump(mapping, f, 2)
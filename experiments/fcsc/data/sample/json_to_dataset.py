#!/usr/bin/env python

import os
import os.path as osp
import labelme
import json

import chainer_mask_rcnn as mrcnn


json_file = './000000.json'
data_id = osp.splitext(osp.basename(json_file))[0]
data = json.load(open(json_file))
print(data.keys())

img_file = './000000.jpg'
import skimage.io
img = skimage.io.imread(img_file)

print(img.shape)

lbl, label_names = labelme.utils.labelme_shapes_to_label(img.shape, data['shapes'])
print(lbl.shape)
import numpy as np
print(np.unique(lbl))
print(label_names)

lbl_cls = np.zeros(img.shape[:2], dtype=np.int32)
lbl_ins = np.zeros(img.shape[:2], dtype=np.int32)

n_instance = 0
for label_id, label_name in enumerate(label_names):
    mask_ins = lbl == label_id
    if label_name == 'background':
        lbl_ins[mask_ins] = -1
        lbl_cls[mask_ins] = 0
    else:
        if '-' in label_name:
            class_id, _ = [int(x) for x in label_name.split('-')]
        else:
            class_id = int(label_name)
        ins_id = n_instance
        lbl_cls[mask_ins] = class_id
        lbl_ins[mask_ins] = ins_id
        n_instance += 1

print('lbl_cls: ', np.unique(lbl_cls), lbl_cls.shape)
print('lbl_ins: ', np.unique(lbl_ins), lbl_ins.shape)

class_names = []
with open('./class_list.txt') as f:
    for line in f:
        class_name, _ = line.strip().split(' ')
        class_names.append(class_name)
class_names = np.asarray(class_names)

viz = mrcnn.utils.visualize_instance_segmentation(lbl_ins, lbl_cls, img, class_names)
# import mvtk
# viz = mvtk.image.tile([img, viz])
# mvtk.io.imshow(viz)
# mvtk.io.waitkey()

try:
    os.makedirs(data_id)
except:
    pass
skimage.io.imsave(osp.join(data_id, 'image.jpg'), img)
np.savez_compressed(osp.join(data_id, 'lbl_ins.npz'), lbl_ins)
np.savez_compressed(osp.join(data_id, 'lbl_cls.npz'), lbl_cls)
skimage.io.imsave(osp.join(data_id, 'label_viz.jpg'), viz)

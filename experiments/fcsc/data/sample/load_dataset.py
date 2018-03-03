#!/usr/bin/env python

import os.path as osp
import skimage.io
import numpy as np

data_id = './000000'

img = skimage.io.imread(osp.join(data_id, 'image.jpg'))
lbl_cls = np.load(osp.join(data_id, 'lbl_cls.npz'))['arr_0']
lbl_ins = np.load(osp.join(data_id, 'lbl_ins.npz'))['arr_0']

print(img.shape)
print(lbl_cls.shape, np.unique(lbl_cls))
print(lbl_ins.shape, np.unique(lbl_ins))

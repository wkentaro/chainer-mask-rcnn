#!/usr/bin/env python

import chainer
from chainer import cuda
import numpy as np
import scipy.misc

import mask_rcnn


gpu = 0

if gpu >= 0:
    cuda.get_device(gpu).use()

model = mask_rcnn.models.MaskRcnn()
if gpu >= 0:
    model.to_gpu()

img = scipy.misc.face()
img = img[:, :, ::-1]  # bgr
img = img.astype(np.float32)
mean = np.array([104.00698793, 116.66876762, 122.67891434])  # bgr
img -= mean
img = img.astype(np.float32)
x_data = img.transpose(2, 0, 1)
x_data = x_data[np.newaxis, :, :, :]
if gpu >= 0:
    x_data = cuda.to_gpu(x_data)
x = chainer.Variable(x_data)
print('x: {}'.format(x.shape))

roi_cls_locs, roi_scores, roi_masks, rois, roi_indices = model(x)
print('roi_cls_locs: {}'.format(roi_cls_locs.shape))
print('roi_scores: {}'.format(roi_scores.shape))
print('roi_masks: {}'.format(roi_masks.shape))
print('rois: {}'.format(rois.shape))  # (y1, x1, y2, x2)
print('roi_indices: {}'.format(roi_indices.shape))

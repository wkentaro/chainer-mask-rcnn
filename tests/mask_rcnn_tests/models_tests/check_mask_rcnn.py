#!/usr/bin/env python

import chainer
from chainer import cuda
import chainercv
import numpy as np

import mask_rcnn


gpu = 0

cuda.get_device(gpu).use()

model = mask_rcnn.models.MaskRCNNVGG16(n_fg_class=20)
model.to_gpu()

dataset_ins = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
dataset = mask_rcnn.datasets.MaskRcnnDataset(dataset_ins)

img, bbox, label, mask = dataset[0]
img = img.transpose(2, 0, 1)

_, H, W = img.shape
img = model.prepare(img)
_, o_H, o_W = img.shape
scale = 1. * o_H / H
bbox = chainercv.transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

imgs = np.asarray([img])
bboxes = np.asarray([bbox])
labels = np.asarray([label])

imgs = cuda.to_gpu(imgs)
bboxes = cuda.to_gpu(bboxes)
labels = cuda.to_gpu(labels)

x = chainer.Variable(imgs)
print('x: {}'.format(x.shape))

roi_cls_locs, roi_scores, roi_masks, rois, roi_indices = model(x, scale)
print('roi_cls_locs: {}'.format(roi_cls_locs.shape))
print('roi_scores: {}'.format(roi_scores.shape))
print('roi_masks: {}'.format(roi_masks.shape))
print('rois: {}'.format(rois.shape))  # (y1, x1, y2, x2)
print('roi_indices: {}'.format(roi_indices.shape))

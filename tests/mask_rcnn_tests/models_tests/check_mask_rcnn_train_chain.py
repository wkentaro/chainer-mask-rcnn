#!/usr/bin/env python

from __future__ import print_function

import chainer
from chainer import cuda
import chainercv
import cv2
import numpy as np

import mask_rcnn


# gpu
gpu = 0
if gpu >= 0:
    cuda.get_device(gpu).use()

# model
frcnn = mask_rcnn.models.MaskRCNNVGG16(
    n_fg_class=20, pretrained_model='imagenet')
# frcnn = mask_rcnn.models.MaskRCNNVGG16(
#     n_fg_class=20, pretrained_model='voc07')
model = mask_rcnn.models.MaskRCNNTrainChain(frcnn)
if gpu >= 0:
    model.to_gpu()

# optimizer
optimizer = chainer.optimizers.MomentumSGD(lr=1e-3, momentum=0.9)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

# dataset
dataset_ins = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
dataset = mask_rcnn.datasets.MaskRcnnDataset(dataset_ins)
img, bbox, label, mask = dataset[0]
img = img.transpose(2, 0, 1)
label -= 1

img_org = img.copy()

_, H, W = img.shape
img = model.mask_rcnn.prepare(img)
_, o_H, o_W = img.shape
scale = 1. * o_H / H
bbox = chainercv.transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
mask_resized = [None] * len(mask)
for i in range(len(mask)):
    if mask[i] is None:
        continue
    mask_resized[i] = cv2.resize(mask[i], None, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_NEAREST)
mask = mask_resized

imgs = np.asarray([img])
bboxes = np.asarray([bbox])
labels = np.asarray([label])
masks = np.asarray([mask])
scale = np.array([scale], dtype=np.float32)
if gpu >= 0:
    imgs = cuda.to_gpu(imgs)
    bboxes = cuda.to_gpu(bboxes)
    labels = cuda.to_gpu(labels)
    masks = cuda.to_gpu(masks)
    scale = cuda.to_gpu(scale)

# training
for i in xrange(50):
    if i % 10 == 0:
        bboxes_pred, labels_pred, scores_pred, masks_pred = \
            model.mask_rcnn.predict([img_org.copy()])

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(bboxes * scale)
        print(labels)
        print('----------------------------------------------------------')
        print(bboxes_pred)
        print(labels_pred)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        # bbox_pred = bboxes_pred[0]
        # bbox_pred = bbox_pred[:, [1, 0, 3, 2]]
        # label_pred = labels_pred[0]
        # viz = img_org.transpose(1, 2, 0).astype(np.uint8)
        # viz = mask_rcnn.utils.draw_instance_boxes(
        #     viz, bbox_pred, label_pred, n_class=20, bg_class=-1)
        # cv2.imshow('viz', viz[:, :, ::-1])
        # cv2.waitKey(0)

    model.zerograds()
    loss = model(imgs, bboxes, labels, masks, scale)
    print('[%02d] Loss: %s' % (i, loss))
    loss.backward()
    optimizer.update()

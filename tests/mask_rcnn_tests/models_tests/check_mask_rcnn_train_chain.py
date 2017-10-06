#!/usr/bin/env python

import pprint

import chainer
from chainer import cuda
import fcn
import numpy as np

import mask_rcnn


gpu = 0

if gpu >= 0:
    cuda.get_device(gpu).use()

chainer.config.train = True
chainer.config.enable_backprop = True

optimizer = chainer.optimizers.Adam(alpha=1e-2)

mrcnn = mask_rcnn.models.MaskRcnn()
model = mask_rcnn.models.MaskRcnnTrainChain(mrcnn)
if gpu >= 0:
    model.to_gpu()
optimizer.setup(model)

dataset_ins = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
dataset = mask_rcnn.datasets.MaskRcnnDataset(dataset_ins)
img, bboxes, labels, masks, scale = dataset[0]
img_org = img
print(img.shape)
print(bboxes.shape)
print(labels.shape)
print(masks.shape)
print(scale)

img, = fcn.datasets.transform_lsvrc2012_vgg16((img,))
img = np.asarray([img])
bboxes = np.asarray([bboxes])
labels = np.asarray([labels])
masks = np.asarray([masks])
scale = np.asarray([scale])
if gpu >= 0:
    img = cuda.to_gpu(img)
    bboxes = cuda.to_gpu(bboxes)
    labels = cuda.to_gpu(labels)
    masks = cuda.to_gpu(masks)
    scale = cuda.to_gpu(scale)

reporter = chainer.Reporter()
reporter.add_observer('main', model)
reporter.add_observers('main', model.namedlinks(skipself=True))

for i in xrange(10000):
    model.zerograds()
    observation = {}
    with reporter.scope(observation):
        loss = model(img, bboxes, labels, masks, scale)
    print(i)
    pprint.pprint(observation)
    loss.backward()
    optimizer.update()

    if i % 10 == 0:
        print(bboxes, labels)
        bboxes_pred, labels_pred, scores_pred, masks_pred = \
            model.mask_rcnn.predict([img_org])
        print(bboxes_pred, labels_pred)

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

frcnn = mask_rcnn.models.faster_rcnn.FasterRCNNVGG16(
    n_fg_class=20, pretrained_model='imagenet')
# frcnn = mask_rcnn.models.faster_rcnn.FasterRCNNVGG16(pretrained_model='voc07')
model = mask_rcnn.models.faster_rcnn.FasterRCNNTrainChain(frcnn)
if gpu >= 0:
    model.to_gpu()

optimizer = chainer.optimizers.MomentumSGD(lr=1e-3, momentum=0.9)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

dataset_ins = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
dataset = mask_rcnn.datasets.MaskRcnnDataset(dataset_ins)
img, bboxes, labels, _, scale = dataset[0]
img_org = img.copy().transpose(2, 0, 1)
print(img.shape)
print(bboxes.shape)
print(labels.shape)
print(scale)

img, = fcn.datasets.transform_lsvrc2012_vgg16((img,))
img = np.asarray([img])
bboxes = np.asarray([bboxes])
labels = np.asarray([labels])
scale = np.asarray([scale])
if gpu >= 0:
    img = cuda.to_gpu(img)
    bboxes = cuda.to_gpu(bboxes)
    labels = cuda.to_gpu(labels)
    scale = cuda.to_gpu(scale)

reporter = chainer.Reporter()
reporter.add_observer('main', model)
reporter.add_observers('main', model.namedlinks(skipself=True))

for i in xrange(10000):
    model.zerograds()
    observation = {}
    with reporter.scope(observation):
        loss = model(img, bboxes, labels, scale)
    print(i)
    pprint.pprint(observation)
    loss.backward()
    optimizer.update()

    if i % 10 == 0:
        print(bboxes, labels)
        bboxes_pred, labels_pred, scores_pred = \
            model.faster_rcnn.predict([img_org])
        print(bboxes_pred, labels_pred)

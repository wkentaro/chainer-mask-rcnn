#!/usr/bin/env python

import chainer

import mask_rcnn


pretrained_model = 'logs/faster_rcnn_cfg001_20170907_020125/model_snapshot_iter_00146400.npz'  # NOQA
model = mask_rcnn.models.faster_rcnn.FasterRCNNVGG16(
    n_fg_class=20,
    pretrained_model=pretrained_model,
)

gpu = 0
if gpu >= 0:
    chainer.cuda.get_device_from_id(gpu).use()
    model.to_gpu()

dataset = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
img, lbl_cls, lbl_ins = dataset.get_example(0)
import skimage.io
skimage.io.imsave('tmp.jpg', img)
img = img.transpose((2, 0, 1))
bboxes, labels, scores = model.predict([img])

for bbox, label, score in zip(bboxes, labels, scores):
    print(bbox)
    print(label)
    print(score)

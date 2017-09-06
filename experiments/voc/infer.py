#!/usr/bin/env python

import chainer

import mask_rcnn


pretrained_model = 'logs/cfg001_20170906_224832/model_snapshot_iter_00014640.npz'  # NOQA
model = mask_rcnn.models.MaskRcnn()
chainer.serializers.load_npz(pretrained_model, model)

gpu = 0
if gpu >= 0:
    chainer.cuda.get_device_from_id(gpu).use()
    model.to_gpu()

dataset = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
img, lbl_cls, lbl_ins = dataset.get_example(0)
bboxes, labels, scores, masks = model.predict([img])

for bbox, label, score, mask in zip(bboxes, labels, scores, masks):
    print(bbox)
    print(label)
    print(score)

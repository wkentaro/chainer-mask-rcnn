#!/usr/bin/env python

import chainer

import mask_rcnn



pretrained_model = 'logs/cfg001_20170906_170003/model_snapshot_iter_00021960.npz'  # NOQA
model = mask_rcnn.models.MaskRcnn()
train_chain = mask_rcnn.models.MaskRcnnTrainChain(mask_rcnn=model)
chainer.serializers.load_npz(pretrained_model, train_chain)
model = train_chain.mask_rcnn

gpu = 0
if gpu >= 0:
    chainer.cuda.get_device_from_id(gpu).use()
    model.to_gpu()

dataset = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
img, lbl_cls, lbl_ins = dataset.get_example(0)
bboxes, labels, scores, masks = model.predict([img])

#!/usr/bin/env python

from __future__ import print_function

import os
import os.path as osp

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
if False:
    frcnn = mask_rcnn.models.MaskRCNNVGG16(
        n_fg_class=20, pretrained_model='imagenet')
else:
    frcnn = mask_rcnn.models.MaskRCNNVGG16(
        n_fg_class=20, pretrained_model='voc0712_faster_rcnn')
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

out = 'logs/check_mask_rcnn_train_chain'
if not osp.exists(out):
    os.makedirs(out)

# training
for i in xrange(5000 * 20):
    idx = np.random.randint(0, len(dataset))
    img, bbox, label, mask = dataset[idx]
    img = img.transpose(2, 0, 1)
    label -= 1

    img_org = img.copy()

    _, H, W = img.shape
    img = model.mask_rcnn.prepare(img)
    _, o_H, o_W = img.shape
    scale = 1. * o_H / H
    bbox = chainercv.transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
    mask_resized = [None] * len(mask)
    for i_m in range(len(mask)):
        if mask[i_m] is None:
            continue
        mask_resized[i_m] = cv2.resize(
            mask[i_m], None, None, fx=scale, fy=scale,
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

    if i % 5000 == 0:
        file_model = osp.join(out, 'model_%08d.npz' % i)
        print('Saving snapshot model: %s' % file_model)
        chainer.serializers.save_npz(file_model, model.mask_rcnn)

    if i % 10 == 0:
        bboxes_pred, labels_pred, scores_pred, masks_pred, rois_pred = \
            model.mask_rcnn.predict([img_org.copy()])

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(bboxes * scale)
        print(labels)
        print('----------------------------------------------------------')
        print(bboxes_pred)
        print(labels_pred)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        # visualize
        import fcn
        bbox_pred = bboxes_pred[0]
        roi_pred = rois_pred[0]
        label_pred = labels_pred[0]
        mask_pred = masks_pred[0]
        viz = img_org.transpose(1, 2, 0).astype(np.uint8)
        viz = np.ascontiguousarray(viz)
        cmap = fcn.utils.label_colormap(20)
        cmap = (cmap * 255).astype(np.uint8)
        for j in range(len(bbox_pred)):
            y1, x1, y2, x2 = map(int, roi_pred[j])
            H_roi, W_roi = y2 - y1, x2 - x1
            m = mask_pred[j]
            m = cv2.resize(m, (W_roi, H_roi))
            m = m >= 0.5
            mask_ins = np.zeros(viz.shape[:2], dtype=bool)
            mask_ins[y1:y2, x1:x2] = m

            color = cmap[label_pred[j]]
            y1, x1, y2, x2 = map(int, bbox_pred[j])
            mask_ins_bbox = np.zeros(viz.shape[:2], dtype=bool)
            mask_ins_bbox[y1:y2, x1:x2] = True
            mask_ins = mask_ins & mask_ins_bbox
            cv2.rectangle(viz, (x1, y1), (x2, y2), color=(0, 0, 0))
            viz[mask_ins] = viz[mask_ins] * 0.5 + color * 0.5
            viz = viz.astype(np.uint8)
        cv2.imwrite(osp.join(out, '%08d.jpg' % i), viz[:, :, ::-1])
        cv2.imwrite(osp.join(out, 'latest.jpg'), viz[:, :, ::-1])

    model.zerograds()
    loss = model(imgs, bboxes, labels, masks, scale)
    print('[%02d] Loss: %s' % (i, loss))
    loss.backward()
    optimizer.update()

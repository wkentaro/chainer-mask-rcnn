#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import os.path as osp

import chainer
from chainer import cuda
import chainercv
import cv2
import numpy as np

import mask_rcnn
import mvtk


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument(
        '-o', '--out',
        default=osp.join(here, 'logs/check_mask_rcnn_train_chain'))
    args = parser.parse_args()

    gpu = args.gpu
    overfit_test = args.overfit
    out = args.out

    # gpu
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
    dataset_ins_val = mask_rcnn.datasets.VOC2012InstanceSeg(split='val')

    # output dir
    if not osp.exists(out):
        os.makedirs(out)

    # training loop
    for i in xrange(5000 * 20):
        if overfit_test:
            idx = np.random.randint(0, 1)
        else:
            idx = np.random.randint(0, len(dataset))
        img, bbox, label, mask = dataset[idx]
        img = img.transpose(2, 0, 1)

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

        # snapshot
        if not overfit_test and i % 5000 == 0:
            file_model = osp.join(out, 'model_%08d.npz' % i)
            print('Saving snapshot model: %s' % file_model)
            chainer.serializers.save_npz(file_model, model.mask_rcnn)

        # validation
        if i % 50 == 0:
            if overfit_test:
                img, lbl_cls_true, lbl_ins_true = dataset_ins[idx]
            else:
                idx_val = np.random.randint(0, len(dataset_ins_val))
                img, lbl_cls_true, lbl_ins_true = dataset_ins_val[idx_val]

            img_chw = img.copy().transpose(2, 0, 1)
            lbl_inss, lbl_clss = model.mask_rcnn.predict([img_chw])
            lbl_ins = lbl_inss[0]
            lbl_cls = lbl_clss[0]

            label_true, bbox_true, mask_true = \
                mask_rcnn.utils.label2instance_boxes(
                    lbl_ins_true, lbl_cls_true, return_masks=True)
            label_true -= 1
            label, bbox, mask = mask_rcnn.utils.label2instance_boxes(
                lbl_ins, lbl_cls, return_masks=True)
            label -= 1
            score = np.ones((len(mask),), dtype=np.float64)

            prec, rec = mask_rcnn.utils.calc_instseg_voc_prec_rec(
                [mask], [label], [score], [mask_true], [label_true])
            ap = chainercv.evaluations.calc_detection_voc_ap(prec, rec)
            mean_ap = 100 * np.nanmean(ap)
            print('[%02d] map: %.2f' % (i, mean_ap))

            viz_true = mask_rcnn.utils.visualize_instance_segmentation(
                lbl_ins_true, lbl_cls_true, img, dataset_ins.class_names)
            viz_pred = mask_rcnn.utils.visualize_instance_segmentation(
                lbl_ins, lbl_cls, img, dataset_ins.class_names)
            viz = mvtk.image.tile([viz_true, viz_pred], shape=(2, 1))
            cv2.imwrite(osp.join(out, '%08d.map=%.1f.jpg' % (i, mean_ap)),
                        viz[:, :, ::-1])
            cv2.imwrite(osp.join(out, 'latest.jpg'), viz[:, :, ::-1])

        # train
        model.zerograds()
        loss = model(imgs, bboxes, labels, masks, scale)
        print('[%02d] Loss: %s' % (i, loss))
        loss.backward()
        optimizer.update()


if __name__ == '__main__':
    main()

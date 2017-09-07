#!/usr/bin/env python

from __future__ import division

import argparse
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainercv import transforms

import mask_rcnn


class Transform(object):

    def __init__(self, faster_rcnn):
        self.faster_rcnn = faster_rcnn

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class FasterRcnnDataset(chainer.dataset.DatasetMixin):

    def __init__(self, instance_dataset):
        self._instance_dataset = instance_dataset

    def __len__(self):
        return len(self._instance_dataset)

    def get_example(self, i):
        from mask_rcnn.utils import label2instance_boxes
        img, lbl_cls, lbl_ins = self._instance_dataset.get_example(i)
        img = img.transpose((2, 0, 1)).astype(np.float32, copy=False)
        labels, bboxes = label2instance_boxes(lbl_ins, lbl_cls)
        labels = labels.astype(np.int32, copy=False)
        bboxes = bboxes[:, [1, 0, 3, 2]]  # xy -> yx
        bboxes = bboxes.astype(np.float32, copy=False)
        return img, bboxes, labels


def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV training example: Faster R-CNN')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--out', '-o', default='logs/faster_rcnn',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=50000)
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1. dataset

    faster_rcnn = mask_rcnn.models.faster_rcnn.FasterRCNNVGG16(
        n_fg_class=20, pretrained_model='imagenet')
    faster_rcnn.use_preset('evaluate')

    train_data = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
    train_data = FasterRcnnDataset(train_data)
    train_data = chainer.datasets.TransformDataset(
        train_data, Transform(faster_rcnn))
    # test_data = mask_rcnn.datasets.VOC2012InstanceSeg(split='val')
    # test_data = FasterRcnnDataset(test_data)
    # test_data = chainer.datasets.TransformDataset(
    #     test_data, Transform(faster_rcnn))

    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size=1, n_processes=None, shared_mem=100000000)
    # test_iter = chainer.iterators.SerialIterator(
    #     test_data, batch_size=1, repeat=False, shuffle=False)

    # 2. model

    model = mask_rcnn.models.faster_rcnn.FasterRCNNTrainChain(faster_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # 3. optimizer

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    # trainer

    trainer = training.Trainer(
        updater, (args.iteration, 'iteration'), out=args.out)

    trainer.extend(
        extensions.snapshot_object(model.faster_rcnn, 'snapshot_model.npz'),
        trigger=(args.iteration, 'iteration'))
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.step_size, 'iteration'))

    log_interval = 20, 'iteration'
    plot_interval = 1000, 'iteration'
    print_interval = 20, 'iteration'

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(
        trigger=log_interval, log_name='log.json'))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/roi_loc_loss',
         'main/roi_cls_loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'validation/main/map',
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )

    # trainer.extend(
    #     DetectionVOCEvaluator(
    #         test_iter, model.faster_rcnn, use_07_metric=True,
    #         label_names=voc_detection_label_names),
    #     trigger=ManualScheduleTrigger(
    #         [args.step_size, args.iteration], 'iteration'))

    # trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()

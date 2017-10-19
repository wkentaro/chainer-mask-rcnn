#!/usr/bin/env python

from __future__ import division

import argparse
import datetime
import os
import os.path as osp
import pprint

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNResNet101
from chainercv.links import FasterRCNNResNet50
from chainercv.links import FasterRCNNVGG16
from chainercv import transforms
import numpy as np

import mask_rcnn as mrcnn


class Transform(object):

    def __init__(self, faster_rcnn, augmentation=True):
        self.faster_rcnn = faster_rcnn
        self._augmentation = augmentation

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

        if self._augmentation:
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
        img, lbl_cls, lbl_ins = self._instance_dataset.get_example(i)
        img = img.transpose(2, 0, 1)
        labels, bboxes, _ = mrcnn.utils.label2instance_boxes(
            lbl_ins, lbl_cls, return_masks=True)
        labels = labels.astype(np.int32, copy=False)
        labels -= 1  # background: 0 -> -1
        bboxes = bboxes.astype(np.float32, copy=False)
        return img, bboxes, labels


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', choices=('vgg16', 'resnet50', 'resnet101'),
                        help='The model to use', default='resnet50')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=50000)
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    args = parser.parse_args()

    args.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.out = osp.join(
        here, 'logs',
        '.'.join([
            'model={model}',
            'lr={lr}',
            'seed={seed}',
            'step_size={step_size}',
            'iteration={iteration}',
            'weight_decay={weight_decay}',
            'timestamp={timestamp}',
        ]).format(**args.__dict__)
    )

    pprint.pprint(args.__dict__)

    np.random.seed(args.seed)

    train_data = FasterRcnnDataset(
        mrcnn.datasets.VOC2012InstanceSeg(split='train'))
    test_data = FasterRcnnDataset(
        mrcnn.datasets.VOC2012InstanceSeg(split='val'))

    if args.model == 'vgg16':
        faster_rcnn = FasterRCNNVGG16(n_fg_class=len(voc_bbox_label_names),
                                      pretrained_model='imagenet',
                                      min_size=800, max_size=9999)
    elif args.model == 'resnet50':
        faster_rcnn = FasterRCNNResNet50(n_fg_class=len(voc_bbox_label_names),
                                         pretrained_model='imagenet',
                                         min_size=800, max_size=9999)
    elif args.model == 'resnet101':
        faster_rcnn = FasterRCNNResNet101(n_fg_class=len(voc_bbox_label_names),
                                          pretrained_model='imagenet',
                                          min_size=800, max_size=9999)
    else:
        raise ValueError

    model = mrcnn.models.RPNTrainChain(faster_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))

    # This is only relevant to ResNet training.
    if args.model in ['resnet50', 'resnet101']:
        for p in model.params():
            # Do not update batch normalization layers.
            if p.name == 'gamma':
                p.update_rule.enabled = False
            elif p.name == 'beta':
                p.update_rule.enabled = False

        # Do not update for the first two blocks.
        faster_rcnn.extractor.conv1.disable_update()
        faster_rcnn.extractor.bn1.disable_update()
        faster_rcnn.extractor.res2.disable_update()

    train_data = TransformDataset(train_data, Transform(faster_rcnn))
    test_data = TransformDataset(test_data,
                                 Transform(faster_rcnn, augmentation=False))

    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size=1, n_processes=None, shared_mem=100000000)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(
        updater, (args.iteration, 'iteration'), out=args.out)

    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.step_size, 'iteration'))

    eval_interval = 3000, 'iteration'
    log_interval = 20, 'iteration'
    plot_interval = 3000, 'iteration'
    print_interval = 20, 'iteration'

    trainer.extend(extensions.Evaluator(test_iter, model),
                   trigger=eval_interval)
    trainer.extend(
        extensions.snapshot_object(model.faster_rcnn, 'snapshot_model.npz'),
        trigger=training.triggers.MinValueTrigger(
            'validation/main/loss', eval_interval))
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'validation/main/loss',
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss',
                 'main/rpn_loc_loss',
                 'main/rpn_cls_loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()

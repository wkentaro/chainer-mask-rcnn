#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import os.path as osp
import random
import socket
import sys

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import cv2  # NOQA

import chainer
from chainer import training
from chainer.training import extensions
import fcn
import numpy as np

import chainer_mask_rcnn as mrcnn


here = osp.dirname(osp.abspath(__file__))


class MaskRCNNDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        example = self.dataset.get_example(i)
        img, bboxes, labels, masks = example[:4]

        masks = masks.astype(np.int32, copy=False)
        labels = labels.astype(np.int32, copy=False)
        bboxes = bboxes.astype(np.float32, copy=False)

        example = list(example)
        example[:4] = img, bboxes, labels, masks
        return tuple(example)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m',
                        choices=['vgg16', 'resnet50', 'resnet101'],
                        default='resnet50', help='base model')
    parser.add_argument('--pooling-func', '-p',
                        choices=['pooling', 'align', 'resize'],
                        default='align', help='pooling function')
    parser.add_argument('--gpu', '-g', type=int, help='GPU id.')
    parser.add_argument('--multi-node', '-n', action='store_true',
                        help='use multi node')
    parser.add_argument('--roi-size', '-r', type=int, default=7,
                        help='roi size')
    args = parser.parse_args()

    if args.multi_node:
        import chainermn
        comm = chainermn.create_communicator('hierarchical')
        device = comm.intra_rank

        args.n_node = comm.inter_size
        args.n_gpu = comm.size
        chainer.cuda.get_device_from_id(device).use()
    else:
        if args.gpu is None:
            print('Option --gpu is required without --multi-node.',
                  file=sys.stderr)
            quit(1)
        args.n_node = 1
        args.n_gpu = 1
        chainer.cuda.get_device_from_id(args.gpu).use()
        device = args.gpu

    args.seed = 0
    now = datetime.datetime.now()
    args.timestamp = now.isoformat()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S'))

    # 0.00125 * 8 = 0.01  in original
    args.batch_size = 1 * args.n_gpu
    args.lr = 0.00125 * args.batch_size
    args.weight_decay = 0.0001

    # (180e3 * 8) / len(coco_trainval)
    args.max_epoch = (180e3 * 8) / 118287
    # lr / 10 at 120k iteration with
    # 160k iteration * 16 batchsize in original
    args.step_size = [(120e3 / 180e3) * args.max_epoch,
                      (160e3 / 180e3) * args.max_epoch]

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.dataset = 'coco'
    train_data = chainer.datasets.ConcatenatedDataset(
        mrcnn.datasets.COCOInstanceSegmentationDataset('train'),
        mrcnn.datasets.COCOInstanceSegmentationDataset('valminusminival'),
    )
    test_data = mrcnn.datasets.COCOInstanceSegmentationDataset(
        'minival', use_crowd=True, return_crowd=True, return_area=True)
    class_names = test_data.class_names

    train_data = MaskRCNNDataset(train_data)
    test_data = MaskRCNNDataset(test_data)

    if args.pooling_func == 'align':
        pooling_func = mrcnn.functions.roi_align_2d
    elif args.pooling_func == 'pooling':
        pooling_func = chainer.functions.roi_pooling_2d
    elif args.pooling_func == 'resize':
        pooling_func = mrcnn.functions.crop_and_resize
    else:
        raise ValueError

    min_size = 800
    max_size = 1333
    anchor_scales = (2, 4, 8, 16, 32)

    if args.model == 'vgg16':
        mask_rcnn = mrcnn.models.MaskRCNNVGG16(
            n_fg_class=len(class_names),
            pretrained_model='imagenet',
            pooling_func=pooling_func,
            anchor_scales=anchor_scales,
            min_size=min_size,
            max_size=max_size,
            roi_size=args.roi_size,
        )
    elif args.model in ['resnet50', 'resnet101']:
        n_layers = int(args.model.lstrip('resnet'))
        mask_rcnn = mrcnn.models.MaskRCNNResNet(
            n_layers=n_layers,
            n_fg_class=len(class_names),
            pretrained_model='imagenet',
            pooling_func=pooling_func,
            anchor_scales=anchor_scales,
            min_size=min_size,
            max_size=max_size,
            roi_size=args.roi_size,
        )
    else:
        raise ValueError
    model = mrcnn.models.MaskRCNNTrainChain(mask_rcnn)
    if args.multi_node or args.gpu >= 0:
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    if args.multi_node:
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))

    if args.model in ['resnet50', 'resnet101']:
        model.mask_rcnn.extractor.mode = 'res3+'
        mask_rcnn.extractor.conv1.disable_update()
        mask_rcnn.extractor.bn1.disable_update()
        mask_rcnn.extractor.res2.disable_update()

    train_data = chainer.datasets.TransformDataset(
        train_data, mrcnn.datasets.MaskRCNNTransform(mask_rcnn))
    test_data = chainer.datasets.TransformDataset(
        test_data, mrcnn.datasets.MaskRCNNTransform(mask_rcnn, train=False))
    if args.multi_node:
        if comm.rank != 0:
            train_data = None
            test_data = None
        train_data = chainermn.scatter_dataset(train_data, comm, shuffle=True)
        test_data = chainermn.scatter_dataset(test_data, comm)

    # FIXME: MultiProcessIterator sometimes hangs
    train_iter = chainer.iterators.SerialIterator(
        train_data, batch_size=1)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)

    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=device,
        converter=mrcnn.datasets.concat_examples)

    trainer = training.Trainer(
        updater, (args.max_epoch, 'epoch'), out=args.out)

    trainer.extend(
        extensions.ExponentialShift('lr', 0.1),
        trigger=training.triggers.ManualScheduleTrigger(
            args.step_size, 'epoch'))

    eval_interval = 1, 'epoch'
    log_interval = 20, 'iteration'
    plot_interval = 0.1, 'epoch'
    print_interval = 20, 'iteration'

    evaluator = mrcnn.extensions.InstanceSegmentationCOCOEvaluator(
        test_iter, model.mask_rcnn, device=device, label_names=class_names)
    if args.multi_node:
        evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=eval_interval)

    if not args.multi_node or comm.rank == 0:
        trainer.extend(
            extensions.snapshot_object(
                model.mask_rcnn, 'snapshot_model.npz'),
            trigger=training.triggers.MaxValueTrigger(
                'validation/main/map', eval_interval))

        args.git_hash = mrcnn.utils.git_hash()
        args.hostname = socket.gethostname()
        trainer.extend(fcn.extensions.ParamsReport(args.__dict__))
        trainer.extend(
            mrcnn.extensions.InstanceSegmentationVisReport(
                test_iter, model.mask_rcnn,
                label_names=class_names),
            trigger=eval_interval)
        trainer.extend(chainer.training.extensions.observe_lr(),
                       trigger=log_interval)
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.PrintReport(
            ['iteration', 'epoch', 'elapsed_time', 'lr',
             'main/loss',
             'main/roi_loc_loss',
             'main/roi_cls_loss',
             'main/roi_mask_loss',
             'main/rpn_loc_loss',
             'main/rpn_cls_loss',
             'validation/main/map']), trigger=print_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

        # plot
        assert extensions.PlotReport.available()
        trainer.extend(
            extensions.PlotReport(
                ['main/loss',
                 'main/roi_loc_loss',
                 'main/roi_cls_loss',
                 'main/roi_mask_loss',
                 'main/rpn_loc_loss',
                 'main/rpn_cls_loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval,
        )
        trainer.extend(
            extensions.PlotReport(
                ['validation/main/map'],
                file_name='accuracy.png', trigger=plot_interval
            ),
            trigger=eval_interval,
        )

        trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()

#!/usr/bin/env python

from __future__ import division

import argparse
import datetime
import os
import os.path as osp
import random
import socket

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
from chainer import training
from chainer.training import extensions
import fcn
import numpy as np

import chainer_mask_rcnn as mrcnn


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d',
                        choices=['voc', 'coco'],
                        default='voc', help='The dataset.')
    parser.add_argument('--model', '-m',
                        choices=['vgg16', 'resnet50', 'resnet101'],
                        default='resnet101', help='Base model of Mask R-CNN.')
    parser.add_argument('--pooling-func', '-pf',
                        choices=['pooling', 'align', 'resize'],
                        default='pooling', help='Pooling function.')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU id.')
    args = parser.parse_args()

    args.seed = 0
    now = datetime.datetime.now()
    args.timestamp = now.isoformat()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S'))

    # 0.00125 * 2 * 8 = 0.02  in original
    args.batch_size = 1
    args.lr = 0.00125 * args.batch_size
    args.weight_decay = 0.0001

    args.max_epoch = 19  # (160e3 * 16) / len(coco_trainval)
    # lr / 10 at 120k iteration with
    # 160k iteration * 16 batchsize in original
    args.step_size = (120e3 / 160e3) * args.max_epoch

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'voc':
        train_data = mrcnn.datasets.SBDInstanceSeg('train')
        test_data = mrcnn.datasets.VOC2012InstanceSeg('val')
    elif args.dataset == 'coco':
        train_data = chainer.datasets.ConcatenatedDataset(
            mrcnn.datasets.CocoInstanceSeg('train'),
            mrcnn.datasets.CocoInstanceSeg('valminusminival'),
        )
        test_data = mrcnn.datasets.CocoInstanceSeg('minival')
        train_data.class_names = test_data.class_names
    else:
        raise ValueError
    instance_class_names = train_data.class_names[1:]
    train_data = mrcnn.datasets.MaskRcnnDataset(train_data)
    test_data = mrcnn.datasets.MaskRcnnDataset(test_data)

    if args.pooling_func == 'align':
        pooling_func = mrcnn.functions.roi_align_2d
    elif args.pooling_func == 'pooling':
        pooling_func = chainer.functions.roi_pooling_2d
    elif args.pooling_func == 'resize':
        pooling_func = mrcnn.functions.crop_and_resize
    else:
        raise ValueError

    if args.model == 'vgg16':
        mask_rcnn = mrcnn.models.MaskRCNNVGG16(
            n_fg_class=len(instance_class_names),
            pretrained_model='imagenet',
            pooling_func=pooling_func)
    elif args.model in ['resnet50', 'resnet101']:
        n_layers = int(args.model.lstrip('resnet'))
        mask_rcnn = mrcnn.models.MaskRCNNResNet(
            n_layers=n_layers,
            n_fg_class=len(instance_class_names),
            pretrained_model='imagenet',
            pooling_func=pooling_func)
    else:
        raise ValueError
    mask_rcnn.use_preset('evaluate')
    model = mrcnn.models.MaskRCNNTrainChain(mask_rcnn)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
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

    train_iter = chainer.iterators.SerialIterator(
        train_data, batch_size=1)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)

    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, device=args.gpu,
        converter=mrcnn.datasets.concat_examples)

    trainer = training.Trainer(
        updater, (args.max_epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.step_size, 'epoch'))

    eval_interval = 1, 'epoch'
    log_interval = 20, 'iteration'
    plot_interval = 0.1, 'epoch'
    print_interval = 20, 'iteration'

    trainer.extend(
        extensions.snapshot_object(model.mask_rcnn, 'snapshot_model.npz'),
        trigger=training.triggers.MaxValueTrigger(
            'validation/main/map', eval_interval))

    trainer.extend(
        mrcnn.extensions.InstanceSegmentationVOCEvaluator(
            test_iter, model.mask_rcnn, use_07_metric=True,
            label_names=instance_class_names),
        trigger=eval_interval)

    args.git_hash = mrcnn.utils.git_hash()
    args.hostname = socket.gethostname()
    trainer.extend(fcn.extensions.ParamsReport(args.__dict__))

    trainer.extend(
        mrcnn.extensions.InstanceSegmentationVisReport(
            test_iter, model.mask_rcnn,
            label_names=instance_class_names),
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
         'validation/main/map',
         ]), trigger=print_interval)
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
        trigger=plot_interval
    )
    trainer.extend(
        extensions.PlotReport(
            ['validation/main/map'],
            file_name='accuracy.png', trigger=plot_interval
        ),
        trigger=plot_interval
    )

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()

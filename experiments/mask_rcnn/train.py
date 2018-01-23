#!/usr/bin/env python

from __future__ import division

import argparse
import copy
import datetime
import multiprocessing
import os
import os.path as osp
import random
import shutil
import subprocess
import sys
import yaml

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
from chainer.datasets import TransformDataset
from chainer import reporter
from chainer import training
from chainer.training import extensions
from chainercv import transforms
from chainercv.utils import apply_prediction_to_iterator
import chainermn
import cupy
import cv2
import numpy as np
import six

import chainer_mask_rcnn as mrcnn
import mvtk


def flip_image(image, x_flip=False, y_flip=False):
    # image has tensor size of (C, H, W)
    if y_flip:
        image = image[:, ::-1, :]
    if x_flip:
        image = image[:, :, ::-1]
    return image


class Transform(object):

    def __init__(self, mask_rcnn, train=True):
        self.mask_rcnn = mask_rcnn
        self.train = train

    def __call__(self, in_data):
        img, bbox, label, mask = in_data
        img = img.transpose(2, 0, 1)  # H, W, C -> C, H, W
        _, H, W = img.shape
        img = self.mask_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        if len(bbox) > 0:
            bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
        if len(mask) > 0:
            mask = transforms.resize(
                mask, size=(o_H, o_W), interpolation=0)

        if self.train:
            # horizontally flip
            img, params = transforms.random_flip(
                img, x_random=True, return_param=True)
            bbox = transforms.flip_bbox(
                bbox, (o_H, o_W), x_flip=params['x_flip'])
            if mask.ndim == 2:
                mask = flip_image(mask[None, :, :], x_flip=params['x_flip'])[0]
            else:
                mask = flip_image(mask, x_flip=params['x_flip'])

        return img, bbox, label, mask, scale


class OverfitDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset, indices=0):
        self._dataset = dataset

        if isinstance(indices, int):
            indices = [indices]
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def get_example(self, i):
        index = self._indices[i]
        return self._dataset.get_example(index)


class InstanceSegmentationVisReport(chainer.training.extensions.Evaluator):

    def __init__(self, iterator, target, label_names,
                 file_name='visualizations/iteration=%08d.jpg',
                 shape=(3, 3), copy_latest=True):
        super(InstanceSegmentationVisReport, self).__init__(iterator, target)
        self.label_names = np.asarray(label_names)
        self.file_name = file_name
        self._shape = shape
        self._copy_latest = copy_latest

    def __call__(self, trainer):
        iterator = self._iterators['main']
        target = self._targets['main']

        target.use_preset('visualize')

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            target.predict_masks, it)

        pred_bboxes, pred_masks, pred_labels, pred_scores = pred_values

        if len(gt_values) == 4:
            gt_bboxes, gt_labels, gt_masks, _ = gt_values
        elif len(gt_values) == 3:
            gt_bboxes, gt_labels, gt_masks = gt_values

        # visualize
        vizs = []
        for img, gt_bbox, gt_label, gt_mask, \
            pred_bbox, pred_label, pred_mask, pred_score \
                in six.moves.zip(imgs, gt_bboxes, gt_labels, gt_masks,
                                 pred_bboxes, pred_labels, pred_masks,
                                 pred_scores):
            # organize input
            img = img.transpose(1, 2, 0)  # CHW -> HWC
            gt_mask = gt_mask.astype(bool)

            n_fg_class = len(self.label_names)

            gt_viz = mrcnn.utils.draw_instance_boxes(
                img, gt_bbox, gt_label, n_class=n_fg_class,
                masks=gt_mask, captions=self.label_names[gt_label],
                bg_class=-1)

            captions = []
            for p_score, l_name in zip(pred_score,
                                       self.label_names[pred_label]):
                caption = '{:s} {:.1%}'.format(l_name, p_score)
                captions.append(caption)
            pred_viz = mrcnn.utils.draw_instance_boxes(
                img, pred_bbox, pred_label, n_class=n_fg_class,
                masks=pred_mask, captions=captions, bg_class=-1)

            viz = np.vstack([gt_viz, pred_viz])
            vizs.append(viz)
            if len(vizs) >= (self._shape[0] * self._shape[1]):
                break

        viz = mvtk.image.tile(vizs, shape=self._shape)
        file_name = osp.join(
            trainer.out, self.file_name % trainer.updater.iteration)
        try:
            os.makedirs(osp.dirname(file_name))
        except OSError:
            pass
        cv2.imwrite(file_name, viz[:, :, ::-1])

        if self._copy_latest:
            shutil.copy(file_name,
                        osp.join(osp.dirname(file_name), 'latest.jpg'))

        target.use_preset('evaluate')


class InstanceSegmentationVOCEvaluator(chainer.training.extensions.Evaluator):

    def __init__(self, iterator, target, device=None,
                 use_07_metric=False, label_names=None):
        super(InstanceSegmentationVOCEvaluator, self).__init__(
            iterator=iterator, target=target, device=device)
        self.use_07_metric = use_07_metric
        self.label_names = label_names

    def __call__(self, trainer=None):
        return super(InstanceSegmentationVOCEvaluator, self).__call__(trainer)

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            target.predict_masks, it)

        pred_bboxes, pred_masks, pred_labels, pred_scores = pred_values

        if len(gt_values) == 4:
            gt_bboxes, gt_labels, gt_masks, gt_difficults = gt_values
        elif len(gt_values) == 3:
            gt_bboxes, gt_labels, gt_masks = gt_values
            gt_difficults = None

        # evaluate
        result = mrcnn.utils.evaluations.eval_instseg_voc(
            pred_masks, pred_labels, pred_scores,
            gt_masks, gt_labels, gt_difficults,
            use_07_metric=self.use_07_metric)

        report = {'map': result['map']}

        if self.label_names is not None:
            for l, label_name in enumerate(self.label_names):
                try:
                    report['ap/{:s}'.format(label_name)] = result['ap'][l]
                except IndexError:
                    report['ap/{:s}'.format(label_name)] = np.nan

        observation = dict()
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation


def git_hash():
    cmd = 'git log -1 --format="%h"'
    return subprocess.check_output(cmd, shell=True).strip().decode()


def git_info():
    cmd = 'git log -1 --format="%d"'
    output = subprocess.check_output(cmd, shell=True).strip()
    output = output.decode()
    branch = output.lstrip('(').rstrip(')').split(',')[-1].strip()

    cmd = 'git log -1 --format="%h - ({}) %B"'.format(branch)
    return subprocess.check_output(cmd, shell=True).strip().decode()


def get_hostname():
    cmd = 'hostname'
    return subprocess.check_output(cmd, shell=True).strip().decode()


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # training parameters
    parser.add_argument('--dataset', '-d',
                        choices=['voc', 'coco'],
                        default='voc', help='The dataset.')
    parser.add_argument('--model', '-m',
                        choices=['vgg16', 'resnet50', 'resnet101'],
                        default='resnet101', help='Base model of Mask R-CNN.')
    parser.add_argument(
        '--pretrained-model', '-pm',
        choices=['imagenet', 'voc12_train_rpn', 'voc12_train_faster_rcnn'],
        default='imagenet', help='Pretrained model.')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay.')
    parser.add_argument('--pooling-func', '-pf',
                        choices=['pooling', 'align', 'resize'],
                        default='pooling', help='Pooling function.')
    args = parser.parse_args()

    comm = chainermn.create_communicator('hierarchical')
    device = comm.intra_rank

    args.git = git_info()
    args.git_hash = git_hash()
    args.hostname = get_hostname()
    now = datetime.datetime.now()
    args.timestamp = now.isoformat()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S'))

    # 0.00125 * 2 * 8 = 0.02  in original
    args.batch_size = 1
    args.n_node = comm.inter_size
    args.n_gpu = comm.intra_size * args.n_node
    args.lr = 0.00125 * args.batch_size * args.n_gpu

    args.max_epoch = 19  # (160e3 * 16) / len(coco_trainval)
    # lr / 10 at 120k iteration with
    # 160k iteration * 16 batchsize in original
    args.step_size = (120e3 / 160e3) * args.max_epoch

    if comm.mpi_comm.rank == 0:
        os.makedirs(args.out)
        with open(osp.join(args.out, 'params.yaml'), 'w') as f:
            yaml.safe_dump(args.__dict__, f, default_flow_style=False)
        print('# ' + '-' * 77)
        yaml.safe_dump(args.__dict__, sys.stdout, default_flow_style=False)
        print('# ' + '-' * 77)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'voc':
        train_data = mrcnn.datasets.SBDInstanceSeg('train')
        test_data = mrcnn.datasets.VOC2012InstanceSeg('val')
    elif args.dataset == 'coco':
        train_data = mrcnn.datasets.CocoInstanceSeg('train')
        test_data = mrcnn.datasets.CocoInstanceSeg('minival')
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
            pretrained_model=args.pretrained_model,
            pooling_func=pooling_func)
    elif args.model in ['resnet50', 'resnet101']:
        n_layers = int(args.model.lstrip('resnet'))
        mask_rcnn = mrcnn.models.MaskRCNNResNet(
            n_layers=n_layers,
            n_fg_class=len(instance_class_names),
            pretrained_model=args.pretrained_model,
            pooling_func=pooling_func)
    else:
        raise ValueError
    mask_rcnn.use_preset('evaluate')
    model = mrcnn.models.MaskRCNNTrainChain(mask_rcnn)

    chainer.cuda.get_device(device).use()
    model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))

    model.mask_rcnn.extractor.mode = 'res3+'
    mask_rcnn.extractor.conv1.disable_update()
    mask_rcnn.extractor.bn1.disable_update()
    mask_rcnn.extractor.res2.disable_update()

    if comm.rank == 0:
        train_data = TransformDataset(
            train_data, Transform(mask_rcnn))
        test_data = TransformDataset(
            test_data, Transform(mask_rcnn, train=False))
    else:
        train_data = None
        test_data = None
    train_data = chainermn.scatter_dataset(train_data, comm, shuffle=True)
    test_data = chainermn.scatter_dataset(test_data, comm)

    # multiprocessing.set_start_method('forkserver')
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size=1, n_processes=4, shared_mem=100000000)
    test_iter = chainer.iterators.MultiprocessIterator(
        test_data, batch_size=1, n_processes=4, shared_mem=100000000,
        repeat=False, shuffle=False)

    def concat_examples(batch, device=None, padding=None):
        from chainer.dataset.convert import _concat_arrays
        from chainer.dataset.convert import to_device
        if len(batch) == 0:
            raise ValueError('batch is empty')

        first_elem = batch[0]

        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            res = _concat_arrays([example[i] for example in batch], padding[i])
            if i in [0, 1]:  # img, bbox
                res = to_device(device, res)
            result.append(res)

        return tuple(result)

    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=device, converter=concat_examples)

    trainer = training.Trainer(
        updater, (args.max_epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.step_size, 'epoch'))

    eval_interval = 1, 'epoch'
    log_interval = 20, 'iteration'
    plot_interval = 0.1, 'epoch'
    print_interval = 20, 'iteration'

    checkpointer = chainermn.create_multi_node_checkpointer(
        name='mask-rcnn', comm=comm)
    checkpointer.maybe_load(trainer, optimizer)
    trainer.extend(checkpointer, trigger=eval_interval)

    evaluator = InstanceSegmentationVOCEvaluator(
        test_iter, model.mask_rcnn, device=device,
        use_07_metric=True, label_names=instance_class_names)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=eval_interval)

    if comm.rank == 0:
        trainer.extend(
            InstanceSegmentationVisReport(
                test_iter, model.mask_rcnn,
                label_names=instance_class_names),
            trigger=eval_interval)
        trainer.extend(
            extensions.snapshot_object(model.mask_rcnn, 'snapshot_model.npz'),
            trigger=training.triggers.MaxValueTrigger(
                'validation/main/map', eval_interval))
        trainer.extend(chainer.training.extensions.observe_lr(),
                       trigger=log_interval)
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.PrintReport([
            'iteration', 'epoch', 'elapsed_time', 'lr',
            'main/loss',
            'main/roi_loc_loss',
            'main/roi_cls_loss',
            'main/roi_mask_loss',
            'main/rpn_loc_loss',
            'main/rpn_cls_loss',
            'validation/main/map',
        ]), trigger=print_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

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
            trigger=plot_interval,
        )

        trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()

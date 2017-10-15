#!/usr/bin/env python

from __future__ import division

import argparse
import copy
import datetime
import numpy as np
import os
import os.path as osp
import pprint

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
from chainer.datasets import TransformDataset
from chainer import reporter
from chainer import training
from chainer.training import extensions

from chainercv.datasets import voc_bbox_label_names
from chainercv import transforms
from chainercv.utils import apply_prediction_to_iterator

import mask_rcnn as mrcnn


class Transform(object):

    def __init__(self, mask_rcnn):
        self.mask_rcnn = mask_rcnn

    def __call__(self, in_data):
        img, bbox, label, mask = in_data
        img = img.transpose(2, 0, 1)  # H, W, C -> C, H, W
        _, H, W = img.shape
        img = self.mask_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
        mask = transforms.resize(mask, size=(o_H, o_W), interpolation=0)

        # # horizontally flip
        # img, params = transforms.random_flip(
        #     img, x_random=True, return_param=True)
        # bbox = transforms.flip_bbox(
        #     bbox, (o_H, o_W), x_flip=params['x_flip'])
        # mask = transforms.flip(mask, (o_H, o_W), x_flip=params['x_flip'])

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


class InstanceSegmentationVOCEvaluator(chainer.training.extensions.Evaluator):

    def __init__(
            self, iterator, target, use_07_metric=False, label_names=None):
        super(InstanceSegmentationVOCEvaluator, self).__init__(
            iterator, target)
        self.use_07_metric = use_07_metric
        self.label_names = label_names

        self._trainer = None

    def __call__(self, trainer=None):
        self._trainer = trainer
        return super(InstanceSegmentationVOCEvaluator, self).__call__(trainer)

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        # TODO(wkentaro): visualize prediction

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            target.predict_masks, it)
        # delete unused iterator explicitly
        del imgs

        pred_bboxes, pred_masks, pred_labels, pred_scores = pred_values
        del pred_bboxes

        if len(gt_values) == 4:
            gt_bboxes, gt_labels, gt_masks, gt_difficults = gt_values
        elif len(gt_values) == 3:
            gt_bboxes, gt_labels, gt_masks = gt_values
            gt_difficults = None

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


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU id.')
    parser.add_argument('--lr', '-l', type=float, default=1e-3,
                        help='Learning rate.')
    # parser.add_argument('--out', '-o', default='logs', help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--step_size', '-ss', type=int, default=50000,
                        help='Step size of iterations.')
    parser.add_argument('--iteration', '-i', type=int, default=70000,
                        help='Iteration size.')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay.')
    parser.add_argument('--overfit', action='store_true',
                        help='Do overfit training (single image).')
    args = parser.parse_args()

    args.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.out = osp.join(
        here, 'logs',
        '.'.join([
            'lr={lr}',
            'seed={seed}',
            'step_size={step_size}',
            'iteration={iteration}',
            'weight_decay={weight_decay}',
            'overfit={overfit}',
            'timestamp={timestamp}',
        ]).format(**args.__dict__)
    )

    pprint.pprint(args.__dict__)

    np.random.seed(args.seed)

    train_data = mrcnn.datasets.MaskRcnnDataset(
        mrcnn.datasets.VOC2012InstanceSeg('train'))
    test_data = mrcnn.datasets.MaskRcnnDataset(
        mrcnn.datasets.VOC2012InstanceSeg('val'))
    if args.overfit:
        train_data = OverfitDataset(train_data, indices=0)
        test_data = OverfitDataset(train_data, indices=0)

    # mask_rcnn = mrcnn.models.MaskRCNNVGG16(
    #     n_fg_class=len(voc_bbox_label_names),
    #     pretrained_model='imagenet')
    mask_rcnn = mrcnn.models.MaskRCNNVGG16(
        n_fg_class=len(voc_bbox_label_names),
        pretrained_model='voc0712_faster_rcnn')
    model = mrcnn.models.MaskRCNNTrainChain(mask_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))

    train_data = TransformDataset(train_data, Transform(mask_rcnn))

    def transform_test_data(in_data):
        img = in_data[0]
        img = img.transpose(2, 0, 1)
        out_data = list(in_data)
        out_data[0] = img
        return tuple(out_data)

    test_data = TransformDataset(test_data, transform_test_data)

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

    if args.overfit:
        eval_interval = 100, 'iteration'
        log_interval = 1, 'iteration'
        plot_interval = 100, 'iteration'
        print_interval = 1, 'iteration'
    else:
        eval_interval = 3000, 'iteration'
        log_interval = 20, 'iteration'
        plot_interval = 3000, 'iteration'
        print_interval = 20, 'iteration'

    trainer.extend(
        InstanceSegmentationVOCEvaluator(
            test_iter, model.mask_rcnn, use_07_metric=True,
            label_names=voc_bbox_label_names),
        trigger=eval_interval)
    trainer.extend(
        extensions.snapshot_object(model.mask_rcnn, 'snapshot_model.npz'),
        trigger=training.triggers.MaxValueTrigger(
            'validation/main/map', eval_interval))
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

    if extensions.PlotReport.available():
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

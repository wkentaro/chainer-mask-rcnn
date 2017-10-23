#!/usr/bin/env python

from __future__ import division

import argparse
import copy
import datetime
import itertools
import os
import os.path as osp
import pprint

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
from chainer.datasets import TransformDataset
from chainer import reporter
from chainer import training
from chainer.training import extensions
# from chainer.training import triggers
from chainercv.datasets import voc_bbox_label_names
from chainercv import transforms
from chainercv.utils import apply_prediction_to_iterator
import cv2
import numpy as np
import six

import mask_rcnn as mrcnn
import mvtk


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
            self, iterator, target, use_07_metric=False, label_names=None,
            file_name='visualizations/iteration=%08d.jpg'):
        super(InstanceSegmentationVOCEvaluator, self).__init__(
            iterator, target)
        self.use_07_metric = use_07_metric
        self.label_names = np.asarray(label_names)
        self.file_name = file_name

        self._trainer = None

    def __call__(self, trainer=None):
        self._trainer = trainer
        return super(InstanceSegmentationVOCEvaluator, self).__call__(trainer)

    def evaluate(self):
        trainer = self._trainer
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

        if trainer:
            gt_bboxes, gt_bboxes2 = itertools.tee(gt_bboxes)
            gt_labels, gt_labels2 = itertools.tee(gt_labels)
            gt_masks, gt_masks2 = itertools.tee(gt_masks)
            pred_bboxes, pred_bboxes2 = itertools.tee(pred_bboxes)
            pred_labels, pred_labels2 = itertools.tee(pred_labels)
            pred_masks, pred_masks2 = itertools.tee(pred_masks)
            pred_scores, pred_scores2 = itertools.tee(pred_scores)

            target.use_preset('visualize')

            # visualize
            n_viz = 9
            vizs = []
            for img, gt_bbox, gt_label, gt_mask, \
                pred_bbox, pred_label, pred_mask, pred_score \
                    in six.moves.zip(imgs, gt_bboxes2, gt_labels2, gt_masks2,
                                     pred_bboxes2, pred_labels2, pred_masks2,
                                     pred_scores2):
                # organize input
                img = img.transpose(1, 2, 0)  # CHW -> HWC
                gt_mask = gt_mask.astype(bool)

                n_fg_class = len(self.label_names)

                gt_viz = mrcnn.utils.draw_instance_boxes(
                    img, gt_bbox, gt_label, n_class=n_fg_class,
                    masks=gt_mask, captions=self.label_names[gt_label],
                    bg_class=-1)

                pred_viz = mrcnn.utils.draw_instance_boxes(
                    img, pred_bbox, pred_label, n_class=n_fg_class,
                    masks=pred_mask, captions=self.label_names[pred_label],
                    bg_class=-1)

                viz = np.vstack([gt_viz, pred_viz])
                vizs.append(viz)
                if len(vizs) >= n_viz:
                    break
            viz = mvtk.image.tile(vizs)
            file_name = osp.join(
                trainer.out, self.file_name % trainer.updater.iteration)
            try:
                os.makedirs(osp.dirname(file_name))
            except OSError:
                if not osp.isdir(osp.dirname(file_name)):
                    raise
            cv2.imwrite(file_name, viz[:, :, ::-1])

        target.use_preset('evaluate')

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


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # training parameters
    parser.add_argument('--model', choices=['vgg16', 'resnet50', 'resnet101'],
                        default='resnet50', help='Base model of Mask R-CNN.')
    parser.add_argument(
        '--pretrained-model',
        choices=['imagenet', 'voc12_train_rpn', 'voc12_train_faster_rcnn'],
        default='voc12_train_rpn', help='Pretrained model.')
    parser.add_argument('--lr', '-l', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--step_size', '-ss', type=int, default=50000,
                        help='Step size of iterations.')
    parser.add_argument('--iteration', '-i', type=int, default=70000,
                        help='Iteration size.')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay.')
    parser.add_argument('--update-policy',
                        choices=['almost_all', 'head_only', 'mask_only'],
                        default='almost_all')
    parser.add_argument('--pooling-func', choices=['pooling', 'align'],
                        default='align', help='Pooling function.')
    parser.add_argument('--overfit', action='store_true',
                        help='Do overfit training (single image).')
    # other parameters
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU id.')
    args = parser.parse_args()

    args.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.out = osp.join(
        here, 'logs',
        '.'.join([
            'model={model}',
            'pretrained_model={pretrained_model}',
            'lr={lr}',
            'seed={seed}',
            'step_size={step_size}',
            'iteration={iteration}',
            'weight_decay={weight_decay}',
            'update_policy={update_policy}',
            'pooling_func={pooling_func}',
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
        train_data = OverfitDataset(train_data, indices=range(0, 9))
        test_data = OverfitDataset(train_data, indices=range(0, 9))

    if args.pooling_func == 'align':
        pooling_func = mrcnn.functions.roi_align_2d
    elif args.pooling_func == 'pooling':
        pooling_func = chainer.functions.roi_pooling_2d
    else:
        raise ValueError

    if args.model == 'vgg16':
        mask_rcnn = mrcnn.models.MaskRCNNVGG16(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model,
            pooling_func=pooling_func)
    elif args.model in ['resnet50', 'resnet101']:
        n_layers = int(args.model.lstrip('resnet'))
        mask_rcnn = mrcnn.models.MaskRCNNResNet(
            n_layers=n_layers,
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model,
            pooling_func=pooling_func)
    else:
        raise ValueError
    model = mrcnn.models.MaskRCNNTrainChain(mask_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))
    if args.update_policy == 'almost_all':
        # This is only relevant to ResNet training.
        if args.model in ['resnet50', 'resnet101']:
            for p in model.params():
                # Do not update batch normalization layers.
                if p.name == 'gamma':
                    p.update_rule.enabled = False
                elif p.name == 'beta':
                    p.update_rule.enabled = False
        # Do not update for the first two blocks.
        mask_rcnn.extractor.conv1.disable_update()
        mask_rcnn.extractor.bn1.disable_update()
        mask_rcnn.extractor.res2.disable_update()
    elif args.update_policy == 'head_only':
        mask_rcnn.extractor.disable_update()
        mask_rcnn.rpn.disable_update()
    elif args.update_policy == 'mask_only':
        mask_rcnn.extractor.disable_update()
        mask_rcnn.rpn.disable_update()
        if args.model == 'vgg16':
            mask_rcnn.head.fc6.disable_update()
            mask_rcnn.head.fc7.disable_update()
            mask_rcnn.head.cls_loc.disable_update()
            mask_rcnn.head.score.disable_update()
        elif args.model in ['resnet50', 'resnet101']:
            mask_rcnn.head.res5.disable_update()
            mask_rcnn.head.cls_loc.disable_update()
            mask_rcnn.head.score.disable_update()
    else:
        raise ValueError

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

    # # 0-4000: lr = 0.002      # warmup lr
    # # 4000 - step_size: 0.02  # base lr
    # # step_size - : 0.002     # stepping lr
    # trainer.extend(extensions.ExponentialShift('lr', 10),
    #                trigger=triggers.ManualScheduleTrigger(
    #                     points=[4000], unit='iteration'))  # only once
    # trainer.extend(extensions.ExponentialShift('lr', 0.1),
    #                trigger=(args.step_size, 'iteration'))

    if args.overfit:
        eval_interval = 100, 'iteration'
        log_interval = 1, 'iteration'
        plot_interval = 100, 'iteration'
        print_interval = 1, 'iteration'
    else:
        eval_interval = 3000, 'iteration'
        log_interval = 20, 'iteration'
        plot_interval = 1000, 'iteration'
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

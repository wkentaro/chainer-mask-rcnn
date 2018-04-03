import copy
import os
import os.path as osp
import shutil

import chainer
from chainercv.utils import apply_prediction_to_iterator
import cv2
import fcn
import numpy as np
import six

import chainer_mask_rcnn as mrcnn


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
            target.predict, it)

        pred_bboxes, pred_masks, pred_labels, pred_scores = pred_values

        gt_bboxes, gt_labels, gt_masks = gt_values[:3]

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

        viz = fcn.utils.get_tile_image(vizs, tile_shape=self._shape)
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

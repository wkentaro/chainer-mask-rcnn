import copy

import chainer
from chainer import reporter
from chainercv.utils import apply_prediction_to_iterator
import numpy as np
import tqdm

import chainer_mask_rcnn as mrcnn


class InstanceSegmentationVOCEvaluator(chainer.training.extensions.Evaluator):

    name = 'validation'

    def __init__(self, iterator, target, device=None,
                 use_07_metric=False, label_names=None, show_progress=False):
        super(InstanceSegmentationVOCEvaluator, self).__init__(
            iterator=iterator, target=target, device=device)
        self.use_07_metric = use_07_metric
        self.label_names = label_names
        self._show_progress = show_progress

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        if self._show_progress:
            it = tqdm.tqdm(it, total=len(it.dataset))

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            target.predict, it)

        pred_bboxes, pred_masks, pred_labels, pred_scores = pred_values

        if len(gt_values) == 4:
            gt_bboxes, gt_labels, gt_masks, gt_difficults = gt_values
        elif len(gt_values) == 3:
            gt_bboxes, gt_labels, gt_masks = gt_values
            gt_difficults = None

        # evaluate
        result = mrcnn.utils.eval_instseg_voc(
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

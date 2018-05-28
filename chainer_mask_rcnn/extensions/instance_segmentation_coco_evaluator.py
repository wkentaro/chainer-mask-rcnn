import copy

import chainer
from chainer import reporter
from chainercv.utils import apply_to_iterator
import numpy as np
import tqdm

from .. import utils


class InstanceSegmentationCOCOEvaluator(chainer.training.extensions.Evaluator):

    name = 'validation'

    def __init__(self, iterator, target, device=None, label_names=None,
                 show_progress=False):
        super(InstanceSegmentationCOCOEvaluator, self).__init__(
            iterator=iterator, target=target, device=device)
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

        in_values, out_values, rest_values = apply_to_iterator(
            target.predict, it)
        del in_values

        pred_bboxes, pred_masks, pred_labels, pred_scores = out_values

        if len(rest_values) == 5:
            gt_bboxes, gt_labels, gt_masks, gt_crowdeds, gt_areas = rest_values
        elif len(rest_values) == 3:
            gt_bboxes, gt_labels, gt_masks = rest_values
            gt_crowdeds = None
            gt_areas = None

        # evaluate
        result = utils.eval_instseg_coco(
            pred_masks, pred_labels, pred_scores,
            gt_masks, gt_labels, gt_crowdeds, gt_areas)

        report = {
            'map': result['map/iou=0.50:0.95/area=all/maxDets=100'],
            'map@0.5': result['map/iou=0.50/area=all/maxDets=100'],
            'map@0.75': result['map/iou=0.75/area=all/maxDets=100'],
        }

        if self.label_names is not None:
            for l, label_name in enumerate(self.label_names):
                try:
                    report['ap/{:s}'.format(label_name)] = \
                        result['ap/iou=0.50:0.95/area=all/maxDets=100'][l]
                except IndexError:
                    report['ap/{:s}'.format(label_name)] = np.nan

        observation = dict()
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation

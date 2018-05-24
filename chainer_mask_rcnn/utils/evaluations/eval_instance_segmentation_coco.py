# Modified work:
# Copyright (c) 2018 Kentaro Wada

# Original work:
# https://github.com/knorth55/chainer-fcis/blob/master/fcis/evaluations/eval_instance_segmentation_coco.py  # NOQA

import contextlib
import itertools
import numpy as np
import os
import sys

import six

import pycocotools.coco
import pycocotools.cocoeval
import pycocotools.mask


def eval_instseg_coco(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels, gt_crowdeds=None, gt_areas=None):
    """Evaluate instance segmentation based on evaluation code of MS COCO.

    Args:
        sizes (iterable of tuple of ints): [(H_1, W_1), ..., (H_N, W_N)]
        pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to :obj:`y_min, x_min, y_max, x_max`
            of a bounding box.
        pred_masks (iterable of list of numpy.ndarray)
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
            bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_masks (iterable of list of numpy.ndarray)
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_crowdeds (iterable of numpy.ndarray): An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the "crowded" label is assigned to the
            corresponding bounding boxes.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not crowded.
        gt_area (iterable of numpy.ndarray): An iterable of float
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This contains the area of the instance mask of an object
            for each bounding box. By default, this is :obj:`None`.
            In that case, this function uses the area of the
            bounding box (i.e. width multiplied by height).

    """
    gt_coco = pycocotools.coco.COCO()
    pred_coco = pycocotools.coco.COCO()

    if gt_crowdeds is None:
        gt_crowdeds = itertools.repeat(None)
    if gt_areas is None:
        gt_areas = itertools.repeat(None)

    images = list()
    pred_anns = list()
    gt_anns = list()
    unique_labels = dict()
    for i, pred_mask, pred_label, pred_score, \
            gt_mask, gt_label, gt_crowded, gt_area in \
            six.moves.zip(
                itertools.count(),
                pred_masks, pred_labels, pred_scores,
                gt_masks, gt_labels, gt_crowdeds, gt_areas):

        if gt_area is None:
            gt_area = itertools.repeat(None)
        if gt_crowded is None:
            gt_crowded = itertools.repeat(None)
        # Starting ids from 1 is important when using COCO.
        img_id = i + 1

        for pred_m, pred_lbl, pred_sc in zip(
                pred_mask, pred_label, pred_score):
            pred_anns.append(
                _create_ann(pred_m, pred_lbl, pred_sc,
                            img_id=img_id, ann_id=len(pred_anns) + 1,
                            crw=0))
            unique_labels[pred_lbl] = True

        for gt_m, gt_lbl, gt_crw, gt_ar in zip(
                gt_mask, gt_label, gt_crowded, gt_area):
            gt_anns.append(
                _create_ann(gt_m, gt_lbl, None,
                            img_id=img_id, ann_id=len(gt_anns) + 1,
                            crw=gt_crw, ar=gt_ar))
            unique_labels[gt_lbl] = True
        size = gt_mask.shape[:2]
        images.append({'id': img_id, 'height': size[0], 'width': size[1]})

    pred_coco.dataset['categories'] = [{'id': i} for i in unique_labels.keys()]
    gt_coco.dataset['categories'] = [{'id': i} for i in unique_labels.keys()]
    pred_coco.dataset['annotations'] = pred_anns
    gt_coco.dataset['annotations'] = gt_anns
    pred_coco.dataset['images'] = images
    gt_coco.dataset['images'] = images

    with _redirect_stdout(open(os.devnull, 'w')):
        pred_coco.createIndex()
        gt_coco.createIndex()
        ev = pycocotools.cocoeval.COCOeval(gt_coco, pred_coco, 'segm')
        ev.evaluate()
        ev.accumulate()

    results = {'coco_eval': ev}
    p = ev.params
    common_kwargs = {
        'prec': ev.eval['precision'],
        'rec': ev.eval['recall'],
        'iou_threshs': p.iouThrs,
        'area_ranges': p.areaRngLbl,
        'max_detection_list': p.maxDets}
    all_kwargs = {
        'ap/iou=0.50:0.95/area=all/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 100},
        'ap/iou=0.50/area=all/maxDets=100': {
            'ap': True, 'iou_thresh': 0.5, 'area_range': 'all',
            'max_detection': 100},
        'ap/iou=0.75/area=all/maxDets=100': {
            'ap': True, 'iou_thresh': 0.75, 'area_range': 'all',
            'max_detection': 100},
        'ap/iou=0.50:0.95/area=small/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'small',
            'max_detection': 100},
        'ap/iou=0.50:0.95/area=medium/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'medium',
            'max_detection': 100},
        'ap/iou=0.50:0.95/area=large/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'large',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=all/maxDets=1': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 1},
        'ar/iou=0.50:0.95/area=all/maxDets=10': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 10},
        'ar/iou=0.50:0.95/area=all/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=small/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'small',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=medium/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'medium',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=large/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'large',
            'max_detection': 100},
    }

    for key, kwargs in all_kwargs.items():
        kwargs.update(common_kwargs)
        metrics, mean_metric = _summarize(**kwargs)
        results[key] = metrics
        results['m' + key] = mean_metric
    return results


def _create_ann(whole_m, lbl, sc, img_id, ann_id, crw=None, ar=None):
    H, W = whole_m.shape
    if crw is None:
        crw = False
    whole_m = np.asfortranarray(whole_m.astype(np.uint8))
    rle = pycocotools.mask.encode(whole_m)
    # Surprisingly, ground truth ar can be different from area(rle)
    if ar is None:
        ar = pycocotools.mask.area(rle)
    ann = {
        'image_id': img_id, 'category_id': lbl,
        'segmentation': rle,
        'area': ar,
        'id': ann_id,
        'iscrowd': crw}
    if sc is not None:
        ann.update({'score': sc})
    return ann


def _summarize(
        prec, rec, iou_threshs, area_ranges,
        max_detection_list,
        ap=True, iou_thresh=None, area_range='all',
        max_detection=100):
    a_idx = area_ranges.index(area_range)
    m_idx = max_detection_list.index(max_detection)
    if ap:
        s = prec.copy()  # (T, R, K, A, M)
        if iou_thresh is not None:
            s = s[iou_thresh == iou_threshs]
        s = s[:, :, :, a_idx, m_idx]
    else:
        s = rec.copy()  # (T, K, A, M)
        if iou_thresh is not None:
            s = s[iou_thresh == iou_threshs]
        s = s[:, :, a_idx, m_idx]

    s[s == -1] = np.nan
    s = s.reshape((-1, s.shape[-1]))
    valid_classes = np.any(np.logical_not(np.isnan(s)), axis=0)
    class_s = np.nan * np.ones(len(valid_classes), dtype=np.float32)
    class_s[valid_classes] = np.nanmean(s[:, valid_classes], axis=0)

    if not np.any(valid_classes):
        mean_s = np.nan
    else:
        mean_s = np.nanmean(class_s)
    return class_s, mean_s


@contextlib.contextmanager
def _redirect_stdout(target):
    original = sys.stdout
    sys.stdout = target
    yield
    sys.stdout = original

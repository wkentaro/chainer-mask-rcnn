# Modified works:
# --------------------------------------------------------
# Copyright (c) 2017 - 2018 Kentaro Wada.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Original works:
# --------------------------------------------------------
# expand_boxes, segm_results
# Copyright (c) 2017-present, Facebook, Inc.
# Licensed under The Apache License [see LICENSE for details]
# https://github.com/facebookresearch/Detectron
# --------------------------------------------------------
# Copyright (c) 2017 Preferred Networks, Inc.
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/chainer/chainercv
# --------------------------------------------------------
# Faster R-CNN implementation by Chainer
# Copyright (c) 2016 Shunta Saito
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/mitmul/chainer-faster-rcnn
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

from __future__ import division

import cv2
import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.utils import non_maximum_suppression

from ..datasets import concat_examples


def expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def segm_results(bbox, label, roi_mask, im_h, im_w):
    if len(bbox) == 0:
        return np.zeros((0, im_h, im_w), dtype=bool)

    mask_size = roi_mask.shape[2]
    assert roi_mask.shape[3] == mask_size
    ref_boxes = bbox[:, [1, 0, 3, 2]]
    masks = roi_mask

    all_masks = []
    mask_ind = 0
    M = mask_size
    scale = (M + 2.0) / M
    ref_boxes = expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    for mask_ind in range(len(ref_boxes)):
        label_i = label[mask_ind]
        padded_mask[1:-1, 1:-1] = masks[mask_ind, label_i, :, :]

        ref_box = ref_boxes[mask_ind, :]
        w = (ref_box[2] - ref_box[0] + 1)
        h = (ref_box[3] - ref_box[1] + 1)
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)

        mask = cv2.resize(padded_mask, (w, h))
        mask = np.array(mask > 0.5, dtype=np.uint8)
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

        x_0 = max(ref_box[0], 0)
        x_1 = min(ref_box[2] + 1, im_w)
        y_0 = max(ref_box[1], 0)
        y_1 = min(ref_box[3] + 1, im_h)

        im_mask[y_0:y_1, x_0:x_1] = mask[
            (y_0 - ref_box[1]):(y_1 - ref_box[1]),
            (x_0 - ref_box[0]):(x_1 - ref_box[0])]
        im_mask = im_mask.astype(bool)

        all_masks.append(im_mask)
    all_masks = np.asarray(all_masks)

    return all_masks


class MaskRCNN(chainer.Chain):

    def __init__(
            self, extractor, rpn, head,
            mean,
            min_size=600,
            max_size=1000,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
            detections_per_im=100):
        super(MaskRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

        self.mean = mean
        self.min_size = min_size
        self.max_size = max_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        self.nms_thresh = 0.5
        self.score_thresh = 0.05

        self._detections_per_im = detections_per_im

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def __call__(self, x, scales):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor =\
            self.rpn(h, img_size, scales)
        roi_cls_locs, roi_scores, roi_masks = self.head(
            h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices, roi_masks

    def prepare(self, imgs):
        prepared_imgs = []
        sizes = []
        scales = []
        for img in imgs:
            _, H, W = img.shape

            scale = 1.

            if self.min_size:
                scale = self.min_size / min(H, W)

            if self.max_size and scale * max(H, W) > self.max_size:
                scale = self.max_size / max(H, W)

            img = img.transpose(1, 2, 0)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            img = img.transpose(2, 0, 1)

            img = (img - self.mean).astype(np.float32, copy=False)

            prepared_imgs.append(img)
            sizes.append((H, W))
            scales.append(scale)
        return prepared_imgs, sizes, scales

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]

            # thresholding by score
            keep = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[keep]
            prob_l = prob_l[keep]

            # thresholding by nms
            keep = non_maximum_suppression(
                cls_bbox_l, self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    def _to_bboxes(self, roi_cls_locs, roi_scores, rois, roi_indices, sizes,
                   scales):
        if isinstance(roi_cls_locs, chainer.Variable):
            roi_cls_locs = roi_cls_locs.array
        probs = F.softmax(roi_scores).array
        del roi_scores

        bboxes = []
        labels = []
        scores = []
        for index in range(len(sizes)):
            scale = scales[index]
            size = sizes[index]

            keep = roi_indices == index
            roi_cls_loc = roi_cls_locs[keep]
            prob = probs[keep]
            roi = rois[keep] / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = self.xp.tile(self.xp.asarray(self.loc_normalize_mean),
                                self.n_class)
            std = self.xp.tile(self.xp.asarray(self.loc_normalize_std),
                               self.n_class)
            roi_cls_loc = (roi_cls_loc * std + mean).astype(np.float32)
            roi_cls_loc = roi_cls_loc.reshape((-1, self.n_class, 4))
            roi_cls = self.xp.broadcast_to(roi[:, None], roi_cls_loc.shape)
            cls_bbox = loc2bbox(roi_cls.reshape((-1, 4)),
                                roi_cls_loc.reshape((-1, 4)))
            cls_bbox = cls_bbox.reshape((-1, self.n_class * 4))
            # clip bounding box
            cls_bbox[:, 0::2] = self.xp.clip(cls_bbox[:, 0::2], 0, size[0])
            cls_bbox[:, 1::2] = self.xp.clip(cls_bbox[:, 1::2], 0, size[1])
            # clip roi
            roi[:, 0::2] = self.xp.clip(roi[:, 0::2], 0, size[0])
            roi[:, 1::2] = self.xp.clip(roi[:, 1::2], 0, size[1])

            raw_cls_bbox = cuda.to_cpu(cls_bbox)
            raw_prob = cuda.to_cpu(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)

            bbox_int = np.round(bbox).astype(np.int32)
            bbox_sizes = ((bbox_int[:, 2] - bbox_int[:, 0]) *
                          (bbox_int[:, 3] - bbox_int[:, 1]))
            keep = bbox_sizes > 0
            bbox = bbox[keep]
            label = label[keep]
            score = score[keep]

            if self._detections_per_im > 0:
                indices = np.argsort(score)
                keep = indices >= (len(indices) - self._detections_per_im)
                bbox = bbox[keep]
                label = label[keep]
                score = score[keep]

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        return bboxes, labels, scores

    def _to_roi_masks(self, h, bboxes, roi_indices, scales):
        batch_size = h.shape[0]
        bboxes = np.concatenate(bboxes, axis=0)
        if bboxes.size == 0:
            n_fg_class = self.n_class - 1
            mask_size = self.head.mask_size
            return [
                np.zeros((0, n_fg_class, mask_size, mask_size),
                         dtype=np.float32)
                for _ in range(batch_size)
            ]

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            # use predicted bbox as rois
            rois = bboxes * scales[roi_indices][:, None]
            rois = self.xp.asarray(rois, dtype=np.float32)
            _, _, roi_masks = self.head(
                x=h,
                rois=rois,
                roi_indices=self.xp.asarray(roi_indices),
                pred_bbox=False,
            )
        roi_masks = cuda.to_cpu(roi_masks.array)
        return [roi_masks[roi_indices == i] for i in range(batch_size)]

    def _to_masks(self, bboxes, labels, scores, roi_masks, sizes):
        masks = []
        for bbox, label, score, roi_mask, size in \
                zip(bboxes, labels, scores, roi_masks, sizes):
            roi_mask = F.sigmoid(roi_mask).array
            mask = segm_results(
                bbox=bbox,
                label=label,
                roi_mask=roi_mask,
                im_h=size[0],
                im_w=size[1],
            )
            masks.append(mask)
        return masks

    def predict(self, imgs):
        imgs, sizes, scales = self.prepare(imgs)

        batch = list(zip(imgs, scales))
        x, scales = concat_examples(batch, padding=0)
        x = self.xp.asarray(x)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            h = self.extractor(x)
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(
                h, x.shape[2:], scales,
            )
            roi_cls_locs, roi_scores, _ = self.head(
                h, rois, roi_indices, pred_mask=False,
            )

        bboxes, labels, scores = self._to_bboxes(
            roi_cls_locs, roi_scores, rois, roi_indices, sizes, scales,
        )

        roi_indices = []
        for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
            assert len(bbox) == len(label) == len(score)
            roi_index = np.full((len(bbox),), i, dtype=np.int32)
            roi_indices.append(roi_index)
        roi_indices = np.concatenate(roi_indices, axis=0)

        roi_masks = self._to_roi_masks(h, bboxes, roi_indices, scales)
        masks = self._to_masks(bboxes, labels, scores, roi_masks, sizes)

        return bboxes, masks, labels, scores

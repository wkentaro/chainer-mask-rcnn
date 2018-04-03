# Mofidied work:
# --------------------------------------------------------
# Copyright (c) 2017 Preferred Networks, Inc.
# --------------------------------------------------------
#
# Original works by:
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
import warnings

import cv2
import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.transforms.image.resize import resize
from chainercv.utils import non_maximum_suppression
import six


class MaskRCNN(chainer.Chain):

    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`chainer.Chain` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    There are two functions :meth:`predict` and :meth:`__call__` to conduct
    object detection.
    :meth:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster R-CNN is treated as a black box function, for instance.
    :meth:`__call__` is provided for a scnerario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :func:`FasterRCNN.predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (callable Chain): A callable that takes a BCHW image
            array and returns feature maps.
        rpn (callable Chain): A callable that has the same interface as
            :class:`chainercv.links.RegionProposalNetwork`. Please refer to
            the documentation found there.
        head (callable Chain): A callable that takes
            a BCHW array, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        mean (numpy.ndarray): A value to be subtracted from an image
            in :meth:`prepare`.
        min_size (int): A preprocessing paramter for :meth:`prepare`. Please
            refer to a docstring found for :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """

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

        self._detections_per_im = detections_per_im

        self.use_preset('visualize')

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def __call__(self, x, scale=1.):
        """Forward Faster R-CNN.

        Scaling paramter :obj:`scale` is used by RPN to determine the
        threshold to select small objects, which are going to be
        rejected irrespective of their confidence scores.

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (~chainer.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image
                during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. \
                Its shape is :math:`(R', (L + 1) \\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', L + 1)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor =\
            self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores, roi_masks = self.head(
            h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices, roi_masks

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.5
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.5
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def prepare(self, img):
        """Preprocess an image for feature extraction.

        The length of the shorter edge is scaled to :obj:`self.min_size`.
        After the scaling, if the length of the longer edge is longer than
        :obj:`self.max_size`, the image is scaled to fit the longer edge
        to :obj:`self.max_size`.

        After resizing the image, the image is subtracted by a mean image value
        :obj:`self.mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        _, H, W = img.shape

        scale = 1.

        if self.min_size:
            scale = self.min_size / min(H, W)

        if self.max_size and scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)

        img = resize(img, (int(H * scale), int(W * scale)))

        img = (img - self.mean).astype(np.float32, copy=False)
        return img

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

    def predict_masks(self, imgs):
        warnings.warn('predict_masks is deprecated, please use predict.')
        return self.predict(imgs)

    def predict(self, imgs):
        prepared_imgs = list()
        sizes = list()
        for img in imgs:
            size = img.shape[1:]
            img = self.prepare(img.astype(np.float32))
            prepared_imgs.append(img)
            sizes.append(size)

        bboxes = list()
        masks = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                img_var = chainer.Variable(self.xp.asarray(img[None]))
                scale = img_var.shape[3] / size[1]

                img_size = img_var.shape[2:]

                h = self.extractor(img_var)
                rpn_locs, rpn_scores, rois, roi_indices, anchor =\
                    self.rpn(h, img_size, scale)
                roi_cls_locs, roi_scores, _, = self.head(
                    h, rois, roi_indices, pred_mask=False)
            # We are assuming that batch size is 1.
            roi_cls_loc = roi_cls_locs.data
            roi_score = roi_scores.data
            roi = rois / scale
            roi_index = roi_indices

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

            prob = F.softmax(roi_score).data

            roi_index = self.xp.broadcast_to(
                roi_index[:, None], roi_cls_loc.shape[:2])
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

            if len(bbox) == 0:
                masks.append(np.zeros((0, size[0], size[1]), dtype=bool))
                continue

            # use predicted bbox as rois
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                rois = self.xp.asarray(bbox) * scale
                roi_indices = self.xp.zeros(
                    (len(bbox),), dtype=np.int32)
                _, _, roi_masks = self.head(
                    x=h, rois=rois, roi_indices=roi_indices,
                    pred_bbox=False, pred_mask=True)
            roi_mask = cuda.to_cpu(roi_masks.data)
            roi_mask = roi_mask[np.arange(len(label)), label]

            roi = np.round(bbox).astype(np.int32)
            n_roi = len(roi)
            mask = np.zeros((n_roi, size[0], size[1]), dtype=bool)
            for i in six.moves.range(n_roi):
                y1, x1, y2, x2 = roi[i]
                roi_H = y2 - y1
                roi_W = x2 - x1
                roi_mask_i = cv2.resize(roi_mask[i], (roi_W, roi_H))
                roi_mask_i = roi_mask_i > 0.5  # float -> bool
                mask[i, y1:y2, x1:x2][roi_mask_i] = True
            masks.append(mask)

        return bboxes, masks, labels, scores

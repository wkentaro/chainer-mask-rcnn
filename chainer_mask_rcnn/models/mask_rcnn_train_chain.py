# Modified works:
# --------------------------------------------------------
# Copyright (c) 2017 - 2018 Kentaro Wada.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# This is modified work of FasterRCNNTrainChain:
# --------------------------------------------------------
# Copyright (c) 2017 Preferred Networks, Inc.
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/chainer/chainercv
# --------------------------------------------------------

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F

from .utils import ProposalTargetCreator
from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import\
    AnchorTargetCreator


class MaskRCNNTrainChain(chainer.Chain):

    """Calculate losses for Faster R-CNN and report them.

    This is used to train Faster R-CNN in the joint training scheme
    [#FRCNN]_.

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.

    .. [#FRCNN] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        faster_rcnn (~chainercv.links.model.faster_rcnn.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
        rpn_sigma (float): Sigma parameter for the localization loss
            of Region Proposal Network (RPN). The default value is 3,
            which is the value used in [#FRCNN]_.
        roi_sigma (float): Sigma paramter for the localization loss of
            the head. The default value is 1, which is the value used
            in [#FRCNN]_.
        anchor_target_creator: An instantiation of
            :obj:`chainercv.links.model.faster_rcnn.AnchorTargetCreator`.
        proposal_target_creator_params: An instantiation of
            :obj:`chainercv.links.model.faster_rcnn.ProposalTargetCreator`.

    """

    def __init__(self, mask_rcnn, rpn_sigma=3., roi_sigma=1.,
                 anchor_target_creator=AnchorTargetCreator(),
                 proposal_target_creator=ProposalTargetCreator(),
                 ):
        super(MaskRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.mask_rcnn = mask_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma

        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = proposal_target_creator

        self.loc_normalize_mean = mask_rcnn.loc_normalize_mean
        self.loc_normalize_std = mask_rcnn.loc_normalize_std

    def __call__(self, imgs, bboxes, labels, masks, scales):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~chainer.Variable): A variable with a batch of images.
            bboxes (~chainer.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~chainer.Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float or ~chainer.Variable): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            chainer.Variable:
            Scalar loss variable.
            This is the sum of losses for Region Proposal Network and
            the head module.

        """
        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.data
        if isinstance(labels, chainer.Variable):
            labels = labels.data
        if isinstance(scales, chainer.Variable):
            scales = scales.data
        scales = cuda.to_cpu(scales)

        batch_size, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.mask_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.mask_rcnn.rpn(
            features, img_size, scales)

        batch_indices = range(batch_size)
        sample_rois = []
        sample_roi_indices = []
        gt_roi_locs = []
        gt_roi_labels = []
        gt_roi_masks = []
        for batch_index, bbox, label, mask in \
                zip(batch_indices, bboxes, labels, masks):
            roi = rois[roi_indices == batch_index]
            sample_roi, gt_roi_loc, gt_roi_label, gt_roi_mask = \
                self.proposal_target_creator(roi, bbox, label, mask)
            del roi
            sample_roi_index = self.xp.full(
                (len(sample_roi),), batch_index, dtype=np.int32)
            sample_rois.append(sample_roi)
            sample_roi_indices.append(sample_roi_index)
            del sample_roi, sample_roi_index
            gt_roi_locs.append(gt_roi_loc)
            gt_roi_labels.append(gt_roi_label)
            gt_roi_masks.append(gt_roi_mask)
            del gt_roi_loc, gt_roi_label, gt_roi_mask
        sample_rois = self.xp.concatenate(sample_rois, axis=0)
        sample_roi_indices = self.xp.concatenate(sample_roi_indices, axis=0)
        gt_roi_locs = self.xp.concatenate(gt_roi_locs, axis=0)
        gt_roi_labels = self.xp.concatenate(gt_roi_labels, axis=0)
        gt_roi_masks = self.xp.concatenate(gt_roi_masks, axis=0)

        roi_cls_locs, roi_scores, roi_masks = self.mask_rcnn.head(
            features, sample_rois, sample_roi_indices)

        # RPN losses
        gt_rpn_locs = []
        gt_rpn_labels = []
        for bbox, rpn_loc, rpn_score in zip(bboxes, rpn_locs, rpn_scores):
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                bbox, anchor, img_size)
            gt_rpn_locs.append(gt_rpn_loc)
            gt_rpn_labels.append(gt_rpn_label)
            del gt_rpn_loc, gt_rpn_label
        gt_rpn_locs = self.xp.concatenate(gt_rpn_locs, axis=0)
        gt_rpn_labels = self.xp.concatenate(gt_rpn_labels, axis=0)
        rpn_locs = F.concat(rpn_locs, axis=0)
        rpn_scores = F.concat(rpn_scores, axis=0)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_locs, gt_rpn_locs, gt_rpn_labels, self.rpn_sigma)
        rpn_cls_loss = F.sigmoid_cross_entropy(rpn_scores, gt_rpn_labels)

        # Losses for outputs of the head.
        n_sample = len(roi_cls_locs)
        roi_cls_locs = roi_cls_locs.reshape((n_sample, -1, 4))
        roi_locs = roi_cls_locs[self.xp.arange(n_sample), gt_roi_labels]
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_locs, gt_roi_locs, gt_roi_labels, self.roi_sigma)
        roi_cls_loss = F.softmax_cross_entropy(roi_scores, gt_roi_labels)

        # Losses for outputs of mask branch
        roi_mask_loss = F.sigmoid_cross_entropy(
            roi_masks[np.arange(n_sample), gt_roi_labels - 1, :, :],
            gt_roi_masks)

        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss + \
            roi_mask_loss
        chainer.reporter.report({'rpn_loc_loss': rpn_loc_loss,
                                 'rpn_cls_loss': rpn_cls_loss,
                                 'roi_loc_loss': roi_loc_loss,
                                 'roi_cls_loss': roi_cls_loss,
                                 'roi_mask_loss': roi_mask_loss,
                                 'loss': loss},
                                self)
        return loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = F.absolute(diff)
    flag = (abs_diff.data < (1. / sigma2)).astype(np.float32)

    y = (flag * (sigma2 / 2.) * F.square(diff) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))

    return F.sum(y)


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    xp = chainer.cuda.get_array_module(pred_loc)

    in_weight = xp.zeros_like(gt_loc)
    # Localization loss is calculated only for positive rois.
    in_weight[gt_label > 0] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= xp.sum(gt_label >= 0)
    return loc_loss

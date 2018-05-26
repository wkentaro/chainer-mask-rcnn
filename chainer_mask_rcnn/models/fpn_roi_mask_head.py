# https://raw.githubusercontent.com/katotetsuro/chainer-maskrcnn/master/chainer_maskrcnn/model/head/fpn_roi_mask_head.py

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

from .. import functions
from .resnet_roi_mask_head import _roi_pooling_2d_yx


class FPNRoIMaskHead(chainer.Chain):

    mask_size = 28  # size of ground truth mask

    def __init__(self,
                 n_class,
                 roi_size_bbox=7,
                 roi_size_mask=14,
                 loc_initialW=None,
                 score_initialW=None,
                 mask_initialW=None,
                 pooling_func=functions.roi_align_2d):
        # n_class includes the background
        super(FPNRoIMaskHead, self).__init__()
        with self.init_scope():
            # layers for box prediction path
            self.fc1 = L.Linear(7 * 7 * 256, 1024)
            self.fc2 = L.Linear(1024, 1024)
            self.cls_loc = L.Linear(1024, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(1024, n_class, initialW=score_initialW)

            # mask prediction path
            self.mask1 = L.Convolution2D(256, 256, ksize=3, pad=1)
            self.mask2 = L.Convolution2D(256, 256, ksize=3, pad=1)
            self.mask3 = L.Convolution2D(256, 256, ksize=3, pad=1)
            self.mask4 = L.Convolution2D(256, 256, ksize=3, pad=1)
            self.deconv1 = L.Deconvolution2D(
                in_channels=256,
                out_channels=256,
                ksize=2,
                stride=2,
                pad=0,
                initialW=mask_initialW)
            self.conv2 = L.Convolution2D(
                in_channels=256,
                out_channels=n_class - 1,
                ksize=1,
                stride=1,
                pad=0,
                initialW=mask_initialW)

        self.n_class = n_class
        self.roi_size_bbox = roi_size_bbox
        self.roi_size_mask = roi_size_mask
        self.pooling_func = pooling_func

    def __call__(self, x, rois, roi_indices, levels, spatial_scales,
                 pred_bbox=True, pred_mask=True):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)

        pool_box = []
        levels = chainer.cuda.to_cpu(levels).astype(np.int32)
        for l, i in zip(levels, indices_and_rois):
            pool_box.append(
                _roi_pooling_2d_yx(
                    x[l],
                    i[None],
                    self.roi_size_bbox,
                    self.roi_size_bbox,
                    spatial_scales[l],
                    self.pooling_func,
                ),
            )

        pool_box = F.concat(pool_box, axis=0)

        h = F.relu(self.fc1(pool_box))
        h = F.relu(self.fc2(h))

        roi_cls_locs = None
        roi_scores = None
        roi_masks = None

        if pred_bbox:
            roi_cls_locs = self.cls_loc(h)
            roi_scores = self.score(h)

        if pred_mask:
            pool_mask = []
            for l, i in zip(levels, indices_and_rois):
                pool_mask.append(
                    _roi_pooling_2d_yx(
                        x[l],
                        i[None],
                        self.roi_size_mask,
                        self.roi_size_mask,
                        spatial_scales[l],
                        self.pooling_func,
                    ),
                )
            pool_mask = F.concat(pool_mask, axis=0)
            h = F.relu(self.mask1(pool_mask))
            h = F.relu(self.mask2(h))
            h = F.relu(self.mask3(h))
            h = F.relu(self.mask4(h))
            h = F.relu(self.deconv1(h))
            roi_masks = self.conv2(h)

        return roi_cls_locs, roi_scores, roi_masks

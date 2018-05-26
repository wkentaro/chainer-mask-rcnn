import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.utils import non_maximum_suppression

from .. import functions
from .mask_rcnn import MaskRCNN
from .mask_rcnn import segm_results
from .fpn_roi_mask_head import FPNRoIMaskHead
from .multilevel_region_proposal_network import MultilevelRegionProposalNetwork
from .resnet_fpn_extractor import ResNetFPNExtractor


class MaskRCNNResNetFPN(MaskRCNN):

    fpn = True

    def __init__(self,
                 n_layers,
                 n_fg_class,
                 pretrained_model=None,
                 min_size=600,
                 max_size=1000,
                 ratios=(0.5, 1, 2),
                 anchor_scales=(4, 8, 16, 32),
                 mean=(123.152, 115.903, 103.063),
                 res_initialW=None,
                 rpn_initialW=None,
                 loc_initialW=None,
                 score_initialW=None,
                 mask_initialW=None,
                 proposal_creator_params=None,
                 pooling_func=functions.roi_align_2d,
                 rpn_hidden=1024,
                 ):
        if loc_initialW is None:
            loc_initialW = chainer.initializers.Normal(0.001)
        if score_initialW is None:
            score_initialW = chainer.initializers.Normal(0.01)
        if mask_initialW is None:
            mask_initialW = chainer.initializers.Normal(0.01)
        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)
        if res_initialW is None and pretrained_model:
            res_initialW = chainer.initializers.constant.Zero()

        if proposal_creator_params is None:
            proposal_creator_params = dict(
                min_size=0,
                n_train_pre_nms=2000,
                n_test_pre_nms=1000,
                n_test_post_nms=1000,
            )

        self.n_layers = n_layers

        extractor = ResNetFPNExtractor(
            n_layers=n_layers,
            pretrained_model='auto',
        )
        assert anchor_scales == tuple(extractor.anchor_scales)
        rpn = MultilevelRegionProposalNetwork(
            anchor_scales=extractor.anchor_scales,
            feat_strides=extractor.feat_strides,
            ratios=ratios,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = FPNRoIMaskHead(
            n_fg_class + 1,
            roi_size_bbox=7,
            roi_size_mask=14,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW,
            mask_initialW=mask_initialW,
            pooling_func=pooling_func,
        )

        if len(mean) != 3:
            raise ValueError('The mean must be tuple of RGB values.')
        mean = np.asarray(mean, dtype=np.float32)[:, None, None]

        super(MaskRCNNResNetFPN, self).__init__(
            extractor,
            rpn,
            head,
            mean=mean,
            min_size=min_size,
            max_size=max_size
        )

        if pretrained_model == 'imagenet':
            raise ValueError('Unsupported value of pretrained_model: {}'
                             .format(pretrained_model))
        if pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def __call__(self, x, scale=1.):
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor, levels =\
            self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores, roi_masks = self.head(
            h, rois, roi_indices, levels, self.extractor.spatial_scales)
        return roi_cls_locs, roi_scores, rois, roi_indices, roi_masks

    def _suppress(self, raw_cls_bbox, raw_prob, raw_level=None):
        bbox = list()
        label = list()
        score = list()
        level = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]

            # thresholding by score
            keep = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[keep]
            prob_l = prob_l[keep]
            level_l = raw_level[keep]

            # thresholding by nms
            keep = non_maximum_suppression(
                cls_bbox_l, self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
            level.append(level_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        level = np.concatenate(level, axis=0).astype(np.int32)
        return bbox, label, score, level

    def predict(self, imgs):
        from .mask_rcnn import im_list_to_blob
        prepared_imgs = list()
        scales = list()
        for img in imgs:
            size = img.shape[1:]
            img = self.prepare(img.astype(np.float32))
            scale = img.shape[2] / size[1]
            prepared_imgs.append(img)
            scales.append(scale)
        prepared_imgs = im_list_to_blob(prepared_imgs, fpn=True)

        bboxes = list()
        masks = list()
        labels = list()
        scores = list()
        for img, scale in zip(prepared_imgs, scales):
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                img_var = chainer.Variable(self.xp.asarray(img[None]))
                img_size = img_var.shape[2:]

                h = self.extractor(img_var)
                rpn_locs, rpn_scores, rois, roi_indices, anchor, levels =\
                    self.rpn(h, img_size, scale)
                roi_cls_locs, roi_scores, _, = self.head(
                    h,
                    rois,
                    roi_indices,
                    levels=levels,
                    spatial_scales=self.extractor.spatial_scales,
                    pred_mask=False,
                )
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
            raw_level = cuda.to_cpu(levels)

            bbox, label, score, level = self._suppress(
                raw_cls_bbox, raw_prob, raw_level)

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
                    x=h,
                    rois=rois,
                    roi_indices=roi_indices,
                    levels=levels,
                    spatial_scales=self.extractor.spatial_scales,
                    pred_bbox=False,
                    pred_mask=True,
                )
                roi_masks = F.sigmoid(roi_masks)
            roi_mask = cuda.to_cpu(roi_masks.data)

            mask = segm_results(bbox, label, roi_mask, size[0], size[1],
                                mask_size=self.head.mask_size)
            masks.append(mask)

        return bboxes, masks, labels, scores

# Modified works:
# --------------------------------------------------------
# Copyright (c) 2017 - 2018 Kentaro Wada.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# This is modified work of FasterRCNNVGG16:
# --------------------------------------------------------
# Copyright (c) 2017 Preferred Networks, Inc.
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/chainer/chainercv
# --------------------------------------------------------

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainer.links.model.vision.resnet import BuildingBlock

from .. import functions
from .mask_rcnn import MaskRCNN
from .region_proposal_network import RegionProposalNetwork
from .resnet_extractor import _convert_bn_to_affine
from .resnet_extractor import ResNet101Extractor
from .resnet_extractor import ResNet50Extractor


class MaskRCNNResNet(MaskRCNN):

    feat_stride = 16

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
                 proposal_creator_params=dict(
                     min_size=0,
                     n_test_pre_nms=6000,
                     n_test_post_nms=1000,
                 ),
                 pooling_func=functions.roi_align_2d,
                 rpn_hidden=1024,
                 roi_size=7,
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

        if n_layers == 50:
            extractor = ResNet50Extractor(
                pretrained_model=None if pretrained_model else 'auto',
                remove_layers=['res5', 'fc6'],
            )
        elif n_layers == 101:
            extractor = ResNet101Extractor(
                pretrained_model=None if pretrained_model else 'auto',
                remove_layers=['res5', 'fc6'],
            )
        else:
            raise ValueError

        rpn = RegionProposalNetwork(
            1024, rpn_hidden,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = ResNetRoIHead(
            n_layers=n_layers,
            n_class=n_fg_class + 1,
            roi_size=roi_size,
            spatial_scale=1. / self.feat_stride,
            pretrained_model=None if pretrained_model else 'auto',
            res_initialW=res_initialW,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW,
            mask_initialW=mask_initialW,
            pooling_func=pooling_func,
        )

        if len(mean) != 3:
            raise ValueError('The mean must be tuple of RGB values.')
        mean = np.asarray(mean, dtype=np.float32)[:, None, None]

        super(MaskRCNNResNet, self).__init__(
            extractor,
            rpn,
            head,
            mean=mean,
            min_size=min_size,
            max_size=max_size
        )

        if pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)


class ResNetRoIHead(chainer.Chain):

    mask_size = 14  # Size of the predicted mask.

    def __init__(self, n_layers, n_class, roi_size, spatial_scale,
                 pretrained_model='auto',
                 res_initialW=None, loc_initialW=None, score_initialW=None,
                 mask_initialW=None, pooling_func=functions.roi_align_2d,
                 ):
        # n_class includes the background
        super(ResNetRoIHead, self).__init__()
        with self.init_scope():
            self.res5 = BuildingBlock(
                3, 1024, 512, 2048, stride=roi_size // 7,
                initialW=res_initialW)
            self.cls_loc = L.Linear(2048, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)

            # 7 x 7 x 2048 -> 14 x 14 x 256
            self.deconv6 = L.Deconvolution2D(
                2048, 256, 2, stride=2, initialW=mask_initialW)
            # 14 x 14 x 256 -> 14 x 14 x 20
            n_fg_class = n_class - 1
            self.mask = L.Convolution2D(
                256, n_fg_class, 1, initialW=mask_initialW)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.pooling_func = pooling_func

        _convert_bn_to_affine(self)

        if pretrained_model == 'auto':
            self._copy_imagenet_pretrained_resnet(n_layers)
        else:
            assert pretrained_model is None, \
                'Unsupported pretrained_model: {}'.format(pretrained_model)

    def _copy_imagenet_pretrained_resnet(self, n_layers):
        if n_layers == 50:
            pretrained_model = ResNet50Extractor(pretrained_model='auto')
        elif n_layers == 101:
            pretrained_model = ResNet101Extractor(pretrained_model='auto')
        else:
            raise ValueError
        self.res5.copyparams(pretrained_model.res5)
        _copy_persistent_chain(self.res5, pretrained_model.res5)

    def __call__(self, x, rois, roi_indices, pred_bbox=True, pred_mask=True):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = self.pooling_func(
            x,
            indices_and_rois,
            outh=self.roi_size,
            outw=self.roi_size,
            spatial_scale=self.spatial_scale,
            axes='yx',
        )

        res5 = self.res5(pool)

        roi_cls_locs = None
        roi_scores = None
        roi_masks = None

        if pred_bbox:
            pool5 = F.average_pooling_2d(res5, 7, stride=7)
            roi_cls_locs = self.cls_loc(pool5)
            roi_scores = self.score(pool5)

        if pred_mask:
            deconv6 = F.relu(self.deconv6(res5))
            roi_masks = self.mask(deconv6)

        return roi_cls_locs, roi_scores, roi_masks


def _copy_persistent_link(dst, src):
    for name in dst._persistent:
        d = dst.__dict__[name]
        s = src.__dict__[name]
        if isinstance(d, np.ndarray):
            d[:] = s
        elif isinstance(d, int):
            d = s
        else:
            raise ValueError


def _copy_persistent_chain(dst, src):
    _copy_persistent_link(dst, src)
    for l in dst.children():
        name = l.name
        if (isinstance(dst.__dict__[name], chainer.Chain) and
                isinstance(src.__dict__[name], chainer.Chain)):
            _copy_persistent_chain(dst.__dict__[name], src.__dict__[name])
        elif (isinstance(dst.__dict__[name], chainer.Link) and
                isinstance(src.__dict__[name], chainer.Link)):
            _copy_persistent_link(dst.__dict__[name], src.__dict__[name])

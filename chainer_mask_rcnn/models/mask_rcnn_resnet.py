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

from .. import functions
from .mask_rcnn import MaskRCNN
from .resnet_roi_mask_head import ResNetRoIMaskHead


class MaskRCNNResNet(MaskRCNN):

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

        if proposal_creator_params is None:
            proposal_creator_params = dict(
                min_size=0,
                n_test_pre_nms=6000,
                n_test_post_nms=1000,
            )

        self.n_layers = n_layers

        from .region_proposal_network import RegionProposalNetwork
        from .resnet_extractor import ResNetExtractor
        extractor = ResNetExtractor(
            n_layers=n_layers,
            pretrained_model='auto',
            remove_layers=['res5', 'fc6'],
        )
        rpn = RegionProposalNetwork(
            1024,
            rpn_hidden,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=16,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = ResNetRoIMaskHead(
            n_layers=n_layers,
            n_class=n_fg_class + 1,
            roi_size=roi_size,
            spatial_scale=1 / 16.,
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

        if pretrained_model == 'imagenet':
            raise ValueError('Unsupported value of pretrained_model: {}'
                             .format(pretrained_model))
        if pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

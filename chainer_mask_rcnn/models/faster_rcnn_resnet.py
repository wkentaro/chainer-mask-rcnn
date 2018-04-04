# Modified works:
# --------------------------------------------------------
# Copyright (c) 2017 - 2018 Kentaro Wada.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Original work of FasterRCNN, _copy_persistent_link, _copy_persistent_chain:
# --------------------------------------------------------
# Copyright (c) 2017 Preferred Networks, Inc.
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/chainer/chainercv
# --------------------------------------------------------

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links.model.vision.resnet import BuildingBlock
from chainer.links.model.vision.resnet import ResNet101Layers
from chainer.links.model.vision.resnet import ResNet50Layers
from chainercv.links.model.faster_rcnn.faster_rcnn import FasterRCNN
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.utils import download_model
import numpy as np


class FasterRCNNResNet(FasterRCNN):

    _models = {}
    feat_stride = 16

    def __init__(self,
                 n_layers,
                 n_fg_class=None,
                 pretrained_model=None,
                 min_size=600, max_size=1000,
                 ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                 res_initialW=None, rpn_initialW=None,
                 loc_initialW=None, score_initialW=None,
                 proposal_creator_params=dict(),
                 pooling_func=F.roi_pooling_2d,
                 ):
        if n_fg_class is None:
            if pretrained_model not in self._models:
                raise ValueError(
                    'The n_fg_class needs to be supplied as an argument')
            n_fg_class = self._models[pretrained_model]['n_fg_class']

        if loc_initialW is None:
            loc_initialW = chainer.initializers.Normal(0.001)
        if score_initialW is None:
            score_initialW = chainer.initializers.Normal(0.01)
        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)
        if res_initialW is None and pretrained_model:
            res_initialW = chainer.initializers.constant.Zero()

        if n_layers == 50:
            self._ResNetLayers = ResNet50Layers
        elif n_layers == 101:
            self._ResNetLayers = ResNet101Layers
        else:
            raise ValueError

        class Extractor(self._ResNetLayers):

            def __init__(self, *args, **kwargs):
                super(Extractor, self).__init__(*args, **kwargs)
                # Remove no need layers to save memory
                self.res5 = chainer.Link()
                self.fc6 = chainer.Link()

            def __call__(self, x):
                pick = 'res4'
                with chainer.using_config('train', False):
                    feat = super(Extractor, self).__call__(x, layers=[pick])
                return feat[pick]

        extractor = Extractor(pretrained_model=None)
        rpn = RegionProposalNetwork(
            1024, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = ResNetRoIHead(
            n_fg_class + 1,
            roi_size=7, spatial_scale=1. / self.feat_stride,
            res_initialW=res_initialW,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW,
            pooling_func=pooling_func,
        )

        super(FasterRCNNResNet, self).__init__(
            extractor,
            rpn,
            head,
            mean=np.array([123.152, 115.903, 103.063],
                          dtype=np.float32)[:, None, None],
            min_size=min_size,
            max_size=max_size
        )

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model == 'imagenet':
            self._copy_imagenet_pretrained_resnet()
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def _copy_imagenet_pretrained_resnet(self):
        pretrained_model = self._ResNetLayers(pretrained_model='auto')

        self.extractor.conv1.copyparams(pretrained_model.conv1)
        # The pretrained weights are trained to accept BGR images.
        # Convert weights so that they accept RGB images.
        self.extractor.conv1.W.data[:] = \
            self.extractor.conv1.W.data[:, ::-1]

        self.extractor.bn1.copyparams(pretrained_model.bn1)
        _copy_persistent_chain(self.extractor.bn1, pretrained_model.bn1)

        self.extractor.res2.copyparams(pretrained_model.res2)
        _copy_persistent_chain(self.extractor.res2, pretrained_model.res2)

        self.extractor.res3.copyparams(pretrained_model.res3)
        _copy_persistent_chain(self.extractor.res3, pretrained_model.res3)

        self.extractor.res4.copyparams(pretrained_model.res4)
        _copy_persistent_chain(self.extractor.res4, pretrained_model.res4)

        self.head.res5.copyparams(pretrained_model.res5)
        _copy_persistent_chain(self.head.res5, pretrained_model.res5)


class FasterRCNNResNet50(FasterRCNNResNet):

    def __init__(self, *args, **kwargs):
        return super(FasterRCNNResNet50, self).__init__(
            n_layers=50, *args, **kwargs)


class FasterRCNNResNet101(FasterRCNNResNet):

    def __init__(self, *args, **kwargs):
        return super(FasterRCNNResNet101, self).__init__(
            n_layers=101, *args, **kwargs)


class ResNetRoIHead(chainer.Chain):

    def __init__(self, n_class, roi_size, spatial_scale,
                 res_initialW=None, loc_initialW=None, score_initialW=None,
                 pooling_func=F.roi_pooling_2d):
        # n_class includes the background
        super(ResNetRoIHead, self).__init__()
        with self.init_scope():
            self.res5 = BuildingBlock(
                3, 1024, 512, 2048, stride=2, initialW=res_initialW)
            self.cls_loc = L.Linear(2048, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.pooling_func = pooling_func

    def __call__(self, x, rois, roi_indices):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = _roi_pooling_2d_yx(
            x, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale, pooling_func=self.pooling_func)

        with chainer.using_config('train', False):
            res5 = self.res5(pool)
        pool5 = _global_average_pooling_2d(res5)
        roi_cls_locs = self.cls_loc(pool5)
        roi_scores = self.score(pool5)
        return roi_cls_locs, roi_scores


def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale,
                       pooling_func):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = pooling_func(x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = F.reshape(h, (n, channel))
    return h


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

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

import collections

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainer.links.model.vision.resnet import BuildingBlock
from chainer.links.model.vision.resnet import ResNet101Layers
from chainer.links.model.vision.resnet import ResNet50Layers
from chainercv.utils import download_model

from .. import functions
from .. import links
from .faster_rcnn_resnet import _copy_persistent_chain
from .mask_rcnn import MaskRCNN
from .region_proposal_network import RegionProposalNetwork


def _get_affine_from_bn(bn):
    channels = bn.gamma.size
    bn_mean = bn.avg_mean
    bn_var = bn.avg_var
    scale = bn.gamma.data
    bias = bn.beta.data
    xp = chainer.cuda.get_array_module(bn_var)
    std = xp.sqrt(bn_var + 1e-5)
    new_scale = scale / std
    new_bias = bias - bn_mean * new_scale
    affine = links.AffineChannel2D(channels)
    affine.W.data[:] = new_scale[:]
    affine.b.data[:] = new_bias[:]
    return affine


class ResNet50Extractor(ResNet50Layers):

    mode = 'all'

    def __init__(self, *args, **kwargs):
        remove_layers = kwargs.pop('remove_layers', True)
        super(ResNet50Extractor, self).__init__(*args, **kwargs)
        if remove_layers:
            # Remove no need layers to save memory
            delattr(self, 'res5')
            delattr(self, 'fc6')
        # BatchNormalization -> AffineChannel2D
        for name, link in self.namedlinks():
            if not isinstance(link, L.BatchNormalization):
                continue
            for key in name.split('/')[:-1]:
                if key == '':
                    parent = self
                else:
                    parent = getattr(parent, key)
            key = name.split('/')[-1]
            delattr(parent, key)
            link2 = _get_affine_from_bn(link)
            parent.add_link(key, link2)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, 3, stride=2, pad=1)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
        ])

    def __call__(self, x):
        assert self.mode in ['head', 'res3+', 'res4+', 'all']
        h = x
        for key, funcs in self.functions.items():
            for func in funcs:
                h = func(h)
            if key == 'res2' and self.mode == 'res3+':
                h.unchain_backward()
            if key == 'res3' and self.mode == 'res4+':
                h.unchain_backward()
            if key == 'res4':
                if self.mode == 'head':
                    h.unchain_backward()
                break
        return h


class ResNet101Extractor(ResNet101Layers):

    mode = 'all'

    def __init__(self, *args, **kwargs):
        remove_layers = kwargs.pop('remove_layers', True)
        super(ResNet101Extractor, self).__init__(*args, **kwargs)
        if remove_layers:
            # Remove no need layers to save memory
            delattr(self, 'res5')
            delattr(self, 'fc6')
        # BatchNormalization -> AffineChannel2D
        for name, link in self.namedlinks():
            if not isinstance(link, L.BatchNormalization):
                continue
            for key in name.split('/')[:-1]:
                if key == '':
                    parent = self
                else:
                    parent = getattr(parent, key)
            key = name.split('/')[-1]
            delattr(parent, key)
            link2 = _get_affine_from_bn(link)
            parent.add_link(key, link2)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, F.relu]),
            ('pool1', [lambda x: F.max_pooling_2d(x, ksize=3, stride=2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
        ])

    def __call__(self, x):
        assert self.mode in ['head', 'res3+', 'res4+', 'all']
        h = x
        for key, funcs in self.functions.items():
            for func in funcs:
                h = func(h)
            if key == 'res2' and self.mode == 'res3+':
                h.unchain_backward()
            if key == 'res3' and self.mode == 'res4+':
                h.unchain_backward()
            if key == 'res4':
                if self.mode == 'head':
                    h.unchain_backward()
                break
        return h


class MaskRCNNResNet(MaskRCNN):

    feat_stride = 16
    _models = {}

    def __init__(self,
                 n_layers,
                 n_fg_class=None,
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
                 proposal_creator_params=dict(
                     min_size=0,
                     n_test_pre_nms=6000,
                     n_test_post_nms=1000,
                 ),
                 pooling_func=functions.roi_align_2d,
                 rpn_hidden=1024,
                 roi_size=7,
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
            extractor = ResNet50Extractor(pretrained_model=None)
        elif n_layers == 101:
            extractor = ResNet101Extractor(pretrained_model=None)
        else:
            raise ValueError
        self._n_layers = n_layers

        rpn = RegionProposalNetwork(
            1024, rpn_hidden,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = ResNetRoIHead(
            n_fg_class + 1,
            roi_size=roi_size,
            spatial_scale=1. / self.feat_stride,
            res_initialW=res_initialW,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW,
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

        if pretrained_model in self._models:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model == 'imagenet':
            self._copy_imagenet_pretrained_resnet()
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    def _copy_imagenet_pretrained_resnet(self):
        if self._n_layers == 50:
            pretrained_model = ResNet50Extractor(
                pretrained_model='auto', remove_layers=False)
        elif self._n_layers == 101:
            pretrained_model = ResNet101Extractor(
                pretrained_model='auto', remove_layers=False)
        else:
            raise ValueError

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


class ResNetRoIHead(chainer.Chain):

    def __init__(self, n_class, roi_size, spatial_scale,
                 res_initialW=None, loc_initialW=None, score_initialW=None,
                 pooling_func=functions.roi_align_2d):
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
                2048, 256, 2, stride=2,
                initialW=chainer.initializers.Normal(0.01))
            # 14 x 14 x 256 -> 14 x 14 x 20
            n_fg_class = n_class - 1
            self.mask = L.Convolution2D(
                256, n_fg_class, 1,
                initialW=chainer.initializers.Normal(0.01))

        for name, link in self.res5.namedlinks():
            if not isinstance(link, L.BatchNormalization):
                continue
            for key in name.split('/')[:-1]:
                if key == '':
                    parent = self.res5
                else:
                    parent = getattr(parent, key)
            key = name.split('/')[-1]
            delattr(parent, key)
            link2 = _get_affine_from_bn(link)
            parent.add_link(key, link2)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self._pooling_func = pooling_func

    def __call__(self, x, rois, roi_indices, pred_bbox=True, pred_mask=True):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = _roi_pooling_2d_yx(
            x, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale, self._pooling_func)

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


def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale,
                       pooling_func):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = pooling_func(x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool

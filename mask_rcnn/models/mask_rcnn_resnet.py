import os.path as osp

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainer.links.model.vision.resnet import BuildingBlock
from chainer.links.model.vision.resnet import ResNet101Layers
from chainer.links.model.vision.resnet import ResNet50Layers
from chainercv.links.model.faster_rcnn.region_proposal_network \
    import RegionProposalNetwork
from chainercv.utils import download_model

from .. import functions
from .faster_rcnn_resnet import _copy_persistent_chain
from .faster_rcnn_resnet import _global_average_pooling_2d
from .faster_rcnn_resnet import FasterRCNNResNet101
from .faster_rcnn_resnet import FasterRCNNResNet50
from .mask_rcnn import MaskRCNN


class MaskRCNNResNet(MaskRCNN):

    feat_stride = 16
    _models = {}

    def __init__(self,
                 n_layers,
                 n_fg_class=None,
                 pretrained_model=None,
                 min_size=600, max_size=1000,
                 ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                 res_initialW=None, rpn_initialW=None,
                 loc_initialW=None, score_initialW=None,
                 proposal_creator_params=dict(),
                 pooling_func=functions.roi_align_2d,
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
            self._FasterRCNN = FasterRCNNResNet50
        elif n_layers == 101:
            self._ResNetLayers = ResNet101Layers
            self._FasterRCNN = FasterRCNNResNet101
        else:
            raise ValueError
        self._n_layers = n_layers

        class Extractor(self._ResNetLayers):

            mode = 'all'

            def __init__(self, *args, **kwargs):
                super(Extractor, self).__init__(*args, **kwargs)
                # Remove no need layers to save memory
                delattr(self, 'res5')
                delattr(self, 'fc6')
                self.res5 = self.fc6 = None

            def __call__(self, x):
                assert self.mode in ['head', 'res4+', 'all']
                h = x
                with chainer.using_config('train', False):
                    for key, funcs in self.functions.items():
                        for func in funcs:
                            h = func(h)
                        if key == 'res3' and self.mode == 'res4+':
                            h.unchain_backward()
                        if key == 'res4':
                            if self.mode == 'head':
                                h.unchain_backward()
                            break
                return h

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

        super(MaskRCNNResNet, self).__init__(
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

    def _copy_voc_pretrained_rpn(self, pretrained_model):
        pretrained_model = self._FasterRCNN(
            n_fg_class=n_fg_class, pretrained_model=pretrained_model)

        self.extractor.conv1.copyparams(pretrained_model.extractor.conv1)
        _copy_persistent_chain(self.extractor.conv1,
                               pretrained_model.extractor.conv1)
        self.extractor.bn1.copyparams(pretrained_model.extractor.bn1)
        _copy_persistent_chain(self.extractor.bn1,
                               pretrained_model.extractor.bn1)
        self.extractor.res2.copyparams(pretrained_model.extractor.res2)
        _copy_persistent_chain(self.extractor.res2,
                               pretrained_model.extractor.res2)
        self.extractor.res3.copyparams(pretrained_model.extractor.res3)
        _copy_persistent_chain(self.extractor.res3,
                               pretrained_model.extractor.res3)
        self.extractor.res4.copyparams(pretrained_model.extractor.res4)
        _copy_persistent_chain(self.extractor.res4,
                               pretrained_model.extractor.res4)

        self.rpn.copyparams(pretrained_model.rpn)
        _copy_persistent_chain(self.rpn, pretrained_model.rpn)

        pretrained_model = self._ResNetLayers(pretrained_model='auto')

        self.head.res5.copyparams(pretrained_model.res5)
        _copy_persistent_chain(self.head.res5, pretrained_model.res5)

    def _copy_voc_pretrained_faster_rcnn(self, pretrained_model):
        pretrained_model = self._FasterRCNN(
            n_fg_class=n_fg_class, pretrained_model=pretrained_model)

        self.extractor.copyparams(pretrained_model.extractor)
        _copy_persistent_chain(self.extractor, pretrained_model.extractor)

        self.rpn.copyparams(pretrained_model.rpn)
        _copy_persistent_chain(self.rpn, pretrained_model.rpn)

        self.head.res5.copyparams(pretrained_model.head.res5)
        _copy_persistent_chain(self.head.res5, pretrained_model.head.res5)

        self.head.cls_loc.copyparams(pretrained_model.head.cls_loc)
        _copy_persistent_chain(self.head.cls_loc,
                               pretrained_model.head.cls_loc)

        self.head.score.copyparams(pretrained_model.head.score)
        _copy_persistent_chain(self.head.score, pretrained_model.head.score)


class ResNetRoIHead(chainer.Chain):

    def __init__(self, n_class, roi_size, spatial_scale,
                 res_initialW=None, loc_initialW=None, score_initialW=None,
                 pooling_func=functions.roi_align_2d):
        # n_class includes the background
        super(ResNetRoIHead, self).__init__()
        with self.init_scope():
            self.res5 = BuildingBlock(
                3, 1024, 512, 2048, stride=1, initialW=res_initialW)
            self.cls_loc = L.Linear(2048, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)

            # 7 x 7 x 2048 -> 14 x 14 x 256
            self.deconv6 = L.Deconvolution2D(
                2048, 256, 2, stride=2,
                initialW=chainer.initializers.Normal(0.01))
            # 14 x 14 x 256 -> 14 x 14 x 20
            n_fg_class = n_class - 1
            self.mask = L.Convolution2D(
                256, n_fg_class, 1, initialW=chainer.initializers.Normal(0.01))

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self._pooling_func = pooling_func

    def __call__(self, x, rois, roi_indices, pred_bbox=True, pred_mask=True,
                 res5=None, return_res5=False):
        if res5 is None:
            roi_indices = roi_indices.astype(np.float32)
            indices_and_rois = self.xp.concatenate(
                (roi_indices[:, None], rois), axis=1)
            pool = _roi_pooling_2d_yx(
                x, indices_and_rois, self.roi_size, self.roi_size,
                self.spatial_scale, self._pooling_func)

            with chainer.using_config('train', False):
                res5 = self.res5(pool)

        roi_cls_locs = None
        roi_scores = None
        roi_masks = None

        if pred_bbox:
            pool5 = _global_average_pooling_2d(res5)
            roi_cls_locs = self.cls_loc(pool5)
            roi_scores = self.score(pool5)

        if pred_mask:
            deconv6 = F.relu(self.deconv6(res5))
            roi_masks = self.mask(deconv6)

        if return_res5:
            return roi_cls_locs, roi_scores, roi_masks, res5
        else:
            return roi_cls_locs, roi_scores, roi_masks


def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale,
                       pooling_func):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = pooling_func(x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool

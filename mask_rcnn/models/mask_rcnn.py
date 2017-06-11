import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from chainercv.links.model.faster_rcnn.faster_rcnn_vgg import \
    _roi_pooling_2d_yx, VGG16FeatureExtractor
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork


class MaskRCNN(chainer.Chain):

    feat_stride = 16

    def __init__(self):
        super(MaskRCNN, self).__init__()
        with self.init_scope():
            self.extractor = VGG16FeatureExtractor(
                initialW=chainer.initializers.Normal(0.01),
            )
            self.rpn = RegionProposalNetwork(
                512, 512,
                ratios=[0.5, 1, 2],
                anchor_scales=[8, 16, 32],
                feat_stride=self.feat_stride,
                initialW=chainer.initializers.Normal(0.01),
                proposal_creator_params={},
            )
            self.head = VGG16RoIHead(
                n_class=21,  # 1 bg + 20 object classes
                roi_size=7,
                spatial_scale=1. / self.feat_stride,
                vgg_initialW=chainer.initializers.Normal(0.01),
                loc_initialW=chainer.initializers.Normal(0.001),
                score_initialW=chainer.initializers.Normal(0.01),
            )

    def __call__(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(
            h, img_size, scale)
        roi_cls_locs, roi_scores, roi_masks = self.head(
            h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices, roi_masks

    def _copy_imagenet_pretrained_vgg16(self):
        pretrained_model = L.VGG16Layers()
        self.extractor.conv1_1.copyparams(pretrained_model.conv1_1)
        self.extractor.conv1_2.copyparams(pretrained_model.conv1_2)
        self.extractor.conv2_1.copyparams(pretrained_model.conv2_1)
        self.extractor.conv2_2.copyparams(pretrained_model.conv2_2)
        self.extractor.conv3_1.copyparams(pretrained_model.conv3_1)
        self.extractor.conv3_2.copyparams(pretrained_model.conv3_2)
        self.extractor.conv3_3.copyparams(pretrained_model.conv3_3)
        self.extractor.conv4_1.copyparams(pretrained_model.conv4_1)
        self.extractor.conv4_2.copyparams(pretrained_model.conv4_2)
        self.extractor.conv4_3.copyparams(pretrained_model.conv4_3)
        self.extractor.conv5_1.copyparams(pretrained_model.conv5_1)
        self.extractor.conv5_2.copyparams(pretrained_model.conv5_2)
        self.extractor.conv5_3.copyparams(pretrained_model.conv5_3)
        self.head.fc6.copyparams(pretrained_model.fc6)
        self.head.fc7.copyparams(pretrained_model.fc7)


class VGG16RoIHead(chainer.Chain):

    def __init__(self, n_class, roi_size, spatial_scale,
                 vgg_initialW=None, loc_initialW=None, score_initialW=None):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()
        with self.init_scope():
            # 25088 = 7 x 7 x 512
            self.fc6 = L.Linear(25088, 4096, initialW=vgg_initialW)
            self.fc7 = L.Linear(4096, 4096, initialW=vgg_initialW)
            self.cls_loc = L.Linear(4096, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(4096, n_class, initialW=score_initialW)

            # 7 x 7 x 512 -> 14 x 14 x 256
            self.deconv6 = L.Deconvolution2D(
                512, 256, 2, stride=2,
                initialW=chainer.initializers.Normal(0.01))
            # 14 x 14 x 256 -> 14 x 14 x 20
            n_fg_class = n_class - 1
            self.mask = L.Convolution2D(
                256, n_fg_class, 1, initialW=chainer.initializers.Normal(0.01))

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

    def __call__(self, x, rois, roi_indices):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = _roi_pooling_2d_yx(
            x, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale)

        fc6 = F.relu(self.fc6(pool))
        fc7 = F.relu(self.fc7(fc6))
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        deconv6 = F.relu(self.deconv6(pool))
        roi_masks = self.mask(deconv6)
        return roi_cls_locs, roi_scores, roi_masks

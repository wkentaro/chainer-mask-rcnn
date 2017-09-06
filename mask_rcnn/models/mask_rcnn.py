import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.utils import non_maximum_suppression
import numpy as np

from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.links.model.vgg.vgg16 import VGG16


class MaskRcnn(chainer.Chain):

    feat_stride = 16
    _models = {}

    def __init__(self, pretrained_model=None):
        super(MaskRcnn, self).__init__()
        with self.init_scope():
            self.extractor = VGG16(initialW=chainer.initializers.Zero())
            self.extractor.feature_names = 'conv5_3'
            self.extractor.remove_unused()
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

        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

        self.use_preset('visualize')

        if pretrained_model in self._models:
            pretrained_model = self._models[pretrained_model]
            chainer.serializers.load_npz(pretrained_model, self)
        elif pretrained_model == 'imagenet':
            self._copy_imagenet_pretrained_vgg16()
        elif pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def __call__(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(
            h, img_size, scale)
        roi_cls_locs, roi_scores, roi_masks = self.head(
            h, rois, roi_indices)
        return roi_cls_locs, roi_scores, roi_masks, rois, roi_indices

    def _copy_imagenet_pretrained_vgg16(self):
        pretrained_model = VGG16(pretrained_model='imagenet')
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
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob, raw_mask):
        bbox = list()
        label = list()
        score = list()
        mask = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask_l = raw_mask[:, l - 1]
            # thresh score
            keep = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[keep]
            prob_l = prob_l[keep]
            mask_l = mask_l[keep]
            # thresh nms
            keep = non_maximum_suppression(
                cls_bbox_l, self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
            mask_l = mask_l[keep]
            mask_l = mask_l >= 0.5
            mask.append(mask_l)
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        mask = np.concatenate(mask, axis=0).astype(bool)
        return bbox, label, score, mask

    def predict(self, imgs):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :obj:`(y_min, x_min, y_max, x_max)` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        bboxes = list()
        labels = list()
        scores = list()
        masks = list()
        for img in imgs:
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                # prepare image
                MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])
                img = img.astype(np.float32)
                img -= MEAN_BGR[::-1]
                img = img.transpose(2, 0, 1)  # H, W, C -> C, H, W

                img_var = chainer.Variable(self.xp.asarray(img[None]))
                H, W = img_var.shape[2:]
                roi_cls_locs, roi_scores, roi_masks, rois, _ = \
                    self(img_var)
            # We are assuming that batch size is 1.
            roi_cls_loc = roi_cls_locs.data
            roi_score = roi_scores.data
            roi_mask = roi_masks.data
            roi = rois

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = self.xp.tile(self.xp.asarray(self.loc_normalize_mean),
                                self.n_class)
            std = self.xp.tile(self.xp.asarray(self.loc_normalize_std),
                               self.n_class)
            roi_cls_loc = (roi_cls_loc * std + mean).astype(np.float32)
            roi_cls_loc = roi_cls_loc.reshape((-1, self.n_class, 4))
            roi = self.xp.broadcast_to(roi[:, None], roi_cls_loc.shape)
            cls_bbox = loc2bbox(roi.reshape((-1, 4)),
                                roi_cls_loc.reshape((-1, 4)))
            cls_bbox = cls_bbox.reshape((-1, self.n_class * 4))
            # clip bounding box
            cls_bbox[:, 0::2] = self.xp.clip(cls_bbox[:, 0::2], 0, H)
            cls_bbox[:, 1::2] = self.xp.clip(cls_bbox[:, 1::2], 0, W)

            prob = F.softmax(roi_score).data
            mask = F.sigmoid(roi_mask).data

            raw_cls_bbox = cuda.to_cpu(cls_bbox)
            raw_prob = cuda.to_cpu(prob)
            raw_mask = cuda.to_cpu(mask)

            bbox, label, score, mask = self._suppress(
                raw_cls_bbox, raw_prob, raw_mask)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
            masks.append(mask)

        return bboxes, labels, scores, masks


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
        """Predict class scores and mask in ROIs.

        Parameters
        ----------
        rois: Variable
            (y1, x1, y2, x2)
        """
        roi_indices = roi_indices.astype(np.float32)
        # (batch_index, y1, x1, y2, x2)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = _roi_align_2d_yx(
            x, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale)

        fc6 = F.relu(self.fc6(pool))
        fc7 = F.relu(self.fc7(fc6))
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        deconv6 = F.relu(self.deconv6(pool))
        roi_masks = self.mask(deconv6)
        return roi_cls_locs, roi_scores, roi_masks


def _roi_align_2d_yx(x, indices_and_rois, outh, outw, spatial_scale):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = F.roi_align_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool

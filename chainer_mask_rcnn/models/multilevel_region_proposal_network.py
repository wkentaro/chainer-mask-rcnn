# orignal code from:
# https://raw.githubusercontent.com/katotetsuro/chainer-maskrcnn/master/chainer_maskrcnn/model/rpn/multilevel_region_proposal_network.py  # NOQA

# original code by chainercv
# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/faster_rcnn/region_proposal_network.py
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainercv.links.model.faster_rcnn.region_proposal_network \
    import _enumerate_shifted_anchor

from chainercv.links.model.faster_rcnn.utils.generate_anchor_base import \
    generate_anchor_base
from chainercv.links.model.faster_rcnn.utils.proposal_creator \
    import ProposalCreator


# original code from Detectron
# https://github.com/facebookresearch/Detectron/blob/master/lib/modeling/FPN.py
def map_rois_to_fpn_levels(rois, k_min=0, k_max=4):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    roi: assume (R, 4), y_min, x_min, y_max, x_max
    """
    # Compute level ids
    xp = chainer.backends.cuda.get_array_module(rois)
    area = xp.prod(rois[:, 2:] - rois[:, :2], axis=1)
    s = xp.sqrt(area)
    s0 = 224
    lvl0 = 4

    # Eqn.(1) in FPN paper
    target_lvls = xp.floor(lvl0 + xp.log2(s / s0 + 1e-6))
    target_lvls = xp.clip(target_lvls, k_min, k_max)
    return target_lvls


class MultilevelRegionProposalNetwork(chainer.Chain):

    """Region Proposal Network introduced in Faster R-CNN.
        This is Region Proposal Network introduced in Faster R-CNN [#]_.
        This takes features extracted from images and propose
        class agnostic bounding boxes around "objects".
        .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
        Faster R-CNN: Towards Real-Time Object Detection with \
        Region Proposal Networks. NIPS 2015.
        Args:
            in_channels (int): The channel size of input.
            mid_channels (int): The channel size of the intermediate tensor.
            ratios (list of floats): This is ratios of width to height of
                the anchors.

            anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
            initialW (callable): Initial weight value. If :obj:`None` then this
                function uses Gaussian distribution scaled by 0.1 to
                initialize weight.
                May also be a callable that takes an array
                and edits its values.
            proposal_creator_params (dict): Key valued paramters for
                :class:`~chainercv.links.model.faster_rcnn.ProposalCreator`.
        .. seealso::
            :class:`~chainercv.links.model.faster_rcnn.ProposalCreator`
        """

    def __init__(
            self, anchor_scales, feat_strides,
            in_channels=256, mid_channels=256, ratios=[0.5, 1, 2],
            initialW=None, proposal_creator_params=dict()):
        if len(anchor_scales) != len(feat_strides):
            raise ValueError(
                'length of anchor_scales and feat_strides should be same!')
        self.anchor_bases = [generate_anchor_base(
            anchor_scales=[s], ratios=ratios) for s in anchor_scales]
        self.feat_strides = feat_strides
        self.proposal_layer = ProposalCreator(**proposal_creator_params)
        # note: to share conv layers,
        # number of output channel should be same in all levels.
        # debug
        a = self.anchor_bases[0]
        for ab in self.anchor_bases[1:]:
            assert a.shape == ab.shape
        n_anchor = self.anchor_bases[0].shape[0]
        super(MultilevelRegionProposalNetwork, self).__init__()
        with self.init_scope():
            # note: according fpn paper, parameters are sharable among levels.
            self.conv = L.Convolution2D(
                in_channels, mid_channels, 3, 1, 1, initialW=initialW)
            self.score = L.Convolution2D(
                mid_channels, n_anchor, 1, 1, 0, initialW=initialW)
            self.loc = L.Convolution2D(
                mid_channels, n_anchor * 4, 1, 1, 0, initialW=initialW)

    def __call__(self, xs, img_size, scale=1.):
        """Forward Region Proposal Network.
        Here are notations.
        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.
        Args:
            xs (list of ~chainer.Variable):
                The Features extracted from images in multilevel.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.
        Returns:
            (~chainer.Variable, ~chainer.Variable, array, array, array):
            This is a tuple of five following values.
            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.
        """

        locs = []
        scores = []
        anchors = []
        for i, x in enumerate(xs):
            n, _, hh, ww = x.shape
            anchor = _enumerate_shifted_anchor(
                self.xp.array(self.anchor_bases[i]),
                self.feat_strides[i], hh, ww)
            n_anchor = anchor.shape[0] // (hh * ww)
            h = F.relu(self.conv(x))

            rpn_locs = self.loc(h)
            rpn_locs = rpn_locs.transpose((0, 2, 3, 1)).reshape((n, -1, 4))

            rpn_scores = self.score(h)
            rpn_scores = rpn_scores.transpose((0, 2, 3, 1))
            assert rpn_scores.shape == (n, hh, ww, n_anchor)
            rpn_scores = rpn_scores.reshape((n, -1))

            locs.append(rpn_locs)
            scores.append(rpn_scores)
            anchors.append(anchor)

        # chainer.functions's default axis=1, but explicitly for myself.
        locs = F.concat(locs, axis=1)
        scores = F.concat(scores, axis=1)
        anchors = self.xp.concatenate(anchors, axis=0)

        rois = []
        roi_indices = []
        for i in range(n):
            roi = self.proposal_layer(
                locs[i].array,
                scores[i].array,
                anchors,
                img_size,
                scale=scale,
            )
            batch_index = i * self.xp.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = self.xp.concatenate(rois, axis=0)
        roi_indices = self.xp.concatenate(roi_indices, axis=0)
        levels = map_rois_to_fpn_levels(rois, k_min=0, k_max=len(xs) - 1)
        return locs, scores, rois, roi_indices, anchors, levels

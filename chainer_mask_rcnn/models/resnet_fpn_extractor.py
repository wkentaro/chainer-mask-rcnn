# https://github.com/katotetsuro/chainer-maskrcnn/blob/master/chainer_maskrcnn/model/extractor/feature_pyramid_network.py

import chainer
import chainer.links as L
import chainer.functions as F

from .resnet_extractor import ResNetExtractor


class ResNetFPNExtractor(chainer.Chain):

    # determined by network architecture (where stride >1 occurs.)
    feat_strides = [4, 8, 16, 32, 64]
    # inverse of feat_strides. used in RoIAlign to calculate x
    # in Image Coord to x' in feature map
    spatial_scales = [1. / x for x in feat_strides]
    # from FPN paper.
    anchor_sizes = [32, 64, 128, 256, 512]
    # anchor_sizes / anchor_base (=16)
    anchor_scales = [x / 16. for x in anchor_sizes]

    def __init__(self, n_layers, pretrained_model):
        super(ResNetFPNExtractor, self).__init__()
        with self.init_scope():
            # bottom up
            self.extractor = ResNetExtractor(
                n_layers=n_layers,
                pretrained_model=pretrained_model,
                remove_layers=['fc6'],
            )

            # top layer (reduce channel)
            # XXX: conv_top
            self.toplayer = L.Convolution2D(
                in_channels=2048, out_channels=256, ksize=1, stride=1, pad=0)

            # conv layer for top-down pathway
            # XXX: posthoc_modules
            self.conv_p5 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1)
            self.conv_p4 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1)
            self.conv_p3 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1)
            self.conv_p2 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1)

            # lateral connection
            # XXX: topdown_lateral_modules
            self.lat_p4 = L.Convolution2D(
                in_channels=1024, out_channels=256, ksize=1, stride=1, pad=0)
            self.lat_p3 = L.Convolution2D(
                in_channels=512, out_channels=256, ksize=1, stride=1, pad=0)
            self.lat_p2 = L.Convolution2D(
                in_channels=256, out_channels=256, ksize=1, stride=1, pad=0)

    def __call__(self, x):
        # bottom-up pathway
        h = x
        for func in self.extractor.functions['conv1']:
            h = func(h)
        h = self.extractor.functions['pool1'][0](h)
        c2 = self.extractor.functions['res2'][0](h)
        c3 = self.extractor.functions['res3'][0](c2)
        c4 = self.extractor.functions['res4'][0](c3)
        c5 = self.extractor.functions['res5'][0](c4)

        i5 = self.toplayer(c5)
        # TODO(wkentaro): interpolation should be nearest (instead of linear)
        i4 = F.unpooling_2d(i5, 2, outsize=c4.shape[2:4]) + self.lat_p4(c4)
        i3 = F.unpooling_2d(i4, 2, outsize=c3.shape[2:4]) + self.lat_p3(c3)
        i2 = F.unpooling_2d(i3, 2, outsize=c2.shape[2:4]) + self.lat_p2(c2)

        # top-down
        p5 = self.conv_p5(i5)
        p4 = self.conv_p4(i4)
        p3 = self.conv_p3(i3)
        p2 = self.conv_p2(i2)

        # other
        p6 = F.max_pooling_2d(p5, ksize=1, stride=2, pad=0, cover_all=False)

        # fine to coarse
        return p2, p3, p4, p5, p6

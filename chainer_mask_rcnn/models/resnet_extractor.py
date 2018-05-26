import collections
import os.path as osp

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links.model.vision.resnet import ResNet50Layers
from chainer.links.model.vision.resnet import ResNet101Layers

import fcn

from .. import links


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


def _convert_bn_to_affine(chain):
    for name, link in chain.namedlinks():
        if not isinstance(link, L.BatchNormalization):
            continue
        for key in name.split('/')[:-1]:
            if key == '':
                parent = chain
            else:
                parent = getattr(parent, key)
        key = name.split('/')[-1]
        delattr(parent, key)
        link2 = _get_affine_from_bn(link)
        parent.add_link(key, link2)


class ResNetExtractorBase(object):

    mode = 'all'

    def _init_layers(self, remove_layers):
        if remove_layers:
            # Remove no need layers to save memory
            delattr(self, 'res5')
            delattr(self, 'fc6')
        _convert_bn_to_affine(self)

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


class ResNet50Extractor(ResNetExtractorBase, ResNet50Layers):

    def __init__(self, *args, **kwargs):
        remove_layers = kwargs.pop('remove_layers', True)
        super(ResNet50Extractor, self).__init__(*args, **kwargs)
        self._init_layers(remove_layers)

        root = chainer.dataset.get_dataset_directory('pfnet/chainer/models')
        self.model_path = osp.join(root, 'ResNet-50-model.npz')
        if not osp.exists(self.model_path):
            self.download()

    def download(self):
        url = 'https://drive.google.com/uc?id=1hSGnWZX_kjEWlfvi0fCHc8sczHio0i-t'  # NOQA
        md5 = '841b996a74049800cf0749ac97ab7eba'
        fcn.data.cached_download(url, self.model_path, md5)


class ResNet101Extractor(ResNetExtractorBase, ResNet101Layers):

    def __init__(self, *args, **kwargs):
        remove_layers = kwargs.pop('remove_layers', True)
        super(ResNet101Extractor, self).__init__(*args, **kwargs)
        self._init_layers(remove_layers)

        root = chainer.dataset.get_dataset_directory('pfnet/chainer/models')
        self.model_path = osp.join(root, 'ResNet-101-model.npz')
        if not osp.exists(self.model_path):
            self.download()

    def download(self):
        url = 'https://drive.google.com/uc?id=1c-wtuSDWmBCUTfNKLrQAIjrBMNMW4b7q'  # NOQA
        md5 = '2220786332e361fd7f956d9bf2f9d328'
        fcn.data.cached_download(url, self.model_path, md5)

import chainer
from chainer import cuda
from chainer import functions
from chainer import initializers

import numpy


class AffineChannel2D(chainer.Link):

    """A simple channel-wise affine transformation operation"""

    def __init__(self, channels):
        super(AffineChannel2D, self).__init__()
        with self.init_scope():
            self.W = chainer.variable.Parameter(
                initializers.One(), (channels,))
            self.b = chainer.variable.Parameter(
                initializers.Zero(), (channels,))

    def __call__(self, x):
        W = functions.reshape(self.W, (1, -1, 1, 1))
        b = functions.reshape(self.b, (1, -1, 1, 1))
        return affine_channel_2d(x, W, b)
        # return affine_channel_2d_naive(x, W, b)  # use too large memory


class AffineChannel2DFunction(chainer.Function):

    def forward(self, inputs):
        self.retain_inputs((0, 1, 2))
        xp = cuda.get_array_module(inputs)
        x, W, b = inputs
        if xp is numpy:
            y = W * x + b
        else:
            y = cuda.elementwise(
                'T x, T W, T b', 'T y',
                'y = W * x + b', 'affine_fwd'
            )(x, W, b)
        return y,

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(inputs)
        x, W, b = inputs
        gy, = gy

        if xp is numpy:
            gx = W * gy
            gW = x * gy
        else:
            gx, gW = cuda.elementwise(
                'T x, T W, T b, T gy', 'T gx, T gW',
                'gx = W * gy; gW = x * gy;', 'affine_bwd_x'
            )(x, W, b, gy)
        gb = gy

        gW = xp.mean(gW, axis=(0, 2, 3), keepdims=True)
        gb = xp.mean(gb, axis=(0, 2, 3), keepdims=True)
        return gx, gW, gb


def affine_channel_2d_naive(x, W, b):
    W = functions.broadcast_to(W, x.shape)
    b = functions.broadcast_to(b, x.shape)
    return W * x + b


def affine_channel_2d(x, W, b):
    return AffineChannel2DFunction()(x, W, b)

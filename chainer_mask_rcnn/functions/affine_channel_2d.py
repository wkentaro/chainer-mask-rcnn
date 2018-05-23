import chainer
from chainer import cuda
from chainer import functions
from chainer.utils import type_check
import numpy


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

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x_type = in_types[0]
        w_type = in_types[1]
        b_type = in_types[2]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            b_type.ndim == 4,
            w_type.shape[1] == b_type.shape[1],
        )

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
                'gx = W * gy; gW = x * gy;', 'affine_bwd'
            )(x, W, b, gy)
        gb = gy

        gW = xp.sum(gW, axis=(0, 2, 3), keepdims=True)
        gb = xp.sum(gb, axis=(0, 2, 3), keepdims=True)
        return gx, gW, gb


def affine_channel_2d_naive(x, W, b):
    W = functions.broadcast_to(W, x.shape)
    b = functions.broadcast_to(b, x.shape)
    return W * x + b


def affine_channel_2d(x, W, b):
    return AffineChannel2DFunction()(x, W, b)

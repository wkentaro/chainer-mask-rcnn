import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from chainer_mask_rcnn import functions


class TestAffineChannel2D(unittest.TestCase):

    def setUp(self):
        N = 3
        n_channels = 3
        self.x = numpy.arange(
            N * n_channels * 12 * 8,
            dtype=numpy.float32).reshape((N, n_channels, 12, 8))
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1
        self.W = numpy.random.random(
            (1, n_channels, 1, 1)).astype(numpy.float32)
        self.b = numpy.random.random(
            (1, n_channels, 1, 1)).astype(numpy.float32)
        self.gy = numpy.random.uniform(
            -1, 1, self.x.shape).astype(numpy.float32)
        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}

    def check_forward(self, x_data, W_data, b_data):
        x = chainer.Variable(x_data)
        W = chainer.Variable(W_data)
        b = chainer.Variable(b_data)
        y = functions.affine_channel_2d(x, W, b)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.W, self.b)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.W), cuda.to_gpu(self.b))

    @attr.gpu
    @condition.retry(3)
    def test_forward_cpu_gpu_equal(self):
        # cpu
        x_cpu = chainer.Variable(self.x)
        W = chainer.Variable(self.W)
        b = chainer.Variable(self.b)
        y_cpu = functions.affine_channel_2d(x_cpu, W, b)

        # gpu
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W = chainer.Variable(cuda.to_gpu(self.W))
        b = chainer.Variable(cuda.to_gpu(self.b))
        y_gpu = functions.affine_channel_2d(x_gpu, W, b)
        testing.assert_allclose(y_cpu.data, cuda.to_cpu(y_gpu.data))

    def check_backward(self, x_data, W_data, b_data, y_grad):
        gradient_check.check_backward(
            functions.AffineChannel2DFunction(),
            (x_data, W_data, b_data), y_grad,
            **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)

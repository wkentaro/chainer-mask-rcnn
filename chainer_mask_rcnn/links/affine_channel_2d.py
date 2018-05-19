import chainer
from chainer import functions
from chainer import initializers

from ..functions import affine_channel_2d


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

import chainer
from chainer import functions
from chainer import initializers


class AffineChannel2D(chainer.Link):

    """A simple channel-wise affine transformation operation"""

    def __init__(self, channels):
        super(AffineChannel2D, self).__init__()
        with self.init_scope():
            W_initializer = initializers.One()
            self.W = chainer.variable.Parameter(W_initializer, (channels,))
            b_initializer = initializers.Zero()
            self.b = chainer.variable.Parameter(b_initializer, (channels,))

    def __call__(self, x):
        W = functions.reshape(self.W, (1, -1, 1, 1))
        W = functions.broadcast_to(W, x.shape)
        y = x * W
        del W
        b = functions.reshape(self.b, (1, -1, 1, 1))
        b = functions.broadcast_to(b, x.shape)
        y += b
        del b
        return y

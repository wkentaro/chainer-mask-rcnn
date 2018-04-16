import chainer
from chainer import functions


class AffineChannel2D(chainer.Link):

    """A simple channel-wise affine transformation operation"""

    def __init__(self, channels):
        super(AffineChannel2D, self).__init__()
        with self.init_scope():
            self.W = chainer.variable.Parameter(shape=(channels,))
            self.b = chainer.variable.Parameter(shape=(channels,))

    def __call__(self, x):
        W = functions.reshape(self.W, (1, -1, 1, 1))
        W = functions.broadcast_to(W, x.shape)
        b = functions.reshape(self.b, (1, -1, 1, 1))
        b = functions.broadcast_to(b, x.shape)
        return x * W + b

from chainer.dataset import convert
from chainer.training.updater import StandardUpdater


class GradsAccumulateUpdater(StandardUpdater):

    def __init__(self, n_sample, iterator, optimizer,
                 converter=convert.concat_examples,
                 device=None, loss_func=None):
        super(GradsAccumulateUpdater, self).__init__(
            iterator=iterator, optimizer=optimizer,
            converter=converter, device=device, loss_func=loss_func)
        self.n_sample = n_sample

    def update(self):
        self.update_core()
        self.iteration += self.n_sample

    def update_core(self):
        optimizer = self._optimizers['main']

        model = optimizer.target
        model.cleargrads()

        for i in range(self.n_sample):
            batch = self._iterators['main'].next()
            in_arrays = self.converter(batch, self.device)

            loss_func = self.loss_func or model

            if isinstance(in_arrays, tuple):
                loss = loss_func(*in_arrays)
            elif isinstance(in_arrays, dict):
                loss = loss_func(**in_arrays)
            else:
                loss = loss_func(in_arrays)

            # accumulate grads
            loss.backward()
            loss.unchain_backward()
            del loss

        optimizer.update()

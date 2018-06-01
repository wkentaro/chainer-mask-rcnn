from chainer.dataset.convert import _concat_arrays
from chainer.dataset.convert import to_device
import six


def concat_examples(batch, device=None, padding=None,
                    indices_concat=None, indices_to_device=None):
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    elem_size = len(first_elem)
    if indices_concat is None:
        indices_concat = range(elem_size)
    if indices_to_device is None:
        indices_to_device = range(elem_size)

    result = []
    if not isinstance(padding, tuple):
        padding = [padding] * elem_size

    for i in six.moves.range(elem_size):
        res = [example[i] for example in batch]
        if i in indices_concat:
            res = _concat_arrays(res, padding[i])
        if i in indices_to_device:
            if i in indices_concat:
                res = to_device(device, res)
            else:
                res = [to_device(device, r) for r in res]
        result.append(res)

    return tuple(result)

from chainer.dataset.convert import _concat_arrays
from chainer.dataset.convert import to_device
import six


def concat_examples(batch, device=None, padding=None):
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    result = []
    if not isinstance(padding, tuple):
        padding = [padding] * len(first_elem)

    for i in six.moves.range(len(first_elem)):
        res = _concat_arrays([example[i] for example in batch], padding[i])
        if i in [0, 1]:  # img, bbox
            res = to_device(device, res)
        result.append(res)

    return tuple(result)

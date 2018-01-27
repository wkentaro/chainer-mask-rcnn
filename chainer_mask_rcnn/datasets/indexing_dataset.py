import chainer


class IndexingDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset, indices=0):
        self._dataset = dataset

        if isinstance(indices, int):
            indices = [indices]
        self._indices = indices
        self._size = len(indices)

    def __getattr__(self, key):
        return self._dataset.__getattr__(key)

    def __len__(self):
        return self._size

    def get_example(self, i):
        index = self._indices[i]
        return self._dataset.get_example(index)

import warnings

import chainer
import numpy as np

from ..utils import label2instance_boxes


class MaskRcnnDataset(chainer.dataset.DatasetMixin):

    def __init__(self, instance_dataset):
        warnings.warn('MaskRcnnDataset is deprecated, please stop using it.')
        self._instance_dataset = instance_dataset
        self.fg_class_names = instance_dataset.class_names[1:]  # remove bg
        self.n_fg_class = len(self.fg_class_names)

    def __len__(self):
        return len(self._instance_dataset)

    def get_example(self, i):
        img, lbl_cls, lbl_ins = self._instance_dataset.get_example(i)
        labels, bboxes, masks = label2instance_boxes(
            lbl_ins, lbl_cls, return_masks=True)
        masks = masks.astype(np.int32, copy=False)
        labels = labels.astype(np.int32, copy=False)
        labels -= 1  # background: 0 -> -1
        bboxes = bboxes.astype(np.float32, copy=False)
        return img, bboxes, labels, masks

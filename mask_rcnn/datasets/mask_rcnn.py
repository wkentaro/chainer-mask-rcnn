import chainer
import numpy as np

from ..utils import label2instance_boxes


class MaskRcnnDataset(chainer.dataset.DatasetMixin):

    def __init__(self, instance_dataset, return_masks=False):
        self._instance_dataset = instance_dataset
        self._return_masks = return_masks

    def __len__(self):
        return len(self._instance_dataset)

    def get_example(self, i):
        img, lbl_cls, lbl_ins = self._instance_dataset.get_example(i)
        labels, bboxes, masks = label2instance_boxes(
            lbl_ins, lbl_cls, return_masks=True)
        labels = labels.astype(np.int32, copy=False)
        labels -= 1  # background: 0 -> -1
        bboxes = bboxes.astype(np.float32, copy=False)
        if self._return_masks:
            masks = masks.astype(np.int32, copy=False)
        else:
            masks = lbl_ins
        return img, bboxes, labels, masks

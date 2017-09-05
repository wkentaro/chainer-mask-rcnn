import chainer

from ..utils import label2instance_boxes


class MaskRcnnDataset(chainer.dataset.DatasetMixin):

    def __init__(self, instance_dataset):
        self._instance_dataset = instance_dataset

    def __len__(self):
        return len(self._instance_dataset)

    def get_example(self, i):
        img, lbl_cls, lbl_ins = self._instance_dataset.get_example(i)
        labels, bboxes, masks = label2instance_boxes(
            lbl_ins, lbl_cls, return_masks=True)
        scale = 1.
        return img, bboxes, labels, masks, scale

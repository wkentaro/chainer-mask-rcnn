import numpy as np

from mask_rcnn.datasets.voc import VOCInstanceSegBase
from mask_rcnn.datasets.voc import VOC2012InstanceSeg


def test_VOCInstanceSegBase():
    assert isinstance(VOCInstanceSegBase.class_names, np.ndarray)
    assert not VOCInstanceSegBase.class_names.flags.writeable


def test_VOC2012InstanceSeg():
    dataset_train = VOC2012InstanceSeg('train')
    img, lbl_cls, lbl_ins = dataset_train.get_example(0)
    assert img.shape[:2] == lbl_cls.shape == lbl_ins.shape
    assert img.shape[2] == 3

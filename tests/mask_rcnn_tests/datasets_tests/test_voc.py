import numpy as np

from mask_rcnn.datasets.voc import VOCInstanceSegBase


def test_VOCInstanceSegBase():
    assert isinstance(VOCInstanceSegBase.class_names, np.ndarray)
    assert not VOCInstanceSegBase.class_names.flags.writeable

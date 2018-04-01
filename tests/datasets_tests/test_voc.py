import numpy as np

from chainer_mask_rcnn.datasets.voc.voc \
    import VOCInstanceSegmentationDatasetBase
from chainer_mask_rcnn.datasets.voc import VOC2012InstanceSegmentationDataset


def test_VOCInstanceSegmentationDatasetBase():
    assert isinstance(VOCInstanceSegmentationDatasetBase.class_names,
                      np.ndarray)
    assert not VOCInstanceSegmentationDatasetBase.class_names.flags.writeable


def test_VOC2012InstanceSegmentationDataset():
    dataset_train = VOC2012InstanceSegmentationDataset('train')
    img, bboxes, labels, masks = dataset_train.get_example(0)

    assert img.ndim == 3
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert masks.ndim == 3

    img_shape = img.shape[:2]
    assert img_shape == masks.shape[1:3]

    n_bbox = len(bboxes)
    assert n_bbox == len(labels) == len(masks)
    assert bboxes.shape[1] == 4

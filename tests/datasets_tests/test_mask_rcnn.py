from chainer_mask_rcnn.datasets.mask_rcnn import MaskRcnnDataset
from chainer_mask_rcnn.datasets.voc import VOC2012InstanceSeg


def test_MaskRcnnDataset():
    dataset_ins = VOC2012InstanceSeg(split='train')
    dataset = MaskRcnnDataset(dataset_ins)
    assert len(dataset_ins) == len(dataset)

    img, bboxes, labels, masks = dataset.get_example(0)

    H, W, C = img.shape
    assert C == 3

    N = len(bboxes)
    assert bboxes.shape == (N, 4)
    assert labels.shape == (N,)
    assert masks.shape == (N, H, W)

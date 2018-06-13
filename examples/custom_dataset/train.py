#!/usr/bin/env python

import os
import os.path as osp
import sys

import chainer
import numpy as np
import PIL.Image

import chainer_mask_rcnn as cmr

here = osp.dirname(osp.abspath(__file__))  # NOQA
sys.path.insert(0, osp.join(here, '..'))  # NOQA

import train_common


class VOCLikeDataset(chainer.dataset.DatasetMixin):

    class_names = cmr.datasets.VOC2012InstanceSeg.class_names

    def __init__(self, root_dir):
        self._root_dir = root_dir

        self._ids = []
        img_dir = osp.join(self._root_dir, 'JPEGImages')
        for img_file in os.listdir(img_dir):
            id_ = osp.splitext(img_file)[0]
            self._ids.append(id_)

    def __len__(self):
        return len(self._ids)

    def get_example(self, i):
        id_ = self._ids[i]

        img_file = osp.join(self._root_dir, 'JPEGImages', id_ + '.jpg')
        img = np.array(PIL.Image.open(img_file))
        cls_file = osp.join(self._root_dir, 'SegmentationClass', id_ + '.npy')
        cls = np.load(cls_file)
        ins_file = osp.join(self._root_dir, 'SegmentationObject', id_ + '.npy')
        ins = np.load(ins_file)
        ins[ins == 0] = -1  # instance id 0 should be ignored.

        assert img.shape[:2] == cls.shape == ins.shape

        # ins, cls -> bboxes, labels, masks
        labels, bboxes, masks = cmr.utils.label2instance_boxes(
            label_instance=ins, label_class=cls, return_masks=True,
        )
        masks = masks.astype(np.int32, copy=False)
        labels = labels.astype(np.int32, copy=False)
        labels -= 1  # background: 0 -> -1
        bboxes = bboxes.astype(np.float32, copy=False)

        return img, bboxes, labels, masks


def main():
    args = train_common.parse_args()

    args.logs_dir = osp.join(here, 'logs')

    # Dataset. For demonstration with few images, we use same dataset
    # for both train and test.
    root_dir = osp.join(
        here, 'src/labelme/examples/instance_segmentation/data_dataset_voc',
    )
    args.dataset = 'custom'
    # 1 epoch = 3 images -> 60 images
    train_data = [VOCLikeDataset(root_dir=root_dir)] * 20
    train_data = chainer.datasets.ConcatenatedDataset(*train_data)
    test_data = VOCLikeDataset(root_dir=root_dir)
    args.class_names = tuple(VOCLikeDataset.class_names.tolist())

    # Model.
    args.min_size = 600
    args.max_size = 1000
    args.anchor_scales = (4, 8, 16, 32)

    train_common.train(
        args=args,
        train_data=train_data,
        test_data=test_data,
        evaluator_type='voc',
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import chainer_mask_rcnn as mrcnn


def visualize(dataset, index):
    fg_class_names = dataset.class_names
    n_fg_class = len(fg_class_names)

    img, bboxes, labels, masks = dataset[index]
    masks = masks.astype(bool)
    captions = fg_class_names[labels]
    return mrcnn.utils.draw_instance_bboxes(
        img, bboxes, labels + 1, n_fg_class + 1,
        masks=masks, captions=captions)


def main():
    dataset = mrcnn.datasets.VOC2012InstanceSegmentationDataset('train')
    dataset.split = 'train'
    mrcnn.datasets.view_dataset(dataset, visualize)


if __name__ == '__main__':
    main()

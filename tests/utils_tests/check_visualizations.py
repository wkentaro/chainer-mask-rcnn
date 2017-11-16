#!/usr/bin/env python

import mask_rcnn as mrcnn
import mvtk


def visualize(dataset, index):
    class_names = dataset._instance_dataset.class_names
    n_class = len(class_names)

    img, bboxes, labels, masks = dataset[index]
    labels += 1
    masks = masks.astype(bool)
    captions = class_names[labels]
    return mrcnn.utils.draw_instance_boxes(
        img, bboxes, labels, n_class, masks=masks, captions=captions)


def main():
    dataset = mrcnn.datasets.MaskRcnnDataset(
        mrcnn.datasets.VOC2012InstanceSeg('train'))
    dataset.split = 'train'
    mvtk.datasets.view_dataset(dataset, visualize)


if __name__ == '__main__':
    main()

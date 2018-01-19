from __future__ import print_function

import matplotlib.pyplot as plt

import chainer_mask_rcnn as mask_rcnn


def main():
    dataset = mask_rcnn.datasets.VOC2012InstanceSeg('train')
    class_names = dataset.class_names

    img, lbl_cls, lbl_ins = dataset[0]

    print('img:', img.shape)
    print('lbl_cls:', lbl_cls.shape)
    print('lbl_ins:', lbl_ins.shape)

    viz1 = mask_rcnn.utils.visualize_instance_segmentation(
        lbl_ins, lbl_cls, img, class_names)
    plt.subplot(121)
    plt.imshow(viz1)

    labels, bboxes, masks = mask_rcnn.utils.label2instance_boxes(
        lbl_ins, lbl_cls, return_masks=True)

    print('labels:', labels.shape)
    print('bboxes:', bboxes.shape)
    print('masks:', masks.shape)

    lbl_ins2, lbl_cls2 = mask_rcnn.utils.instance_boxes2label(
        labels, bboxes, masks)
    viz2 = mask_rcnn.utils.visualize_instance_segmentation(
        lbl_ins2, lbl_cls2, img, class_names)

    plt.subplot(122)
    plt.imshow(viz2)

    plt.show()


if __name__ == '__main__':
    main()

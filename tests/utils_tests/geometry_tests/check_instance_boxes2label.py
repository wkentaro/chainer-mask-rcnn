from __future__ import print_function

import matplotlib.pyplot as plt

import chainer_mask_rcnn as mask_rcnn


def main():
    dataset = mask_rcnn.datasets.VOC2012InstanceSeg('train')
    fg_class_names = dataset.class_names
    n_fg_class = len(fg_class_names)

    img, bboxes, labels, masks = dataset[0]

    print('img:', img.shape)
    print('bboxes:', bboxes.shape)
    print('labels:', labels.shape)
    print('masks:', masks.shape)

    # viz1
    captions = fg_class_names[labels]
    viz1 = mask_rcnn.utils.draw_instance_bboxes(
        img, bboxes, labels, n_class=n_fg_class + 1,
        captions=captions, masks=masks.astype(bool))
    plt.subplot(121)
    plt.imshow(viz1)

    # viz2
    lbl_ins, lbl_cls = mask_rcnn.utils.instance_boxes2label(
        labels + 1, bboxes, masks.astype(bool))
    labels, bboxes, masks = mask_rcnn.utils.label2instance_boxes(
        lbl_ins, lbl_cls, return_masks=True)
    labels -= 1
    captions = fg_class_names[labels]
    viz2 = mask_rcnn.utils.draw_instance_bboxes(
        img, bboxes, labels, n_class=n_fg_class + 1,
        captions=captions, masks=masks)

    plt.subplot(122)
    plt.imshow(viz2)

    plt.show()


if __name__ == '__main__':
    main()

import cv2
import mvtk
import numpy as np

import mask_rcnn


def visualize_func(dataset, index):
    vizs = []
    img, lbl_cls, lbl_ins = dataset[index]
    viz = mask_rcnn.utils.visualize_instance_segmentation(
        lbl_ins, lbl_cls, img, dataset.class_names)
    vizs.append(viz)

    labels, boxes, masks = mask_rcnn.utils.label2instance_boxes(
        lbl_ins, lbl_cls, return_masks=True)

    H, W = img.shape[:2]
    rois = mask_rcnn.utils.augment_bboxes(boxes, H, W)

    sample_rois, gt_roi_locs, gt_roi_labels, gt_roi_masks = \
        mask_rcnn.utils.create_proposal_targets(rois, boxes, labels, masks)

    viz = mask_rcnn.utils.draw_instance_boxes(
        img, sample_rois, gt_roi_labels, n_class=21, bg_class=-1)
    vizs.append(viz)

    viz1 = mvtk.image.tile(vizs)

    vizs = []
    for roi, id_cls, mask_ins in zip(sample_rois, gt_roi_labels, gt_roi_masks):
        if id_cls == 0:
            continue
        viz = img.copy()
        viz[~mask_ins] = 255
        viz = mask_rcnn.utils.draw_instance_boxes(
            viz, [roi], [id_cls], n_class=21, bg_class=-1, thickness=2)
        vizs.append(viz)
    viz2 = mvtk.image.tile(vizs)
    scale = 1. * viz1.shape[1] / viz2.shape[1]
    viz2 = cv2.resize(viz2, None, None, fx=scale, fy=scale)

    return np.vstack([viz1, viz2])


def main():
    dataset = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
    dataset.split = 'train'
    mvtk.datasets.view_dataset(dataset, visualize_func)


if __name__ == '__main__':
    main()

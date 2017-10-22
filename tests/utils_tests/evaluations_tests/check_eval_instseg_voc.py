from __future__ import print_function

import cv2
import numpy as np

import mask_rcnn
from mask_rcnn.utils.evaluations import eval_instseg_voc


def check_eval_instseg_voc():
    dataset = mask_rcnn.datasets.VOC2012InstanceSeg('train')
    img, lbl_cls_true, lbl_ins_true = dataset.get_example(0)

    gt_label, gt_bbox, gt_mask = mask_rcnn.utils.label2instance_boxes(
        lbl_ins_true, lbl_cls_true, return_masks=True)

    pred_mask = []
    for m in gt_mask:
        H, W = m.shape[:2]
        m = m.astype(np.float32)
        m = cv2.resize(m, None, None, fx=0.09, fy=0.09)
        m = cv2.resize(m, (W, H))
        m = m >= 0.5
        pred_mask.append(m)
    pred_mask = np.asarray(pred_mask)

    pred_label = gt_label
    pred_score = np.random.uniform(0.6, 0.99, (len(pred_mask),))

    lbl_ins_pred, lbl_cls_pred = mask_rcnn.utils.instance_boxes2label(
        gt_label, gt_bbox, pred_mask)

    viz_true = mask_rcnn.utils.visualize_instance_segmentation(
        lbl_ins_true, lbl_cls_true, img, dataset.class_names)
    viz_pred = mask_rcnn.utils.visualize_instance_segmentation(
        lbl_ins_pred, lbl_cls_pred, img, dataset.class_names)
    viz = np.vstack([viz_true, viz_pred])

    gt_label -= 1  # background: 0 -> -1
    gt_masks = [gt_mask]
    gt_labels = [gt_label]

    pred_label = gt_label
    pred_masks = [pred_mask]
    pred_labels = [pred_label]
    pred_scores = [pred_score]

    ap = eval_instseg_voc(
        pred_masks, pred_labels, pred_scores, gt_masks, gt_labels,
        use_07_metric=True)
    ap = ap['ap']
    for cls_id, cls_name in enumerate(dataset.class_names[1:]):
        if cls_id >= len(ap):
            cls_ap = None
        else:
            cls_ap = ap[cls_id]
        print('ap/%s:' % cls_name, cls_ap)
    print('map:', np.nanmean(ap))

    cv2.imshow(__file__, viz[:, :, ::-1])
    cv2.waitKey(0)


check_eval_instseg_voc()

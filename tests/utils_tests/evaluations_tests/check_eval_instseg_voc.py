from __future__ import print_function

import cv2
import numpy as np

import chainer_mask_rcnn as mask_rcnn
from chainer_mask_rcnn.utils.evaluations import eval_instseg_voc


def check_eval_instseg_voc():
    dataset = mask_rcnn.datasets.VOC2012InstanceSeg('train')
    img, gt_bbox, gt_label, gt_mask = dataset.get_example(0)

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

    fg_class_names = dataset.class_names
    n_fg_class = len(fg_class_names)
    captions = [fg_class_names[l] for l in gt_label]
    viz_true = mask_rcnn.utils.draw_instance_bboxes(
        img, gt_bbox, gt_label + 1, n_class=n_fg_class + 1,
        captions=captions, masks=gt_mask.astype(bool))
    viz_pred = mask_rcnn.utils.draw_instance_bboxes(
        img, gt_bbox, pred_label + 1, n_class=n_fg_class + 1,
        captions=captions, masks=pred_mask.astype(bool))
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

from chainercv.links.model.faster_rcnn.utils.proposal_target_creator\
    import ProposalTargetCreator
import numpy as np


def get_bbox_overlap(bbox1, bbox2):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2
    w1, h1 = x12 - x11, y12 - y11
    w2, h2 = x22 - x21, y22 - y21
    intersect = (max(0, min(x12, x22) - max(x11, x21)) *
                 max(0, min(y12, y22) - max(y11, y21)))
    union = w1 * h1 + w2 * h2 - intersect
    return 1.0 * intersect / union


def get_mask_overlap(mask1, mask2):
    intersect = np.bitwise_and(mask1, mask2).sum()
    union = np.bitwise_or(mask1, mask2).sum()
    return 1.0 * intersect / union


def bboxes_xyyx(bboxes):
    bboxes[:, :2] = bboxes[:, :2][:, ::-1]
    bboxes[:, 2:] = bboxes[:, 2:][:, ::-1]
    return bboxes


def validate_bboxes(bboxes, H, W):
    bboxes = np.asarray(bboxes)
    bboxes[:, 0][bboxes[:, 0] < 0] = 0
    bboxes[:, 0][bboxes[:, 0] >= W] = W - 1
    bboxes[:, 1][bboxes[:, 1] < 0] = 0
    bboxes[:, 1][bboxes[:, 1] >= H] = H - 1
    bboxes[:, 2][bboxes[:, 2] < 0] = 0
    bboxes[:, 2][bboxes[:, 2] >= W] = W - 1
    bboxes[:, 3][bboxes[:, 3] < 0] = 0
    bboxes[:, 3][bboxes[:, 3] >= H] = H - 1
    keep = bboxes[:, 0] < bboxes[:, 2]
    keep = keep & (bboxes[:, 1] < bboxes[:, 3])
    bboxes = bboxes[keep]
    return bboxes


def augment_bboxes(bboxes, H, W):
    bboxes_aug = []
    for _ in xrange(100):
        for box in bboxes:
            roi = []
            for xy in box:
                scale = np.random.normal(1.0, scale=0.2)
                xy = int(scale * xy)
                roi.append(xy)
            bboxes_aug.append(roi)
    bboxes_aug = validate_bboxes(bboxes_aug, H, W)
    return bboxes_aug


def create_proposal_targets(rois, boxes, labels, masks):
    # xy -> yx
    boxes = bboxes_xyyx(boxes).astype(np.float64)
    rois = bboxes_xyyx(rois).astype(np.float64)
    labels -= 1
    proposal_target_creator = ProposalTargetCreator()
    sample_rois, gt_roi_locs, gt_roi_labels = \
        proposal_target_creator(rois, boxes, labels)
    # yx -> xy
    boxes = bboxes_xyyx(boxes).astype(np.int32)
    sample_rois = bboxes_xyyx(sample_rois).astype(np.int32)

    gt_roi_masks = []
    for id_cls, roi in zip(gt_roi_labels, sample_rois):
        if id_cls == 0:
            gt_roi_masks.append(None)
            continue
        idx_ins = np.argmax([get_bbox_overlap(b, roi) for b in boxes])
        mask_ins = masks[idx_ins]
        x1, y1, x2, y2 = roi
        mask_roi = np.zeros_like(mask_ins, dtype=bool)
        mask_roi[y1:y2, x1:x2] = True
        mask_ins = mask_ins & mask_roi
        gt_roi_masks.append(mask_ins)
    return sample_rois, gt_roi_locs, gt_roi_labels, gt_roi_masks

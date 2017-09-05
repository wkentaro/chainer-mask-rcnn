import collections

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
    boxes = boxes[:, [1, 0, 3, 2]].astype(np.float64)
    rois = rois[:, [1, 0, 3, 2]].astype(np.float64)
    labels -= 1
    proposal_target_creator = ProposalTargetCreator()
    sample_rois, gt_roi_locs, gt_roi_labels = \
        proposal_target_creator(rois, boxes, labels)
    # yx -> xy
    boxes = boxes[:, [1, 0, 3, 2]].astype(np.int64)
    sample_rois = sample_rois[:, [1, 0, 3, 2]].astype(np.int64)

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


def label2instance_boxes(label_instance, label_class,
                         ignore_instance=-1, ignore_class=(-1, 0),
                         return_masks=False):
    """Convert instance label to boxes.

    Parameters
    ----------
    label_instance: numpy.ndarray, (H, W)
        Label image for instance id.
    label_class: numpy.ndarray, (H, W)
        Label image for class.
    ignore_instance: int or tuple of int
        Label value ignored about label_instance. (default: -1)
    ignore_class: int or tuple of int
        Label value ignored about label_class. (default: (-1, 0))
    return_masks: bool
        Flag to return each instance mask.

    Returns
    -------
    instance_classes: numpy.ndarray, (n_instance,)
        Class id for each instance.
    boxes: (n_instance, 4)
        Bounding boxes for each instance. (x1, y1, x2, y2)
    instance_masks: numpy.ndarray, (n_instance, H, W), bool
        Masks for each instance. Only returns when return_masks=True.
    """
    if not isinstance(ignore_instance, collections.Iterable):
        ignore_instance = (ignore_instance,)
    if not isinstance(ignore_class, collections.Iterable):
        ignore_class = (ignore_class,)
    # instance_class is 'Class of the Instance'
    instance_classes = []
    boxes = []
    instance_masks = []
    instances = np.unique(label_instance)
    for inst in instances:
        if inst in ignore_instance:
            continue

        mask_inst = label_instance == inst
        count = collections.Counter(label_class[mask_inst].tolist())
        instance_class = max(count.items(), key=lambda x: x[1])[0]

        if instance_class in ignore_class:
            continue

        where = np.argwhere(mask_inst)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1

        instance_classes.append(instance_class)
        boxes.append((x1, y1, x2, y2))
        instance_masks.append(mask_inst)
    instance_classes = np.array(instance_classes)
    boxes = np.array(boxes)
    instance_masks = np.array(instance_masks)
    if return_masks:
        return instance_classes, boxes, instance_masks
    else:
        return instance_classes, boxes


def mask_to_bbox(mask):
    """Convert mask image to bounding box.

    Parameters
    ----------
    mask: :class:`numpy.ndarray`
        Input mask image.

    Returns
    -------
    box: tuple (x1, y1, x2, y2)
        Bounding box.
    """
    where = np.argwhere(mask)
    (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
    bbox = x1, y1, x2, y2
    return bbox


def label_to_bboxes(label, ignore_label=-1):
    """Convert label image to bounding boxes."""
    if not isinstance(ignore_label, collections.Iterable):
        ignore_label = (ignore_label,)
    bboxes = []
    for l in np.unique(label):
        if l in ignore_label:
            continue
        mask = label == l
        bbox = mask_to_bbox(mask)
        bboxes.append(bbox)
    return np.array(bboxes)


def label_rois(rois, label_instance, label_class, overlap_thresh=0.5):
    """Label rois for instance classes.

    Parameters
    ----------
    rois: numpy.ndarray, (n_rois, 4)
    label_instance: numpy.ndarray, (H, W)
    label_class: numpy.ndarray, (H, W)
    overlap_thresh: float, [0, 1]
        Threshold to label as fg. (default: 0.5)

    Returns
    -------
    roi_clss: numpy.ndarray, (n_rois,), numpy.int32
    roi_inst_masks: list of numpy.ndarray
    """
    inst_clss, inst_rois, inst_masks = label2instance_boxes(
        label_instance, label_class, return_masks=True)
    roi_clss = []
    roi_inst_masks = []
    for roi in rois:
        overlaps = [get_bbox_overlap(roi, inst_roi) for inst_roi in inst_rois]
        inst_ind = np.argmax(overlaps)
        overlap = overlaps[inst_ind]

        if overlap > overlap_thresh:
            roi_cls = inst_clss[inst_ind]
            x1, y1, x2, y2 = roi
            roi_inst_mask = inst_masks[inst_ind][y1:y2, x1:x2]
        else:
            roi_cls = 0
            roi_inst_mask = None
        roi_clss.append(roi_cls)
        roi_inst_masks.append(roi_inst_mask)
    roi_clss = np.array(roi_clss, dtype=np.int32)
    return roi_clss, roi_inst_masks


def instance_label_accuracy_score(lbl_ins1, lbl_ins2):
    best_overlaps = []
    for l1 in np.unique(lbl_ins1):
        if l1 == -1:
            continue
        mask1 = lbl_ins1 == l1
        best_overlap = 0
        for l2 in np.unique(lbl_ins2):
            if l2 == -1:
                continue
            mask2 = lbl_ins2 == l2
            overlap = get_mask_overlap(mask1, mask2)
            best_overlap = max(best_overlap, overlap)
        best_overlaps.append(best_overlap)
    return np.mean(best_overlaps)

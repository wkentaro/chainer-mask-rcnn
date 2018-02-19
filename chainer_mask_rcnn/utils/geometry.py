import collections

# import cv2
import numpy as np


def get_bbox_overlap(bbox1, bbox2):
    y11, x11, y12, x12 = bbox1
    y21, x21, y22, x22 = bbox2
    w1, h1 = x12 - x11, y12 - y11
    w2, h2 = x22 - x21, y22 - y21
    intersect = (max(0, min(x12, x22) - max(x11, x21)) *
                 max(0, min(y12, y22) - max(y11, y21)))
    union = w1 * h1 + w2 * h2 - intersect
    return 1.0 * intersect / union


def get_mask_overlap(mask1, mask2, half_if_nounion=False):
    intersect = np.bitwise_and(mask1, mask2).sum()
    union = np.bitwise_or(mask1, mask2).sum()
    if union == 0:
        if half_if_nounion:
            return 0.5
        else:
            return 0.
    else:
        return 1.0 * intersect / union


# def create_proposal_targets(rois, boxes, labels, masks,
#                             loc_normalize_mean, loc_normalize_std,
#                             mask_size):
#     import chainer
#     from chainercv.links.model.faster_rcnn.utils.proposal_target_creator\
#         import ProposalTargetCreator
#     xp = chainer.cuda.get_array_module(rois)
#     rois = chainer.cuda.to_cpu(rois)
#     boxes = chainer.cuda.to_cpu(boxes)
#     labels = chainer.cuda.to_cpu(labels)
#     masks = chainer.cuda.to_cpu(masks)
#
#     proposal_target_creator = ProposalTargetCreator(n_sample=64)
#     sample_rois, gt_roi_locs, gt_roi_labels = \
#         proposal_target_creator(
#             rois, boxes, labels,
#             loc_normalize_mean, loc_normalize_std)
#
#     N, H, W = masks.shape
#     assert boxes.shape == (N, 4)
#     assert labels.shape == (N,)
#
#     n_sample = len(sample_rois)
#     gt_roi_masks = - np.ones(
#        (n_sample, mask_size, mask_size), dtype=np.int32)
#     for i, (id_cls, roi) in enumerate(zip(gt_roi_labels, sample_rois)):
#         y1, x1, y2, x2 = map(int, roi)
#         assert 0 <= y1 and y2 <= H
#         assert 0 <= x1 and x2 <= W
#         if id_cls == 0:
#             continue
#         idx_ins = np.argmax([get_bbox_overlap(b, roi) for b in boxes])
#         mask_ins = masks[idx_ins]
#         assert mask_ins.dtype == np.int32
#         mask_roi = np.zeros_like(mask_ins)
#         mask_roi[y1:y2, x1:x2] = 1
#         mask_ins = mask_ins & mask_roi
#
#         mask_ins = mask_ins[y1:y2, x1:x2]
#         mask_ins = mask_ins.astype(np.float32)
#         mask_ins = cv2.resize(mask_ins, (mask_size, mask_size))
#         mask_ins = np.round(mask_ins).astype(np.int32)
#         gt_roi_masks[i] = mask_ins
#     if xp != np:
#         sample_rois = chainer.cuda.to_gpu(sample_rois)
#         gt_roi_locs = chainer.cuda.to_gpu(gt_roi_locs)
#         gt_roi_labels = chainer.cuda.to_gpu(gt_roi_labels)
#         gt_roi_masks = chainer.cuda.to_gpu(gt_roi_masks)
#     return sample_rois, gt_roi_locs, gt_roi_labels, gt_roi_masks


def label2instance_boxes(label_instance, label_class, return_masks=False):
    """Convert instance label to boxes.

    Parameters
    ----------
    label_instance: numpy.ndarray, (H, W)
        Label image for instance id.
    label_class: numpy.ndarray, (H, W)
        Label image for class.
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
    instances = np.unique(label_instance)
    instances = instances[instances != -1]
    n_instance = len(instances)
    # instance_class is 'Class of the Instance'
    instance_classes = np.zeros((n_instance,), dtype=np.int32)
    boxes = np.zeros((n_instance, 4), dtype=np.int32)
    H, W = label_instance.shape
    instance_masks = np.zeros((n_instance, H, W), dtype=bool)
    for i, inst in enumerate(instances):
        mask_inst = label_instance == inst
        count = collections.Counter(label_class[mask_inst].tolist())
        instance_class = max(count.items(), key=lambda x: x[1])[0]

        assert inst not in [-1]
        assert instance_class not in [-1, 0]

        where = np.argwhere(mask_inst)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1

        instance_classes[i] = instance_class
        boxes[i] = (y1, x1, y2, x2)
        instance_masks[i] = mask_inst
    if return_masks:
        return instance_classes, boxes, instance_masks
    else:
        return instance_classes, boxes


def instance_boxes2label(labels, bboxes, masks, scores=None):
    if scores is not None:
        # sort ascending order of score
        indices = np.argsort(scores)
        labels = labels[indices]
        bboxes = bboxes[indices]
        masks = masks[indices]

    _, H, W = masks.shape
    lbl_ins = - np.ones((H, W), dtype=np.int32)
    lbl_cls = np.zeros((H, W), dtype=np.int32)
    for ins_id, (label, bbox, mask) in enumerate(zip(labels, bboxes, masks)):
        assert label > 0  # instance must be foreground
        assert mask.dtype == bool
        lbl_cls[mask] = label
        lbl_ins[mask] = ins_id

    return lbl_ins, lbl_cls


def mask_to_bbox(mask):
    """Convert mask image to bounding box.

    Parameters
    ----------
    mask: :class:`numpy.ndarray`
        Input mask image.

    Returns
    -------
    box: tuple (y1, x1, y2, x2)
        Bounding box.
    """
    where = np.argwhere(mask)
    (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
    bbox = y1, x1, y2, x2
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
            y1, x1, y2, x2 = roi
            roi_inst_mask = inst_masks[inst_ind][y1:y2, x1:x2]
        else:
            roi_cls = 0
            roi_inst_mask = None
        roi_clss.append(roi_cls)
        roi_inst_masks.append(roi_inst_mask)
    roi_clss = np.array(roi_clss, dtype=np.int32)
    return roi_clss, roi_inst_masks


# def instance_label_accuracy_score(lbl_ins1, lbl_ins2):
#     best_overlaps = []
#     for l1 in np.unique(lbl_ins1):
#         if l1 == -1:
#             continue
#         mask1 = lbl_ins1 == l1
#         best_overlap = 0
#         for l2 in np.unique(lbl_ins2):
#             if l2 == -1:
#                 continue
#             mask2 = lbl_ins2 == l2
#             overlap = get_mask_overlap(mask1, mask2)
#             best_overlap = max(best_overlap, overlap)
#         best_overlaps.append(best_overlap)
#     return np.mean(best_overlaps)

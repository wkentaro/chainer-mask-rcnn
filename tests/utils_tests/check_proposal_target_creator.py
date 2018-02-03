import cv2
import fcn
import numpy as np

import chainer_mask_rcnn as mask_rcnn


def _validate_bboxes(bboxes, H, W):
    bboxes = np.asarray(bboxes)
    bboxes[:, 0][bboxes[:, 0] < 0] = 0
    bboxes[:, 0][bboxes[:, 0] >= H] = H - 1
    bboxes[:, 1][bboxes[:, 1] < 0] = 0
    bboxes[:, 1][bboxes[:, 1] >= W] = W - 1
    bboxes[:, 2][bboxes[:, 2] < 0] = 0
    bboxes[:, 2][bboxes[:, 2] >= H] = H - 1
    bboxes[:, 3][bboxes[:, 3] < 0] = 0
    bboxes[:, 3][bboxes[:, 3] >= W] = W - 1
    keep = bboxes[:, 0] < bboxes[:, 2]
    keep = keep & (bboxes[:, 1] < bboxes[:, 3])
    bboxes = bboxes[keep]
    return bboxes


def _augment_bboxes(bboxes, H, W):
    bboxes_aug = []
    for _ in range(100):
        for box in bboxes:
            y1, x1, y2, x2 = box
            h = y2 - y1
            w = x2 - x1
            cy = (y2 + y1) / 2.
            cx = (x2 + x1) / 2.
            cy = cy + np.random.normal(0, scale=0.05) * h
            cx = cx + np.random.normal(0, scale=0.05) * w
            y1 = int(cy - h / 2.)
            x1 = int(cx - w / 2.)
            y2 = int(cy + h / 2.)
            x2 = int(cx + w / 2.)
            bboxes_aug.append((y1, x1, y2, x2))
    bboxes_aug = _validate_bboxes(bboxes_aug, H, W)
    return bboxes_aug


def visualize_func(dataset, index):
    vizs = []
    img, lbl_cls, lbl_ins = dataset[index]
    viz = mask_rcnn.utils.visualize_instance_segmentation(
        lbl_ins, lbl_cls, img, dataset.class_names)
    vizs.append(viz)

    mrcnn_dataset = mask_rcnn.datasets.MaskRcnnDataset(dataset)
    img, boxes, labels, masks = mrcnn_dataset[index]
    H, W = img.shape[:2]

    if False:
        rois = _augment_bboxes(boxes, H, W).astype(np.float32)
    else:
        import selectivesearch
        img_lbl, regions = selectivesearch.selective_search(
            img, scale=500, sigma=0.9, min_size=10)
        rois = []
        for region in regions:
            x1, y1, w, h = region['rect']
            if (w * h) == 0:
                continue
            x2 = x1 + w
            y2 = y1 + h
            rois.append((y1, x1, y2, x2))
        rois = np.asarray(rois, dtype=np.float32)

    if False:
        from chainercv.links.model.faster_rcnn.utils.proposal_target_creator \
            import ProposalTargetCreator
        proposal_target_creator = ProposalTargetCreator()

        sample_rois, gt_roi_locs, gt_roi_labels = proposal_target_creator(
            rois, boxes, labels,
            loc_normalize_mean=(0., 0., 0., 0.),
            loc_normalize_std=(0.1, 0.1, 0.2, 0.2))
        gt_roi_masks = [None] * len(sample_rois)
    else:
        from chainer_mask_rcnn.utils.proposal_target_creator import\
            ProposalTargetCreator
        proposal_target_creator = ProposalTargetCreator()
        sample_rois, gt_roi_locs, gt_roi_labels, gt_roi_masks = \
            proposal_target_creator(
                rois, boxes, labels, masks,
                loc_normalize_mean=(0., 0., 0., 0.),
                loc_normalize_std=(0.1, 0.1, 0.2, 0.2))

    masks = []
    for roi, mask in zip(sample_rois, gt_roi_masks):
        y1, x1, y2, x2 = roi
        mask = mask.astype(float)
        mask[mask < 0] = 0
        mask = cv2.resize(mask, (int(x2 - x1), int(y2 - y1)))
        mask = np.round(mask).astype(bool)
        masks.append(mask)

    captions = dataset.class_names[gt_roi_labels]
    masks = np.asarray(masks)

    keep = gt_roi_labels != 0
    viz = mask_rcnn.utils.draw_instance_boxes(
        img, sample_rois[keep], gt_roi_labels[keep], n_class=21,
        captions=captions[keep], masks=masks[keep], bg_class=-1)
    vizs.append(viz)

    keep = gt_roi_labels == 0
    viz = mask_rcnn.utils.draw_instance_boxes(
        img, sample_rois[keep], gt_roi_labels[keep], n_class=21,
        captions=captions[keep], masks=masks[keep], bg_class=-1)
    vizs.append(viz)

    return fcn.utils.get_tile_image(vizs)

    # vizs = []
    # for roi, id_cls, gt_roi_mask in \
    #         zip(sample_rois, gt_roi_labels, gt_roi_masks):
    #     if id_cls == 0:
    #         continue
    #     viz = img.copy()
    #     if gt_roi_mask is not None:
    #         mask_ins = np.zeros(img.shape[:2], dtype=bool)
    #         y1, x1, y2, x2 = roi
    #         gt_roi_mask = gt_roi_mask.astype(np.float32)
    #         gt_roi_mask = cv2.resize(gt_roi_mask,
    #                                  (int(x2 - x1), int(y2 - y1)))
    #         gt_roi_mask = np.round(gt_roi_mask).astype(bool)
    #         mask_ins[y1:y2, x1:x2] = gt_roi_mask
    #         viz[~mask_ins] = 255
    #     viz = mask_rcnn.utils.draw_instance_boxes(
    #         viz, [roi], [id_cls], n_class=21, bg_class=0, thickness=2)
    #     vizs.append(viz)
    # viz2 = fcn.utils.get_tile_image(vizs)
    # scale = 1. * viz1.shape[1] / viz2.shape[1]
    # viz2 = cv2.resize(viz2, None, None, fx=scale, fy=scale)
    #
    # return np.vstack([viz1, viz2])


def main():
    dataset = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
    dataset.split = 'train'
    mask_rcnn.datasets.view_dataset(dataset, visualize_func)


if __name__ == '__main__':
    main()

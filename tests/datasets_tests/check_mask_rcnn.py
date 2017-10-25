import math

import cv2
import mvtk
import numpy as np

import mask_rcnn


def visualize_func(dataset, index):
    img, bboxes, labels, lbl_ins = dataset[index]
    labels += 1
    bboxes = bboxes.astype(np.int32)

    viz = mask_rcnn.utils.draw_instance_boxes(
        img, bboxes, labels, n_class=dataset.n_class)

    viz1 = mvtk.image.tile([img, viz])

    instance_ids = np.unique(lbl_ins)
    instance_ids = instance_ids[instance_ids != -1]

    vizs = []
    for idx_ins, (label, bbox) in enumerate(zip(labels, bboxes)):
        ins_id = instance_ids[idx_ins]
        mask = lbl_ins == ins_id
        viz = img.copy()
        viz[~mask] = 255
        y1, x1, y2, x2 = bbox
        viz = viz[y1:y2, x1:x2]
        scale = math.sqrt((400. * 400.) / (viz.shape[0] * viz.shape[1]))
        viz = cv2.resize(viz, None, None, fx=scale, fy=scale)
        H, W = viz.shape[:2]
        caption = dataset._instance_dataset.class_names[label]
        viz = mask_rcnn.utils.draw_instance_boxes(
            viz, [(0, 0, H, W)], [label],
            captions=[caption], n_class=dataset.n_class, thickness=10)
        vizs.append(viz)
    viz2 = mvtk.image.tile(vizs)

    return mvtk.image.tile([viz1, viz2], (2, 1))


def main():
    instance_dataset = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
    dataset = mask_rcnn.datasets.MaskRcnnDataset(instance_dataset)
    dataset.split = 'train'
    dataset.n_class = len(instance_dataset.class_names)
    mvtk.datasets.view_dataset(dataset, visualize_func)


if __name__ == '__main__':
    main()

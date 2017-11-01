import math

import cv2
import mvtk
import numpy as np

import mask_rcnn


def visualize_func(dataset, index):
    img, bboxes, labels, masks = dataset[index]
    bboxes = bboxes.astype(np.int32)
    masks = masks.astype(bool)

    viz = mask_rcnn.utils.draw_instance_boxes(
        img, bboxes, labels, n_class=dataset.n_fg_class)

    viz1 = mvtk.image.tile([img, viz])

    print('[%06d] %s' %
          (index, dataset.fg_class_names[labels]))

    vizs = []
    for label, bbox, mask in zip(labels, bboxes, masks):
        viz = img.copy()
        viz[~mask] = 255
        y1, x1, y2, x2 = bbox
        viz = viz[y1:y2, x1:x2]
        scale = math.sqrt((400. * 400.) / (viz.shape[0] * viz.shape[1]))
        viz = cv2.resize(viz, None, None, fx=scale, fy=scale)
        H, W = viz.shape[:2]
        caption = dataset.fg_class_names[label]
        viz = mask_rcnn.utils.draw_instance_boxes(
            viz, [(0, 0, H, W)], [label],
            captions=[caption], n_class=dataset.n_fg_class, thickness=10)
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

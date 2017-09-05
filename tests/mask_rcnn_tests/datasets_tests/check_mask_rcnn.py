import mvtk

import mask_rcnn


def visualize_func(dataset, index):
    img, bboxes, labels, masks, scale = dataset[index]
    bboxes = bboxes[:, [1, 0, 3, 2]]  # yx -> xy

    viz = mask_rcnn.utils.draw_instance_boxes(
        img, bboxes, labels, n_class=dataset.n_class)

    viz1 = mvtk.image.tile([img, viz])

    vizs = []
    for label, bbox, mask in zip(labels, bboxes, masks):
        viz = img.copy()
        viz[~mask] = 255
        viz = mask_rcnn.utils.draw_instance_boxes(
            viz, [bbox], [label], n_class=21, thickness=2)
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

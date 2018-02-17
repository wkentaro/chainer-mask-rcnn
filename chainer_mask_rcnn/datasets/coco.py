import os.path as osp
import sys

import chainer
import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import skimage.io

from .. import utils
from ..external import pycocotools
from ..external.pycocotools.coco import COCO


class CocoInstanceSeg(chainer.dataset.DatasetMixin):

    class_names = None  # initialized by __init__

    def __init__(self, split,
                 use_crowd=False, return_crowd=False, return_area=False):
        if split == 'train':
            split = split + '2014'
            data_type = 'train2014'
        elif split in ['val', 'minival', 'valminusminival']:
            split = split + '2014'
            data_type = 'val2014'
        else:
            raise ValueError
        dataset_dir = osp.expanduser('~/data/datasets/COCO')
        ann_file = osp.join(
            dataset_dir, 'annotations/instances_%s.json' % split)

        self._use_crowd = use_crowd
        self._return_crowd = return_crowd
        self._return_area = return_area

        # suppress loading message of annotations
        sys.stdout = None
        self.coco = COCO(ann_file)
        sys.stdout = sys.__stdout__

        self.img_fname = osp.join(
            dataset_dir, data_type, 'COCO_%s_{:012}.jpg' % data_type)

        # set class_names
        cats = self.coco.loadCats(self.coco.getCatIds())
        cat_id_to_class_id = {}
        class_names = ['__background__']
        for cat in sorted(cats, key=lambda x: x['id']):
            class_id = len(class_names)
            cat_id_to_class_id[cat['id']] = class_id
            class_names.append(cat['name'])
        class_names = np.asarray(class_names)
        class_names.setflags(write=0)
        self.cat_id_to_class_id = cat_id_to_class_id
        self.class_names = class_names

        self.img_ids = self.coco.getImgIds()

    def get_example(self, i):
        img_id = self.img_ids[i]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_fname = self.img_fname.format(img_id)
        img = skimage.io.imread(img_fname)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        example = self._annotations_to_example(
            anns, img.shape[0], img.shape[1])

        # img, bboxes, labels, masks
        # or img, bboxes, labels, masks, crowds
        # or img, bboxes, labels, masks, areas
        # or img, bboxes, labels, masks, crowds, areas
        return tuple([img] + example)

    def _annotations_to_example(self, anns, height, width):
        bboxes = []
        labels = []
        masks = []
        if self._return_crowd:
            crowds = []
        if self._return_area:
            areas = []
        for ins_id, ann in enumerate(anns):
            if 'segmentation' not in ann:
                continue
            if not self._use_crowd and ann['iscrowd'] == 1:
                continue
            class_id = self.cat_id_to_class_id[ann['category_id']]
            if isinstance(ann['segmentation'], list):
                # polygon
                mask = np.zeros((height, width), dtype=np.uint8)
                mask = PIL.Image.fromarray(mask)
                for seg in ann['segmentation']:
                    xy = np.array(seg).reshape((-1, 2))
                    xy = [tuple(xy_i) for xy_i in xy]
                    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
                mask = np.asarray(mask)
            else:
                # mask
                if isinstance(ann['segmentation']['counts'], list):
                    rle = pycocotools.mask.frPyObjects(
                        [ann['segmentation']], height, width)
                else:
                    rle = [ann['segmentation']]
                mask = pycocotools.mask.decode(rle)[:, :, 0]
                # FIXME: some of minival annotations are malformed.
                if mask.shape != (height, width):
                    continue
            mask = mask == 1  # int32 -> bool
            bbox = utils.mask_to_bbox(mask)  # y1, x1, y2, x2
            bboxes.append(bbox)
            masks.append(mask)
            labels.append(class_id)
            if self._return_crowd:
                crowds.append(ann['iscrowd'])
            if self._return_area:
                areas.append(ann['area'])
        bboxes = np.asarray(bboxes, dtype=np.float32)
        bboxes = bboxes.reshape((-1, 4))
        labels = np.asarray(labels, dtype=np.int32)
        masks = np.asarray(masks, dtype=bool)
        masks = masks.reshape((-1, height, width))
        example = [bboxes, labels, masks]
        if self._return_crowd:
            crowds = np.asarray(crowds, dtype=np.int32)
            example.append(crowds)
        if self._return_area:
            areas = np.asarray(areas, dtype=np.float32)
            example.append(areas)
        return example

    def __len__(self):
        return len(self.img_ids)


if __name__ == '__main__':
    from .view_dataset import view_dataset
    import fcn

    split = 'val'
    dataset = CocoInstanceSeg(split)
    dataset.split = split
    print(dataset.class_names)
    print(len(dataset.class_names))

    def visualize_func(dataset, index):
        img, bboxes, labels, masks = dataset[index]
        captions = [dataset.class_names[l] for l in labels]
        viz = utils.draw_instance_boxes(
            img, bboxes, labels, n_class=len(dataset.class_names),
            masks=masks, captions=captions)
        return fcn.utils.get_tile_image([img, viz])

    view_dataset(dataset, visualize_func)

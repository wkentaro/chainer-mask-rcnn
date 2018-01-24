import os.path as osp

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

    def __init__(self, split):
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
        self.coco = COCO(ann_file)
        self.img_fname = osp.join(
            dataset_dir, data_type, 'COCO_%s_{:012}.jpg' % data_type)

        # set class_names
        labels = self.coco.loadCats(self.coco.getCatIds())
        max_label = max(labels, key=lambda x: x['id'])['id']
        n_label = max_label + 1
        class_names = ['__none__'] * n_label
        for label in labels:
            class_names[label['id']] = label['name']
        class_names[0] = '__background__'
        class_names = np.asarray(class_names)
        class_names.setflags(write=0)
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

        lbl_cls, lbl_ins = self._annotations_to_label(
            anns, img.shape[0], img.shape[1])

        return img, lbl_cls, lbl_ins

    @staticmethod
    def _annotations_to_label(anns, height, width):
        lbl_cls = np.zeros((height, width), dtype=np.int32)
        lbl_ins = - np.ones((height, width), dtype=np.int32)
        for ins_id, ann in enumerate(anns):
            if 'segmentation' not in ann:
                continue
            if isinstance(ann['segmentation'], list):
                # polygon
                for seg in ann['segmentation']:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask = PIL.Image.fromarray(mask)
                    xy = np.array(seg).reshape((-1, 2))
                    xy = [tuple(xy_i) for xy_i in xy]
                    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
                    mask = np.array(mask)
                    lbl_cls[mask == 1] = ann['category_id']
                    lbl_ins[mask == 1] = ins_id
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
                lbl_cls[mask == 1] = ann['category_id']
                lbl_ins[mask == 1] = ins_id
        return lbl_cls, lbl_ins

    def __len__(self):
        return len(self.img_ids)


if __name__ == '__main__':
    from .view_dataset import view_dataset
    import fcn

    split = 'val'
    dataset = CocoInstanceSeg(split)
    dataset.split = split

    def visualize_func(dataset, index):
        img, lbl_cls, lbl_ins = dataset[index]
        viz = utils.visualize_instance_segmentation(
            lbl_ins, lbl_cls, img, dataset.class_names)
        return fcn.utils.get_tile_image([img, viz])

    view_dataset(dataset, visualize_func)

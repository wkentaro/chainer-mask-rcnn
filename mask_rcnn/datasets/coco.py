import os.path as osp

import chainer
import numpy as np
import PIL.Image
import PIL.ImageDraw
import skimage.io

from mvtk.external import pycocotools
from mvtk.external.pycocotools.coco import COCO

from .. import utils


class CocoInstanceSeg(chainer.dataset.DatasetMixin):

    class_names = None  # initialized by __init__

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        data_type = data_type + '2014'
        dataset_dir = osp.expanduser('~/data/datasets/COCO')
        ann_file = osp.join(
            dataset_dir, 'annotations/instances_%s.json' % data_type)
        self.coco = COCO(ann_file)
        self.img_fname = osp.join(
            dataset_dir, data_type, 'COCO_%s_{:012}.jpg' % data_type)

        labels = self.coco.loadCats(self.coco.getCatIds())
        max_label = max(labels, key=lambda x: x['id'])['id']
        n_label = max_label + 1
        self.class_names = [None] * n_label
        for label in labels:
            self.class_names[label['id']] = label['name']
        self.class_names[0] = '__background__'

        self.img_ids = self.coco.getImgIds()

    def get_example(self, i):
        img_id = self.img_ids[i]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_fname = self.img_fname.format(img_id)
        img = skimage.io.imread(img_fname)

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
                    xy = np.array(seg).reshape((len(seg) / 2, 2))
                    xy = map(tuple, xy)
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
                lbl_cls[mask == 1] = ann['category_id']
                lbl_ins[mask == 1] = ins_id
        return lbl_cls, lbl_ins

    def __len__(self):
        return len(self.img_ids)


if __name__ == '__main__':
    import mvtk
    split = 'val'
    dataset = CocoInstanceSeg(split)
    dataset.split = split

    def visualize_func(dataset, index):
        img, lbl_cls, lbl_ins = dataset[index]
        viz = utils.visualize_instance_segmentation(
            lbl_ins, lbl_cls, img, dataset.class_names)
        return mvtk.image.tile([img, viz])

    mvtk.datasets.view_dataset(dataset, visualize_func)

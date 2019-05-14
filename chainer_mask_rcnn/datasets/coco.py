import os
import os.path as osp
import sys
import warnings

import chainer
import cv2
import fcn
import numpy as np
import PIL.Image
import PIL.ImageDraw
import pycocotools
from pycocotools.coco import COCO
import skimage.io

from .. import utils


class COCOInstanceSegmentationDataset(chainer.dataset.DatasetMixin):

    class_names = None  # initialized by __init__
    root_dir = osp.expanduser('~/data/datasets/COCO')

    @classmethod
    def download(cls):
        data = [
            (None,  # '0da8c0bd3d6becc4dcb32757491aca88',
             'http://msvocds.blob.core.windows.net/coco2014/train2014.zip',
             'train2014.zip'),
            (None,  # 'a3d79f5ed8d289b7a7554ce06a5782b3',
             'http://msvocds.blob.core.windows.net/coco2014/val2014.zip',
             'val2014.zip'),
            ('59582776b8dd745d649cd249ada5acf7',
             'http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip',  # NOQA
             'instances_train-val2014.zip'),
            ('395a089042d356d97017bf416e4e99fb',
             'https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip',  # NOQA
             'annotations/instances_minival2014.json.zip'),
            ('f72ed643338e184978e8228948972e84',
             'https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip',  # NOQA
             'annotations/instances_valminusminival2014.json.zip'),
        ]
        for md5, url, basename in data:
            path = osp.join(cls.root_dir, basename)
            fcn.data.cached_download(
                url=url,
                path=path,
                md5=md5,
                postprocess=fcn.data.extract_file,
            )

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
        ann_file = osp.join(
            self.root_dir, 'annotations/instances_%s.json' % split)

        if not osp.exists(ann_file):
            self.download()

        self._use_crowd = use_crowd
        self._return_crowd = return_crowd
        self._return_area = return_area

        # suppress loading message of annotations
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            self.coco = COCO(ann_file)
            sys.stdout = sys.__stdout__

        self.img_fname = osp.join(
            self.root_dir, data_type, 'COCO_%s_{:012}.jpg' % data_type)

        # set class_names
        cats = self.coco.loadCats(self.coco.getCatIds())
        cat_id_to_class_id = {}
        class_names = []
        for cat in sorted(cats, key=lambda x: x['id']):
            class_id = len(class_names)
            cat_id_to_class_id[cat['id']] = class_id
            class_names.append(cat['name'])
        class_names = np.asarray(class_names)
        class_names.setflags(write=0)
        self.cat_id_to_class_id = cat_id_to_class_id
        self.class_names = class_names

        # filter images without any annotations
        img_ids = []
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(img_id)
            if len(ann_ids) >= 1:
                img_ids.append(img_id)
        self.img_ids = img_ids

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
        masks = np.asarray(masks, dtype=np.int32)
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


class CocoInstanceSeg(COCOInstanceSegmentationDataset):

    def __init__(self):
        warnings.warn('CocoInstanceSeg is renamed to '
                      'COCOInstanceSegmentationDataset.')


if __name__ == '__main__':
    from .view_dataset import view_dataset

    split = 'minival'
    dataset = COCOInstanceSegmentationDataset(split)
    dataset.split = split
    print(dataset.class_names)
    print(len(dataset.class_names))

    def visualize_func(dataset, index):
        img, bboxes, labels, masks = dataset[index]
        print('{:08d}: # of instances = {:d}'.format(index, len(bboxes)))
        masks = masks.astype(bool)
        captions = [dataset.class_names[l] for l in labels]
        viz = utils.draw_instance_bboxes(
            img, bboxes, labels + 1, n_class=len(dataset.class_names) + 1,
            masks=masks, captions=captions)
        return fcn.utils.get_tile_image([img, viz])

    view_dataset(dataset, visualize_func)

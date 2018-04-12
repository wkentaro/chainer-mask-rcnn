import os.path as osp
import warnings

import fcn.datasets.voc
import numpy as np
import PIL.Image
import scipy

from ... import utils
from .voc import VOCInstanceSegmentationDatasetBase


here = osp.dirname(osp.realpath(__file__))


class SBDInstanceSegmentationDataset(VOCInstanceSegmentationDatasetBase):

    def __init__(self, split='train'):
        dataset_dir = osp.expanduser(
            '~/data/datasets/VOC/benchmark_RELEASE/dataset')
        if not osp.exists(dataset_dir):
            self.download()
        imgsets_file = osp.join(
            here, 'data/VOCdevkit/VOCSDS/ImageSets/Main/%s.txt' % split)
        self.files = []
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
            cls_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
            ins_file = osp.join(dataset_dir, 'inst/%s.mat' % did)
            self.files.append({
                'img': img_file,
                'cls': cls_file,
                'ins': ins_file,
            })

    def get_example(self, index):
        data_file = self.files[index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load class label
        cls_file = data_file['cls']
        mat = scipy.io.loadmat(cls_file)
        lbl_cls = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl_cls[lbl_cls == 255] = -1
        # load instance label
        ins_file = data_file['ins']
        mat = scipy.io.loadmat(ins_file)
        lbl_ins = mat['GTinst'][0]['Segmentation'][0].astype(np.int32)
        lbl_ins[lbl_ins == 255] = -1
        lbl_ins[np.isin(lbl_cls, [-1, 0])] = -1

        # lbl_ins, lbl_cls -> bboxes, labels, masks
        labels, bboxes, masks = utils.label2instance_boxes(
            lbl_ins, lbl_cls, return_masks=True)
        masks = masks.astype(np.int32, copy=False)
        labels = labels.astype(np.int32, copy=False)
        labels -= 1  # background: 0 -> -1
        bboxes = bboxes.astype(np.float32, copy=False)

        return img, bboxes, labels, masks

    def __len__(self):
        return len(self.files)

    @staticmethod
    def download():
        return fcn.datasets.voc.SBDClassSeg.download()


class SBDInstanceSeg(SBDInstanceSegmentationDataset):

    def __init__(self, *args, **kwargs):
        warnings.warn('SBDInstanceSeg is renamed to '
                      'SBDInstanceSegmentationDataset.')
        super(SBDInstanceSeg, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    from ..view_dataset import view_dataset
    import fcn

    dataset = SBDInstanceSegmentationDataset('val')

    def visualize_func(dataset, index):
        img, bboxes, labels, masks = dataset[index]
        print('[%08d] labels: %s' % (index, labels))
        masks = masks.astype(bool, copy=False)
        captions = [dataset.class_names[l] for l in labels]
        viz = utils.draw_instance_bboxes(
            img, bboxes, labels, n_class=len(dataset.class_names),
            masks=masks, captions=captions, bg_class=-1)
        return fcn.utils.get_tile_image([img, viz])

    view_dataset(dataset, visualize_func)

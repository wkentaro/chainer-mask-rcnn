import os.path as osp

import fcn.datasets.voc
import numpy as np
import PIL.Image
import scipy

from ... import utils
from .voc import VOCInstanceSegBase


here = osp.dirname(osp.realpath(__file__))


class SBDInstanceSeg(VOCInstanceSegBase):

    def __init__(self, split='train'):
        dataset_dir = osp.expanduser(
            '~/data/datasets/VOC/benchmark_RELEASE/dataset')
        imgsets_file = osp.join(
            here, 'data/VOCdevkit/VOCSDS/ImageSets/Main/%s.txt' % split)
        self.files = []
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
            lbl_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
            ins_file = osp.join(dataset_dir, 'inst/%s.mat' % did)
            self.files.append({
                'img': img_file,
                'lbl': lbl_file,
                'ins': ins_file,
            })

    def get_example(self, index):
        data_file = self.files[index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load class label
        lbl_file = data_file['lbl']
        mat = scipy.io.loadmat(lbl_file)
        lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl[lbl == 255] = -1
        # load instance label
        ins_file = data_file['ins']
        mat = scipy.io.loadmat(ins_file)
        ins = mat['GTinst'][0]['Segmentation'][0].astype(np.int32)
        ins[ins == 255] = -1
        return img, lbl, ins

    def __len__(self):
        return len(self.files)

    @staticmethod
    def download():
        return fcn.datasets.voc.SBDClassSeg.download()


if __name__ == '__main__':
    import cv2
    import mvtk
    split = 'val'
    dataset = SBDInstanceSeg(split)
    dataset.split = split

    def visualize_func(dataset, index):
        img, lbl_cls, lbl_ins = dataset[index]
        viz = utils.visualize_instance_segmentation(
            lbl_ins, lbl_cls, img, dataset.class_names)
        return mvtk.image.tile([img, viz])

    mvtk.datasets.view_dataset(dataset, visualize_func)

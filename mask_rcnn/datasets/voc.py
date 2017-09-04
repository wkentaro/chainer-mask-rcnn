import os.path as osp

import chainer
import fcn.datasets.voc
import numpy as np
import PIL.Image
import scipy

from .. import utils


class VOCInstanceSegBase(chainer.dataset.DatasetMixin):

    class_names = fcn.datasets.voc.VOCClassSegBase.class_names


class VOC2012InstanceSeg(VOCInstanceSegBase):

    def __init__(self, split):
        assert split in ('train', 'val')
        dataset_dir = osp.expanduser('~/data/datasets/VOC/VOCdevkit/VOC2012')
        imgsets_file = osp.join(
            dataset_dir, 'ImageSets/Segmentation/{}.txt'.format(split))
        self.files = []
        for i, data_id in enumerate(open(imgsets_file).readlines()):
            data_id = data_id.strip()
            img_file = osp.join(
                dataset_dir, 'JPEGImages/{}.jpg'.format(data_id))
            seg_class_file = osp.join(
                dataset_dir, 'SegmentationClass/{}.png'.format(data_id))
            seg_object_file = osp.join(
                dataset_dir, 'SegmentationObject/{}.png'.format(data_id))
            self.files.append({
                'img': img_file,
                'seg_class': seg_class_file,
                'seg_object': seg_object_file,
            })

    def __len__(self):
        return len(self.files)

    def get_example(self, i):
        """Return data example for instance segmentation of given index.

        Parameters
        ----------
        i: int
            Index of the example.

        Returns
        -------
        datum: numpy.ndarray, (channels, height, width), float32
            Image data.
        lbl_cls: numpy.ndaray, (height, width), int32
            Class label image.
        lbl_ins: numpy.ndarray, (height, width), int32
            Instance label image.
        """
        data_file = self.files[i]
        # load image
        img_file = data_file['img']
        img = scipy.misc.imread(img_file, mode='RGB')
        # load class segmentaion gt
        seg_class_file = data_file['seg_class']
        lbl_cls = PIL.Image.open(seg_class_file)
        lbl_cls = np.array(lbl_cls, dtype=np.int32)
        lbl_cls[lbl_cls == 255] = -1
        # load instance segmentation gt
        seg_object_file = data_file['seg_object']
        lbl_ins = PIL.Image.open(seg_object_file)
        lbl_ins = np.array(lbl_ins, dtype=np.int32)
        lbl_ins[lbl_ins == 255] = -1
        return img, lbl_cls, lbl_ins


if __name__ == '__main__':
    import cv2
    split = 'val'
    dataset = VOC2012InstanceSeg(split)
    for i in xrange(len(dataset)):
        img, lbl_cls, lbl_ins = dataset[i]
        viz = utils.visualize_instance_segmentation(
            lbl_ins, lbl_cls, img, dataset.class_names)
        cv2.imshow('VOC2012 (%s)' % split, viz[:, :, ::-1])
        if cv2.waitKey(0) == ord('q'):
            break

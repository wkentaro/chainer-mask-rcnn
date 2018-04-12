import os.path as osp
import warnings

import chainer
import fcn.datasets.voc
import numpy as np
import PIL.Image
import scipy

from ... import utils


class VOCInstanceSegmentationDatasetBase(chainer.dataset.DatasetMixin):

    class_names = fcn.datasets.voc.VOCClassSegBase.class_names[1:]
    class_names.setflags(write=0)


class VOC2012InstanceSegmentationDataset(VOCInstanceSegmentationDatasetBase):

    def __init__(self, split):
        assert split in ('train', 'val')

        dataset_dir = osp.expanduser('~/data/datasets/VOC/VOCdevkit/VOC2012')
        if not osp.exists(dataset_dir):
            self.download()

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

    @staticmethod
    def download():
        return fcn.datasets.voc.VOC2012ClassSeg.download()

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
        lbl_ins[np.isin(lbl_cls, [-1, 0])] = -1

        # lbl_ins, lbl_cls -> bboxes, labels, masks
        labels, bboxes, masks = utils.label2instance_boxes(
            lbl_ins, lbl_cls, return_masks=True)
        masks = masks.astype(np.int32, copy=False)
        labels = labels.astype(np.int32, copy=False)
        labels -= 1  # background: 0 -> -1
        bboxes = bboxes.astype(np.float32, copy=False)

        return img, bboxes, labels, masks


class VOCInstanceSegBase(VOCInstanceSegmentationDatasetBase):

    def __init__(self, *args, **kwargs):
        warnings.warn('VOCInstanceSegBase is renamed to '
                      'VOC2012InstanceSegmentationDatasetBase.')
        super(VOCInstanceSegBase, self).__init__(*args, **kwargs)


class VOC2012InstanceSeg(VOC2012InstanceSegmentationDataset):

    def __init__(self, *args, **kwargs):
        warnings.warn('VOC2012InstanceSeg is renamed to '
                      'VOC2012InstanceSegmentationDataset.')
        super(VOC2012InstanceSeg, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    from ..view_dataset import view_dataset
    import fcn

    split = 'val'
    dataset = VOC2012InstanceSegmentationDataset(split)

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

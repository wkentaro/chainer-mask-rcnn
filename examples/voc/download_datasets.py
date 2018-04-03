#!/usr/bin/env python

if __name__ == '__main__':
    import chainer_mask_rcnn as mrcnn
    mrcnn.datasets.SBDInstanceSegmentationDataset.download()

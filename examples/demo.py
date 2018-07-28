#!/usr/bin/env python

from __future__ import print_function

import argparse
import os.path as osp
import pprint
import tempfile

import chainer
import numpy as np
import skimage.io
import yaml

import chainer_mask_rcnn as cmr


def main():
    default_img = 'https://raw.githubusercontent.com/facebookresearch/Detectron/master/demo/33823288584_1d21cf0a26_k.jpg'  # NOQA
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('log_dir', help='log dir')
    parser.add_argument(
        '--img',
        '-i',
        nargs='+',
        default=[default_img],
        help='img file or url',
    )
    parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    print('Using image file: {}'.format(args.img))

    # XXX: see also evaluate.py
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # param
    params = yaml.load(open(osp.join(args.log_dir, 'params.yaml')))
    print('Training config:')
    print('# ' + '-' * 77)
    pprint.pprint(params)
    print('# ' + '-' * 77)

    # dataset
    if 'class_names' in params:
        class_names = params['class_names']
    elif params['dataset'] == 'voc':
        test_data = cmr.datasets.SBDInstanceSegmentationDataset('val')
        class_names = test_data.class_names
    elif params['dataset'] == 'coco':
        test_data = cmr.datasets.COCOInstanceSegmentationDataset('minival')
        class_names = test_data.class_names
    else:
        raise ValueError

    # model

    if params['dataset'] == 'voc':
        if 'min_size' not in params:
            params['min_size'] = 600
        if 'max_size' not in params:
            params['max_size'] = 1000
        if 'anchor_scales' not in params:
            params['anchor_scales'] = (4, 8, 16, 32)
    elif params['dataset'] == 'coco':
        if 'min_size' not in params:
            params['min_size'] = 800
        if 'max_size' not in params:
            params['max_size'] = 1333
        if 'anchor_scales' not in params:
            params['anchor_scales'] = (2, 4, 8, 16, 32)
    else:
        assert 'min_size' in params
        assert 'max_size' in params
        assert 'anchor_scales' in params

    if params['pooling_func'] == 'align':
        pooling_func = cmr.functions.roi_align_2d
    elif params['pooling_func'] == 'pooling':
        pooling_func = cmr.functions.roi_pooling_2d
    elif params['pooling_func'] == 'resize':
        pooling_func = cmr.functions.crop_and_resize
    else:
        raise ValueError(
            'Unsupported pooling_func: {}'.format(params['pooling_func'])
        )

    pretrained_model = osp.join(args.log_dir, 'snapshot_model.npz')
    print('Using pretrained_model: %s' % pretrained_model)

    model = params['model']
    mask_rcnn = cmr.models.MaskRCNNResNet(
        n_layers=int(model.lstrip('resnet')),
        n_fg_class=len(class_names),
        pretrained_model=pretrained_model,
        pooling_func=pooling_func,
        anchor_scales=params['anchor_scales'],
        mean=params.get('mean', (123.152, 115.903, 103.063)),
        min_size=params['min_size'],
        max_size=params['max_size'],
        roi_size=params.get('roi_size', 7),
    )
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        mask_rcnn.to_gpu()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    imgs_chw = []
    for img_file in args.img:
        img = skimage.io.imread(img_file)
        img_chw = img.transpose(2, 0, 1)
        imgs_chw.append(img_chw)
        del img, img_chw

    def batch_predict(mask_rcnn, imgs_chw):
        for batch in cmr.utils.batch(imgs_chw, n=2):
            bboxes, masks, labels, scores = mask_rcnn.predict(batch)
            for bbox, mask, label, score in zip(bboxes, masks, labels, scores):
                yield bbox, mask, label, score

    results = batch_predict(mask_rcnn, imgs_chw)

    out_dir = tempfile.mkdtemp(dir='.')

    for img_file, img_chw, (bbox, mask, label, score) in \
            zip(args.img, imgs_chw, results):
        img = img_chw.transpose(1, 2, 0)
        del img_chw

        k = score >= 0.7
        bbox, mask, label, score = bbox[k], mask[k], label[k], score[k]
        i = np.argsort(score)
        bbox, mask, label, score = bbox[i], mask[i], label[i], score[i]

        captions = [
            '{}: {:.1%}'.format(class_names[l], s)
            for l, s in zip(label, score)
        ]
        for caption in captions:
            print(caption)
        viz = cmr.utils.draw_instance_bboxes(
            img=img,
            bboxes=bbox,
            labels=label + 1,
            n_class=len(class_names) + 1,
            captions=captions,
            masks=mask,
        )
        out_file = osp.join(out_dir, osp.basename(img_file))
        skimage.io.imsave(out_file, viz)
        print('Saved result: {}'.format(out_file))


if __name__ == '__main__':
    main()

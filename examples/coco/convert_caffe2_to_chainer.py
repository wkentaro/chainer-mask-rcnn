#!/usr/bin/env python

import os
import os.path as osp
import pickle
import shutil

import chainer
import chainercv
import numpy as np
import six
import yaml

from chainer_mask_rcnn.models import MaskRCNNResNet


dataset_dir = chainer.dataset.get_dataset_directory(
    'wkentaro/chainer-mask-rcnn/R-50-C4_1x_caffe2')
dst_file = osp.join(dataset_dir, 'model_final_caffe2_to_chainer.npz')
if osp.exists(dst_file):
    print('Model file already exists: {}'.format(dst_file))
    quit()

src_file = osp.join(dataset_dir, 'model_final.pkl')
if not osp.exists(src_file):
    url = 'https://dl.fbaipublicfiles.com/detectron/36224121/12_2017_baselines/mask_rcnn_R-50-C4_1x.yaml.08_24_37.wdU8r5Jo/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl'  # NOQA
    cache_path = chainercv.utils.download.cached_download(url)
    shutil.move(cache_path, src_file)

print('Loading from: {}'.format(src_file))
with open(src_file, 'rb') as f:
    if six.PY2:
        blobs = pickle.load(f)['blobs']
    else:
        blobs = pickle.load(f, encoding='latin-1')['blobs']

model = MaskRCNNResNet(
    n_layers=50,
    n_fg_class=80,
    anchor_scales=[2, 4, 8, 16, 32],
    pretrained_model=None,
    roi_size=14,
)

# /conv1, /bn1
assert all(isinstance(v, np.ndarray) for v in blobs.values())
np.copyto(model.extractor.conv1.W.data, blobs['conv1_w'][:, ::-1])
np.copyto(model.extractor.conv1.b.data, blobs['conv1_b'])
np.copyto(model.extractor.bn1.W.data, blobs['res_conv1_bn_s'])
np.copyto(model.extractor.bn1.b.data, blobs['res_conv1_bn_b'])
# /res2/a
np.copyto(model.extractor.res2.a.conv1.W.data, blobs['res2_0_branch2a_w'])
np.copyto(model.extractor.res2.a.bn1.W.data, blobs['res2_0_branch2a_bn_s'])
np.copyto(model.extractor.res2.a.bn1.b.data, blobs['res2_0_branch2a_bn_b'])
np.copyto(model.extractor.res2.a.conv2.W.data, blobs['res2_0_branch2b_w'])
np.copyto(model.extractor.res2.a.bn2.W.data, blobs['res2_0_branch2b_bn_s'])
np.copyto(model.extractor.res2.a.bn2.b.data, blobs['res2_0_branch2b_bn_b'])
np.copyto(model.extractor.res2.a.conv3.W.data, blobs['res2_0_branch2c_w'])
np.copyto(model.extractor.res2.a.bn3.W.data, blobs['res2_0_branch2c_bn_s'])
np.copyto(model.extractor.res2.a.bn3.b.data, blobs['res2_0_branch2c_bn_b'])
np.copyto(model.extractor.res2.a.conv4.W.data, blobs['res2_0_branch1_w'])
np.copyto(model.extractor.res2.a.bn4.W.data, blobs['res2_0_branch1_bn_s'])
np.copyto(model.extractor.res2.a.bn4.b.data, blobs['res2_0_branch1_bn_b'])
# /res2/b1, /res2/b2
np.copyto(model.extractor.res2.b1.conv1.W.data, blobs['res2_1_branch2a_w'])
np.copyto(model.extractor.res2.b1.bn1.W.data, blobs['res2_1_branch2a_bn_s'])
np.copyto(model.extractor.res2.b1.bn1.b.data, blobs['res2_1_branch2a_bn_b'])
np.copyto(model.extractor.res2.b1.conv2.W.data, blobs['res2_1_branch2b_w'])
np.copyto(model.extractor.res2.b1.bn2.W.data, blobs['res2_1_branch2b_bn_s'])
np.copyto(model.extractor.res2.b1.bn2.b.data, blobs['res2_1_branch2b_bn_b'])
np.copyto(model.extractor.res2.b1.conv3.W.data, blobs['res2_1_branch2c_w'])
np.copyto(model.extractor.res2.b1.bn3.W.data, blobs['res2_1_branch2c_bn_s'])
np.copyto(model.extractor.res2.b1.bn3.b.data, blobs['res2_1_branch2c_bn_b'])
np.copyto(model.extractor.res2.b2.conv1.W.data, blobs['res2_2_branch2a_w'])
np.copyto(model.extractor.res2.b2.bn1.W.data, blobs['res2_2_branch2a_bn_s'])
np.copyto(model.extractor.res2.b2.bn1.b.data, blobs['res2_2_branch2a_bn_b'])
np.copyto(model.extractor.res2.b2.conv2.W.data, blobs['res2_2_branch2b_w'])
np.copyto(model.extractor.res2.b2.bn2.W.data, blobs['res2_2_branch2b_bn_s'])
np.copyto(model.extractor.res2.b2.bn2.b.data, blobs['res2_2_branch2b_bn_b'])
np.copyto(model.extractor.res2.b2.conv3.W.data, blobs['res2_2_branch2c_w'])
np.copyto(model.extractor.res2.b2.bn3.W.data, blobs['res2_2_branch2c_bn_s'])
np.copyto(model.extractor.res2.b2.bn3.b.data, blobs['res2_2_branch2c_bn_b'])
# /res3/a
np.copyto(model.extractor.res3.a.conv1.W.data, blobs['res3_0_branch2a_w'])
np.copyto(model.extractor.res3.a.bn1.W.data, blobs['res3_0_branch2a_bn_s'])
np.copyto(model.extractor.res3.a.bn1.b.data, blobs['res3_0_branch2a_bn_b'])
np.copyto(model.extractor.res3.a.conv2.W.data, blobs['res3_0_branch2b_w'])
np.copyto(model.extractor.res3.a.bn2.W.data, blobs['res3_0_branch2b_bn_s'])
np.copyto(model.extractor.res3.a.bn2.b.data, blobs['res3_0_branch2b_bn_b'])
np.copyto(model.extractor.res3.a.conv3.W.data, blobs['res3_0_branch2c_w'])
np.copyto(model.extractor.res3.a.bn3.W.data, blobs['res3_0_branch2c_bn_s'])
np.copyto(model.extractor.res3.a.bn3.b.data, blobs['res3_0_branch2c_bn_b'])
np.copyto(model.extractor.res3.a.conv4.W.data, blobs['res3_0_branch1_w'])
np.copyto(model.extractor.res3.a.bn4.W.data, blobs['res3_0_branch1_bn_s'])
np.copyto(model.extractor.res3.a.bn4.b.data, blobs['res3_0_branch1_bn_b'])
# /res3/b1, /res3/b2, /res3/b3
np.copyto(model.extractor.res3.b1.conv1.W.data, blobs['res3_1_branch2a_w'])
np.copyto(model.extractor.res3.b1.bn1.W.data, blobs['res3_1_branch2a_bn_s'])
np.copyto(model.extractor.res3.b1.bn1.b.data, blobs['res3_1_branch2a_bn_b'])
np.copyto(model.extractor.res3.b1.conv2.W.data, blobs['res3_1_branch2b_w'])
np.copyto(model.extractor.res3.b1.bn2.W.data, blobs['res3_1_branch2b_bn_s'])
np.copyto(model.extractor.res3.b1.bn2.b.data, blobs['res3_1_branch2b_bn_b'])
np.copyto(model.extractor.res3.b1.conv3.W.data, blobs['res3_1_branch2c_w'])
np.copyto(model.extractor.res3.b1.bn3.W.data, blobs['res3_1_branch2c_bn_s'])
np.copyto(model.extractor.res3.b1.bn3.b.data, blobs['res3_1_branch2c_bn_b'])
np.copyto(model.extractor.res3.b2.conv1.W.data, blobs['res3_2_branch2a_w'])
np.copyto(model.extractor.res3.b2.bn1.W.data, blobs['res3_2_branch2a_bn_s'])
np.copyto(model.extractor.res3.b2.bn1.b.data, blobs['res3_2_branch2a_bn_b'])
np.copyto(model.extractor.res3.b2.conv2.W.data, blobs['res3_2_branch2b_w'])
np.copyto(model.extractor.res3.b2.bn2.W.data, blobs['res3_2_branch2b_bn_s'])
np.copyto(model.extractor.res3.b2.bn2.b.data, blobs['res3_2_branch2b_bn_b'])
np.copyto(model.extractor.res3.b2.conv3.W.data, blobs['res3_2_branch2c_w'])
np.copyto(model.extractor.res3.b2.bn3.W.data, blobs['res3_2_branch2c_bn_s'])
np.copyto(model.extractor.res3.b2.bn3.b.data, blobs['res3_2_branch2c_bn_b'])
np.copyto(model.extractor.res3.b3.conv1.W.data, blobs['res3_3_branch2a_w'])
np.copyto(model.extractor.res3.b3.bn1.W.data, blobs['res3_3_branch2a_bn_s'])
np.copyto(model.extractor.res3.b3.bn1.b.data, blobs['res3_3_branch2a_bn_b'])
np.copyto(model.extractor.res3.b3.conv2.W.data, blobs['res3_3_branch2b_w'])
np.copyto(model.extractor.res3.b3.bn2.W.data, blobs['res3_3_branch2b_bn_s'])
np.copyto(model.extractor.res3.b3.bn2.b.data, blobs['res3_3_branch2b_bn_b'])
np.copyto(model.extractor.res3.b3.conv3.W.data, blobs['res3_3_branch2c_w'])
np.copyto(model.extractor.res3.b3.bn3.W.data, blobs['res3_3_branch2c_bn_s'])
np.copyto(model.extractor.res3.b3.bn3.b.data, blobs['res3_3_branch2c_bn_b'])
# /res4/a
np.copyto(model.extractor.res4.a.conv1.W.data, blobs['res4_0_branch2a_w'])
np.copyto(model.extractor.res4.a.bn1.W.data, blobs['res4_0_branch2a_bn_s'])
np.copyto(model.extractor.res4.a.bn1.b.data, blobs['res4_0_branch2a_bn_b'])
np.copyto(model.extractor.res4.a.conv2.W.data, blobs['res4_0_branch2b_w'])
np.copyto(model.extractor.res4.a.bn2.W.data, blobs['res4_0_branch2b_bn_s'])
np.copyto(model.extractor.res4.a.bn2.b.data, blobs['res4_0_branch2b_bn_b'])
np.copyto(model.extractor.res4.a.conv3.W.data, blobs['res4_0_branch2c_w'])
np.copyto(model.extractor.res4.a.bn3.W.data, blobs['res4_0_branch2c_bn_s'])
np.copyto(model.extractor.res4.a.bn3.b.data, blobs['res4_0_branch2c_bn_b'])
np.copyto(model.extractor.res4.a.conv4.W.data, blobs['res4_0_branch1_w'])
np.copyto(model.extractor.res4.a.bn4.W.data, blobs['res4_0_branch1_bn_s'])
np.copyto(model.extractor.res4.a.bn4.b.data, blobs['res4_0_branch1_bn_b'])
# /res4/b1, /res4/b2, /res4/b3, /res4/b4, /res4/b5
np.copyto(model.extractor.res4.b1.conv1.W.data, blobs['res4_1_branch2a_w'])
np.copyto(model.extractor.res4.b1.bn1.W.data, blobs['res4_1_branch2a_bn_s'])
np.copyto(model.extractor.res4.b1.bn1.b.data, blobs['res4_1_branch2a_bn_b'])
np.copyto(model.extractor.res4.b1.conv2.W.data, blobs['res4_1_branch2b_w'])
np.copyto(model.extractor.res4.b1.bn2.W.data, blobs['res4_1_branch2b_bn_s'])
np.copyto(model.extractor.res4.b1.bn2.b.data, blobs['res4_1_branch2b_bn_b'])
np.copyto(model.extractor.res4.b1.conv3.W.data, blobs['res4_1_branch2c_w'])
np.copyto(model.extractor.res4.b1.bn3.W.data, blobs['res4_1_branch2c_bn_s'])
np.copyto(model.extractor.res4.b1.bn3.b.data, blobs['res4_1_branch2c_bn_b'])
np.copyto(model.extractor.res4.b2.conv1.W.data, blobs['res4_2_branch2a_w'])
np.copyto(model.extractor.res4.b2.bn1.W.data, blobs['res4_2_branch2a_bn_s'])
np.copyto(model.extractor.res4.b2.bn1.b.data, blobs['res4_2_branch2a_bn_b'])
np.copyto(model.extractor.res4.b2.conv2.W.data, blobs['res4_2_branch2b_w'])
np.copyto(model.extractor.res4.b2.bn2.W.data, blobs['res4_2_branch2b_bn_s'])
np.copyto(model.extractor.res4.b2.bn2.b.data, blobs['res4_2_branch2b_bn_b'])
np.copyto(model.extractor.res4.b2.conv3.W.data, blobs['res4_2_branch2c_w'])
np.copyto(model.extractor.res4.b2.bn3.W.data, blobs['res4_2_branch2c_bn_s'])
np.copyto(model.extractor.res4.b2.bn3.b.data, blobs['res4_2_branch2c_bn_b'])
np.copyto(model.extractor.res4.b3.conv1.W.data, blobs['res4_3_branch2a_w'])
np.copyto(model.extractor.res4.b3.bn1.W.data, blobs['res4_3_branch2a_bn_s'])
np.copyto(model.extractor.res4.b3.bn1.b.data, blobs['res4_3_branch2a_bn_b'])
np.copyto(model.extractor.res4.b3.conv2.W.data, blobs['res4_3_branch2b_w'])
np.copyto(model.extractor.res4.b3.bn2.W.data, blobs['res4_3_branch2b_bn_s'])
np.copyto(model.extractor.res4.b3.bn2.b.data, blobs['res4_3_branch2b_bn_b'])
np.copyto(model.extractor.res4.b3.conv3.W.data, blobs['res4_3_branch2c_w'])
np.copyto(model.extractor.res4.b3.bn3.W.data, blobs['res4_3_branch2c_bn_s'])
np.copyto(model.extractor.res4.b3.bn3.b.data, blobs['res4_3_branch2c_bn_b'])
np.copyto(model.extractor.res4.b4.conv1.W.data, blobs['res4_4_branch2a_w'])
np.copyto(model.extractor.res4.b4.bn1.W.data, blobs['res4_4_branch2a_bn_s'])
np.copyto(model.extractor.res4.b4.bn1.b.data, blobs['res4_4_branch2a_bn_b'])
np.copyto(model.extractor.res4.b4.conv2.W.data, blobs['res4_4_branch2b_w'])
np.copyto(model.extractor.res4.b4.bn2.W.data, blobs['res4_4_branch2b_bn_s'])
np.copyto(model.extractor.res4.b4.bn2.b.data, blobs['res4_4_branch2b_bn_b'])
np.copyto(model.extractor.res4.b4.conv3.W.data, blobs['res4_4_branch2c_w'])
np.copyto(model.extractor.res4.b4.bn3.W.data, blobs['res4_4_branch2c_bn_s'])
np.copyto(model.extractor.res4.b4.bn3.b.data, blobs['res4_4_branch2c_bn_b'])
np.copyto(model.extractor.res4.b5.conv1.W.data, blobs['res4_5_branch2a_w'])
np.copyto(model.extractor.res4.b5.bn1.W.data, blobs['res4_5_branch2a_bn_s'])
np.copyto(model.extractor.res4.b5.bn1.b.data, blobs['res4_5_branch2a_bn_b'])
np.copyto(model.extractor.res4.b5.conv2.W.data, blobs['res4_5_branch2b_w'])
np.copyto(model.extractor.res4.b5.bn2.W.data, blobs['res4_5_branch2b_bn_s'])
np.copyto(model.extractor.res4.b5.bn2.b.data, blobs['res4_5_branch2b_bn_b'])
np.copyto(model.extractor.res4.b5.conv3.W.data, blobs['res4_5_branch2c_w'])
np.copyto(model.extractor.res4.b5.bn3.W.data, blobs['res4_5_branch2c_bn_s'])
np.copyto(model.extractor.res4.b5.bn3.b.data, blobs['res4_5_branch2c_bn_b'])
# /rpn: dx, dy, dw, dh -> dy, dx, dh, dw
np.copyto(model.rpn.conv1.W.data, blobs['conv_rpn_w'])
np.copyto(model.rpn.conv1.b.data, blobs['conv_rpn_b'])
W = blobs['rpn_bbox_pred_w']
W = W.reshape(15, 4, 1024, 1, 1)
W = W[:, [1, 0, 3, 2], :, :, :]
W = W.reshape(15 * 4, 1024, 1, 1)
np.copyto(model.rpn.loc.W.data, W)
b = blobs['rpn_bbox_pred_b']
b = b.reshape(15, 4)
b = b[:, [1, 0, 3, 2]]
b = b.reshape(60)
np.copyto(model.rpn.loc.b.data, b)
np.copyto(model.rpn.score.W.data, blobs['rpn_cls_logits_w'])
np.copyto(model.rpn.score.b.data, blobs['rpn_cls_logits_b'])
# /head/res5/a
np.copyto(model.head.res5.a.conv1.W.data, blobs['res5_0_branch2a_w'])
np.copyto(model.head.res5.a.bn1.W.data, blobs['res5_0_branch2a_bn_s'])
np.copyto(model.head.res5.a.bn1.b.data, blobs['res5_0_branch2a_bn_b'])
np.copyto(model.head.res5.a.conv2.W.data, blobs['res5_0_branch2b_w'])
np.copyto(model.head.res5.a.bn2.W.data, blobs['res5_0_branch2b_bn_s'])
np.copyto(model.head.res5.a.bn2.b.data, blobs['res5_0_branch2b_bn_b'])
np.copyto(model.head.res5.a.conv3.W.data, blobs['res5_0_branch2c_w'])
np.copyto(model.head.res5.a.bn3.W.data, blobs['res5_0_branch2c_bn_s'])
np.copyto(model.head.res5.a.bn3.b.data, blobs['res5_0_branch2c_bn_b'])
np.copyto(model.head.res5.a.conv4.W.data, blobs['res5_0_branch1_w'])
np.copyto(model.head.res5.a.bn4.W.data, blobs['res5_0_branch1_bn_s'])
np.copyto(model.head.res5.a.bn4.b.data, blobs['res5_0_branch1_bn_b'])
# /head/res5/b1, /head/res5/b2
np.copyto(model.head.res5.b1.conv1.W.data, blobs['res5_1_branch2a_w'])
np.copyto(model.head.res5.b1.bn1.W.data, blobs['res5_1_branch2a_bn_s'])
np.copyto(model.head.res5.b1.bn1.b.data, blobs['res5_1_branch2a_bn_b'])
np.copyto(model.head.res5.b1.conv2.W.data, blobs['res5_1_branch2b_w'])
np.copyto(model.head.res5.b1.bn2.W.data, blobs['res5_1_branch2b_bn_s'])
np.copyto(model.head.res5.b1.bn2.b.data, blobs['res5_1_branch2b_bn_b'])
np.copyto(model.head.res5.b1.conv3.W.data, blobs['res5_1_branch2c_w'])
np.copyto(model.head.res5.b1.bn3.W.data, blobs['res5_1_branch2c_bn_s'])
np.copyto(model.head.res5.b1.bn3.b.data, blobs['res5_1_branch2c_bn_b'])
np.copyto(model.head.res5.b2.conv1.W.data, blobs['res5_2_branch2a_w'])
np.copyto(model.head.res5.b2.bn1.W.data, blobs['res5_2_branch2a_bn_s'])
np.copyto(model.head.res5.b2.bn1.b.data, blobs['res5_2_branch2a_bn_b'])
np.copyto(model.head.res5.b2.conv2.W.data, blobs['res5_2_branch2b_w'])
np.copyto(model.head.res5.b2.bn2.W.data, blobs['res5_2_branch2b_bn_s'])
np.copyto(model.head.res5.b2.bn2.b.data, blobs['res5_2_branch2b_bn_b'])
np.copyto(model.head.res5.b2.conv3.W.data, blobs['res5_2_branch2c_w'])
np.copyto(model.head.res5.b2.bn3.W.data, blobs['res5_2_branch2c_bn_s'])
np.copyto(model.head.res5.b2.bn3.b.data, blobs['res5_2_branch2c_bn_b'])
# /head/score: dx, dy, dw, dh -> dy, dx, dh, dw
np.copyto(model.head.score.W.data, blobs['cls_score_w'])
np.copyto(model.head.score.b.data, blobs['cls_score_b'])
W = blobs['bbox_pred_w']
W = W.reshape(81, 4, 2048)
W = W[:, [1, 0, 3, 2], :]
W = W.reshape(324, 2048)
# /head/cls_loc
np.copyto(model.head.cls_loc.W.data, W)
b = blobs['bbox_pred_b']
b = b.reshape(81, 4)
b = b[:, [1, 0, 3, 2]]
b = b.reshape(324)
np.copyto(model.head.cls_loc.b.data, b)
# /head/deconv6
np.copyto(model.head.deconv6.W.data, blobs['conv5_mask_w'])
np.copyto(model.head.deconv6.b.data, blobs['conv5_mask_b'])
# /head/mask: remove background class
np.copyto(model.head.mask.W.data, blobs['mask_fcn_logits_w'][1:])
np.copyto(model.head.mask.b.data, blobs['mask_fcn_logits_b'][1:])

# -----------------------------------------------------------------------------

params_src = []
for k, v in sorted(blobs.items()):
    if k.endswith('_momentum'):
        continue
    if k.startswith('fc1000'):
        continue
    if (k.endswith('branch1_b') or k.endswith('branch2a_b') or
            k.endswith('branch2b_b') or k.endswith('branch2c_b')):
        continue
    if k.startswith('mask_fcn_logits_'):
        v = v[1:]
    params_src.extend(v.flatten().tolist())
params_src = np.asarray(params_src)
print(params_src.shape, params_src.min(), params_src.mean(), params_src.max())

params_dst = []
for k, v in model.namedparams():
    v = v.data
    params_dst.extend(v.flatten().tolist())
params_dst = np.asarray(params_dst)
print(params_dst.shape, params_dst.min(), params_dst.mean(), params_dst.max())

# -----------------------------------------------------------------------------

chainer.serializers.save_npz(dst_file, model)
print('Saved to: {}'.format(dst_file))

here = osp.dirname(osp.abspath(__file__))
log_dir = osp.join(here, 'logs/R-50-C4_x1_caffe2_to_chainer')
if not osp.exists(log_dir):
    os.makedirs(log_dir)
link_file = osp.join(log_dir, 'snapshot_model.npz')
if not osp.exists(link_file):
    os.symlink(dst_file, link_file)
yaml_file = osp.join(log_dir, 'params.yaml')
with open(yaml_file, 'w') as f:
    # 0: person ... 79: toothbrush
    with open('coco_class_names.txt') as f2:
        class_names = [n.strip() for n in f2]
    params = dict(
        model='resnet50',
        pooling_func='align',
        roi_size=14,
        mean=(122.7717, 115.9465, 102.9801),
        dataset='coco',
        anchor_scales=(2, 4, 8, 16, 32),
        min_size=800,
        max_size=1333,
        class_names=class_names,
    )
    yaml.safe_dump(params, f, default_flow_style=False)

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

from chainer_mask_rcnn.models import MaskRCNNResNetFPN

from convert_caffe2_to_chainer import convert_caffe2_to_chainer_resnet50_c4
from convert_caffe2_to_chainer import convert_caffe2_to_chainer_resnet50_res5


dataset_dir = chainer.dataset.get_dataset_directory(
    'wkentaro/chainer-mask-rcnn/R-50-FPN_1x_caffe2')
dst_file = osp.join(dataset_dir, 'model_final_caffe2_to_chainer.npz')
if osp.exists(dst_file):
    print('Model file already exists: {}'.format(dst_file))
    quit()

src_file = osp.join(dataset_dir, 'model_final.pkl')
if not osp.exists(src_file):
    url = 'https://s3-us-west-2.amazonaws.com/detectron/35858933/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl'  # NOQA
    cache_path = chainercv.utils.download.cached_download(url)
    shutil.move(cache_path, src_file)

print('Loading from: {}'.format(src_file))
with open(src_file, 'rb') as f:
    if six.PY2:
        blobs = pickle.load(f)['blobs']
    else:
        blobs = pickle.load(f, encoding='latin-1')['blobs']
assert all(isinstance(v, np.ndarray) for v in blobs.values())

model = MaskRCNNResNetFPN(
    n_layers=50,
    n_fg_class=80,
    anchor_scales=[2, 4, 8, 16, 32],
    pretrained_model=None,
)

# class MyDict(object):
#
#     def __init__(self, data):
#         self.data = data
#         self.all_keys = []
#         for k in self.data:
#             if k.endswith('_momentum'):
#                 continue
#             if k.startswith('fc1000'):
#                 continue
#             if (k.endswith('branch1_b') or k.endswith('branch2a_b') or
#                     k.endswith('branch2b_b') or k.endswith('branch2c_b')):
#                 continue
#             self.all_keys.append(k)
#         self.used_keys = []
#
#     def __getitem__(self, key):
#         self.used_keys.append(key)
#         return self.data[key]
#
#     @property
#     def left_keys(self):
#         left_keys = []
#         for key in self.all_keys:
#             if key not in self.used_keys:
#                 left_keys.append(key)
#         return left_keys
#
# blobs = MyDict(blobs)


convert_caffe2_to_chainer_resnet50_c4(model.extractor.extractor, blobs)
convert_caffe2_to_chainer_resnet50_res5(model.extractor.extractor, blobs)
np.copyto(model.extractor.toplayer.W.data, blobs['fpn_inner_res5_2_sum_w'])
np.copyto(model.extractor.toplayer.b.data, blobs['fpn_inner_res5_2_sum_b'])
np.copyto(model.extractor.lat_p4.W.data,
          blobs['fpn_inner_res4_5_sum_lateral_w'])
np.copyto(model.extractor.lat_p4.b.data,
          blobs['fpn_inner_res4_5_sum_lateral_b'])
np.copyto(model.extractor.lat_p3.W.data,
          blobs['fpn_inner_res3_3_sum_lateral_w'])
np.copyto(model.extractor.lat_p3.b.data,
          blobs['fpn_inner_res3_3_sum_lateral_b'])
np.copyto(model.extractor.lat_p2.W.data,
          blobs['fpn_inner_res2_2_sum_lateral_w'])
np.copyto(model.extractor.lat_p2.b.data,
          blobs['fpn_inner_res2_2_sum_lateral_b'])
np.copyto(model.extractor.conv_p5.W.data, blobs['fpn_res5_2_sum_w'])
np.copyto(model.extractor.conv_p5.b.data, blobs['fpn_res5_2_sum_b'])
np.copyto(model.extractor.conv_p4.W.data, blobs['fpn_res4_5_sum_w'])
np.copyto(model.extractor.conv_p4.b.data, blobs['fpn_res4_5_sum_b'])
np.copyto(model.extractor.conv_p3.W.data, blobs['fpn_res3_3_sum_w'])
np.copyto(model.extractor.conv_p3.b.data, blobs['fpn_res3_3_sum_b'])
np.copyto(model.extractor.conv_p2.W.data, blobs['fpn_res2_2_sum_w'])
np.copyto(model.extractor.conv_p2.b.data, blobs['fpn_res2_2_sum_b'])


def convert_caffe2_to_chainer_rpn_fpn(rpn, blobs):
    # /rpn: dx, dy, dw, dh -> dy, dx, dh, dw
    np.copyto(rpn.conv.W.data, blobs['conv_rpn_fpn2_w'])
    np.copyto(rpn.conv.b.data, blobs['conv_rpn_fpn2_b'])
    W = blobs['rpn_bbox_pred_fpn2_w']
    W = W.reshape(3, 4, 256, 1, 1)
    W = W[:, [1, 0, 3, 2], :, :, :]
    W = W.reshape(3 * 4, 256, 1, 1)
    np.copyto(rpn.loc.W.data, W)
    b = blobs['rpn_bbox_pred_fpn2_b']
    b = b.reshape(3, 4)
    b = b[:, [1, 0, 3, 2]]
    b = b.reshape(-1)
    np.copyto(rpn.loc.b.data, b)
    np.copyto(rpn.score.W.data, blobs['rpn_cls_logits_fpn2_w'])
    np.copyto(rpn.score.b.data, blobs['rpn_cls_logits_fpn2_b'])


convert_caffe2_to_chainer_rpn_fpn(model.rpn, blobs)

np.copyto(model.head.fc1.W.data, blobs['fc6_w'])
np.copyto(model.head.fc1.b.data, blobs['fc6_b'])
np.copyto(model.head.fc2.W.data, blobs['fc7_w'])
np.copyto(model.head.fc2.b.data, blobs['fc7_b'])
np.copyto(model.head.score.W.data, blobs['cls_score_w'])
np.copyto(model.head.score.b.data, blobs['cls_score_b'])
W = blobs['bbox_pred_w']
W = W.reshape(81, 4, 1024)
W = W[:, [1, 0, 3, 2], :]
W = W.reshape(324, 1024)
np.copyto(model.head.cls_loc.W.data, W)
b = blobs['bbox_pred_b']
b = b.reshape(81, 4)
b = b[:, [1, 0, 3, 2]]
b = b.reshape(324)
np.copyto(model.head.cls_loc.b.data, b)

np.copyto(model.head.mask1.W.data, blobs['_[mask]_fcn1_w'])
np.copyto(model.head.mask1.b.data, blobs['_[mask]_fcn1_b'])
np.copyto(model.head.mask2.W.data, blobs['_[mask]_fcn2_w'])
np.copyto(model.head.mask2.b.data, blobs['_[mask]_fcn2_b'])
np.copyto(model.head.mask3.W.data, blobs['_[mask]_fcn3_w'])
np.copyto(model.head.mask3.b.data, blobs['_[mask]_fcn3_b'])
np.copyto(model.head.mask4.W.data, blobs['_[mask]_fcn4_w'])
np.copyto(model.head.mask4.b.data, blobs['_[mask]_fcn4_b'])
np.copyto(model.head.deconv1.W.data, blobs['conv5_mask_w'])
np.copyto(model.head.deconv1.b.data, blobs['conv5_mask_b'])
np.copyto(model.head.conv2.W.data, blobs['mask_fcn_logits_w'][1:])
np.copyto(model.head.conv2.b.data, blobs['mask_fcn_logits_b'][1:])


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
print(params_src.shape, params_src.min(), params_src.mean(),
      params_src.std(), params_src.max())

params_dst = []
for k, v in model.namedparams():
    v = v.data
    if v is None:
        print(k, v)
    params_dst.extend(v.flatten().tolist())
params_dst = np.asarray(params_dst)
print(params_dst.shape, params_dst.min(), params_dst.mean(),
      params_dst.std(), params_dst.max())

# -----------------------------------------------------------------------------

chainer.serializers.save_npz(dst_file, model)
print('Saved to: {}'.format(dst_file))

here = osp.dirname(osp.abspath(__file__))
log_dir = osp.join(here, 'logs/R-50-FPN_x1_caffe2_to_chainer')
if not osp.exists(log_dir):
    os.makedirs(log_dir)
link_file = osp.join(log_dir, 'snapshot_model.npz')
if not osp.exists(link_file):
    os.symlink(dst_file, link_file)
yaml_file = osp.join(log_dir, 'params.yaml')
with open(yaml_file, 'w') as f:
    params = dict(
        model='resnet50_fpn',
        pooling_func='align',
        roi_size=14,
        mean=[122.7717, 115.9465, 102.9801],
    )
    yaml.safe_dump(params, f, default_flow_style=False)

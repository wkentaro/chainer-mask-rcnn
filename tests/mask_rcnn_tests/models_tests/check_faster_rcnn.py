#!/usr/bin/env python

from chainer import cuda
import chainercv
import matplotlib.pyplot as plt
import numpy as np

import mask_rcnn


gpu = 0
cuda.get_device_from_id(gpu).use()

label_names = chainercv.datasets.voc_bbox_label_names
label_names = np.asarray(label_names)

if False:
    dataset = chainercv.datasets.VOCDetectionDataset(
        split='train', year='2012', use_difficult=True)
    img, bbox, label = dataset.get_example(0)
else:
    dataset_ins = mask_rcnn.datasets.VOC2012InstanceSeg(split='train')
    dataset = mask_rcnn.datasets.MaskRcnnDataset(dataset_ins)
    img, bbox, label, _ = dataset.get_example(0)
    img = img.transpose(2, 0, 1)
    label -= 1

model = mask_rcnn.models.faster_rcnn.FasterRCNNVGG16(pretrained_model='voc07')
model.to_gpu()

bboxes_pred, labels_pred, _ = model.predict([img])
bbox_pred = bboxes_pred[0]
label_pred = labels_pred[0]

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print(bbox)
print(label)
print(label_names[label])
print('----------------------------------------------------------')
print(bbox_pred)
print(label_pred)
print(label_names[label_pred])
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

img = img.transpose(1, 2, 0)  # C, H, W -> H, W, C
viz = mask_rcnn.utils.draw_instance_boxes(
    img, bbox, label, n_class=20, bg_class=-1)
viz_pred = mask_rcnn.utils.draw_instance_boxes(
    img, bbox_pred, label_pred, n_class=20, bg_class=-1)
plt.subplot(121)
plt.imshow(viz)
plt.subplot(122)
plt.imshow(viz_pred)
plt.show()

import argparse

import chainer
from chainer import gradient_check
import matplotlib.pyplot as plt
import numpy as np

import mask_rcnn as mrcnn


parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=int, default=-1)
parser.add_argument('-f', '--func', choices=['align', 'pool', 'resize'],
                    default='align')
parser.add_argument('-s', '--show', action='store_true')
args = parser.parse_args()

gpu = args.gpu
if gpu >= 0:
    chainer.cuda.get_device_from_id(gpu).use()

np.set_printoptions(precision=2)

input = np.array([
    [0.88, 0.44, 0.14, 0.16, 0.37, 0.77, 0.96, 0.27],
    [0.19, 0.45, 0.57, 0.16, 0.63, 0.29, 0.71, 0.70],
    [0.66, 0.26, 0.82, 0.64, 0.54, 0.73, 0.59, 0.26],
    [0.85, 0.34, 0.76, 0.84, 0.29, 0.75, 0.62, 0.25],
    [0.32, 0.74, 0.21, 0.39, 0.34, 0.03, 0.33, 0.48],
    [0.20, 0.14, 0.16, 0.13, 0.73, 0.65, 0.96, 0.32],
    [0.19, 0.69, 0.09, 0.86, 0.88, 0.07, 0.01, 0.48],
    [0.83, 0.24, 0.97, 0.04, 0.24, 0.35, 0.50, 0.91],
], dtype=np.float32)
print('input:')
print(input)
print('-' * 79)

x = input[np.newaxis, np.newaxis, :, :]
if gpu >= 0:
    x = chainer.cuda.to_gpu(x)
x = chainer.Variable(x)
# batch_index, x1, y1, x2, y2
roiss = [
    np.array([[0, 0, 0, 2, 2]], dtype=np.float32),
    np.array([[0, 0, 0, 3, 2]], dtype=np.float32),
    np.array([[0, 0, 2, 6, 7]], dtype=np.float32),
]
for rois in roiss:
    print('rois:')
    print(rois)
    print('-' * 79)
    if gpu >= 0:
        rois = chainer.cuda.to_gpu(rois)
    rois = chainer.Variable(rois)

    if args.func == 'align':
        y = mrcnn.functions.roi_align_2d(
            x, rois, outh=2, outw=2, spatial_scale=1)
    elif args.func == 'pool':
        y = chainer.functions.roi_pooling_2d(
            x, rois, outh=2, outw=2, spatial_scale=1)
    elif args.func == 'resize':
        y = mrcnn.functions.crop_and_resize(
            x, rois, outh=2, outw=2, spatial_scale=1)

    grad = np.ones((1, 1, 2, 2), dtype=np.float32)
    if gpu >= 0:
        grad = chainer.cuda.to_gpu(grad)
    y.grad = grad
    y.backward()
    print('x.grad:')
    print(x.grad)
    print('-' * 79)
    output = y.data[0, 0]
    output = chainer.cuda.to_cpu(output)
    print('output:')
    print(output)
    print('-' * 79)

    print('check_backward:')
    gradient_check.check_backward(
        mrcnn.functions.ROIAlign2D(2, 2, 1),
        (x.data, rois.data), y.grad, no_grads=[False, True],
    )
    print('Passed!')
    print('-' * 79)

    if not args.show:
        continue

    input_viz = plt.cm.jet(input)
    input_viz = (input_viz * 255).astype(np.uint8)
    plt.subplot(121)
    plt.imshow(input_viz)
    plt.title('input')
    for j in xrange(input.shape[0]):
        for i in xrange(input.shape[1]):
            plt.text(i, j, str(input[j][i]), fontsize=8,
                    horizontalalignment='center', verticalalignment='center')

    output_viz = plt.cm.jet(output)
    output_viz = (output_viz * 255).astype(np.uint8)
    plt.subplot(122)
    plt.imshow(output_viz)
    plt.title('output')
    for j in xrange(output.shape[0]):
        for i in xrange(output.shape[1]):
            plt.text(i, j, str(output[j][i]), fontsize=8,
                    horizontalalignment='center', verticalalignment='center')

    plt.show()

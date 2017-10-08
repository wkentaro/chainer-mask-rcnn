import chainer
import chainer.functions as F
import matplotlib.pyplot as plt
import numpy as np


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
print(input)

x = input[np.newaxis, np.newaxis, :, :]
x = chainer.cuda.to_gpu(x)
x = chainer.Variable(x)
# batch_index, x1, y1, x2, y2
rois = np.array([[0, 0, 2, 6, 7]], dtype=np.float32)
rois = chainer.cuda.to_gpu(rois)
rois = chainer.Variable(rois)
y = F.roi_align_2d(x, rois, outh=2, outw=2, spatial_scale=1)
import cupy as cp
y.grad = cp.ones((1, 1, 2, 2), dtype=cp.float32)
y.backward()
print(x.grad)
output = y.data[0, 0]
output = chainer.cuda.to_cpu(output)
print(output)

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

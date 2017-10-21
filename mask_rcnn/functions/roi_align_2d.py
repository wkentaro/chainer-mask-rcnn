import collections
import six

import chainer
from chainer import cuda
from chainer import functions


def roi_align_2d(bottom_data, bottom_rois, outh, outw, spatial_scale):
    if isinstance(bottom_rois, chainer.Variable):
        bottom_rois = bottom_rois.data
    bottom_rois = cuda.to_cpu(bottom_rois)

    B, C, H, W = bottom_data.shape
    N = bottom_rois.shape[0]

    ys = collections.defaultdict(list)
    for i_roi in six.moves.range(N):
        i_batch, x1, y1, x2, y2 = bottom_rois[i_roi].tolist()
        i_batch = int(i_batch)
        x1 = int(spatial_scale * x1)
        x2 = int(spatial_scale * x2)
        y1 = int(spatial_scale * y1)
        y2 = int(spatial_scale * y2)
        x_roi = bottom_data[i_batch][:, y1:y2, x1:x2]
        x_roi = x_roi[None, :, :, :]  # N, C, H, W
        y = functions.resize_images(x_roi, (outh, outw))
        ys[i_batch].append(y)

    yss = []
    for i_batch in six.moves.range(B):
        ys = functions.vstack(ys[i_batch])
        yss.append(ys)
    yss = functions.concat(yss, axis=0)
    return yss

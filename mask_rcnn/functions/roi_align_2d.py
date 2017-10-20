import collections
import six

from chainer import functions


def roi_align_2d(inputs, outh, outw):
    bottom_data, bottom_rois = inputs
    B, C, H, W = bottom_data.shape
    N = bottom_rois.shape[0]

    ys = collections.defaultdict(list)
    for i_roi in six.moves.range(N):
        i_batch, x1, y1, x2, y2 = bottom_rois[i_roi]
        i_batch = int(i_batch.data)
        x1 = int(x1.data)
        x2 = int(x2.data)
        y1 = int(y1.data)
        y2 = int(y2.data)
        x_roi = bottom_data[i_batch][:, y1:y2, x1:x2]
        x_roi = x_roi[None, :, :, :]  # N, C, H, W
        y = functions.resize_images(x_roi, (outh, outw))
        ys[i_batch].append(y)

    yss = []
    for i_batch in six.moves.range(B):
        ys = functions.concat(ys[i_batch], axis=1)
        yss.append(ys)
    yss = functions.concat(yss, axis=0)
    return yss

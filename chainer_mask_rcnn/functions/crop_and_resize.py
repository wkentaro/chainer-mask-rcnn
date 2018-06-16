import collections

import chainer
import six


def crop_and_resize(
        x, rois, outh, outw, spatial_scale, axes='xy'
):
    """ROI feature transformation by crop-and-resize."""
    if axes not in ['xy', 'yx']:
        raise ValueError('Unsupported axes: {}'.format(axes))
    if axes == 'yx':
        rois = rois[:, [0, 2, 1, 4, 3]]

    if isinstance(rois, chainer.Variable):
        rois = rois.data
    rois = chainer.cuda.to_cpu(rois)

    B, C, H, W = x.shape
    N = rois.shape[0]

    ys = collections.defaultdict(list)
    for i_roi in six.moves.range(N):
        i_batch, x1, y1, x2, y2 = rois[i_roi].tolist()
        i_batch = int(i_batch)
        x1 = int(round(spatial_scale * x1))
        x2 = max(int(round(spatial_scale * x2)), x1 + 1)
        y1 = int(round(spatial_scale * y1))
        y2 = max(int(round(spatial_scale * y2)), y1 + 1)
        x_roi = x[i_batch][:, y1:y2, x1:x2]
        x_roi = x_roi[None, :, :, :]  # N, C, H, W
        y = chainer.functions.resize_images(x_roi, (outh, outw))
        ys[i_batch].append(y)

    yss = []
    for i_batch in six.moves.range(B):
        ys = chainer.functions.vstack(ys[i_batch])
        yss.append(ys)
    yss = chainer.functions.concat(yss, axis=0)
    return yss

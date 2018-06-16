import chainer


def roi_pooling_2d(x, rois, outh, outw, spatial_scale, axes='xy'):
    """ROI feature pooling for given rois.

    See ~chainer.functions.roi_pooling_2d.
    """
    if axes not in ['xy', 'yx']:
        raise ValueError('Unsupported axes: {}'.format(axes))
    if axes == 'yx':
        rois = rois[:, [0, 2, 1, 4, 3]]

    return chainer.functions.roi_pooling_2d(x, rois, outh, outw, spatial_scale)

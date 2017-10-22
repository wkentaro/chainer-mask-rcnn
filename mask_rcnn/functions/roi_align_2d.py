import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ROIAlign2D(function.Function):

    """ROI align over a set of 2d planes."""

    def __init__(self, outh, outw, spatial_scale):
        self.outh, self.outw = outh, outw
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, roi_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4,
            roi_type.dtype == numpy.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 5,
        )

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.outh,
                                    self.outw), dtype=numpy.float32)
        self.argmax_data = cuda.cupy.empty(top_data.shape, numpy.int32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 bottom_data, float32 spatial_scale, int32 channels,
            int32 height, int32 width, int32 pooled_height, int32 pooled_width,
            raw float32 bottom_rois
            ''',
            'float32 top_data, int32 argmax_data',
            '''
            // (n, c, ph, pw) is an element in the pooled output
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int n = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = bottom_rois[n * 5 + 0];

            if (roi_batch_ind < 0) {
                top_data = 0;
                argmax_data = 0;
                continue;
            }

            float roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
            float roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
            float roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w, 0);
            //float roi_width = fmaxf(roi_end_w - roi_start_w + 1, 0);
            float roi_height = fmaxf(roi_end_h - roi_start_h, 0);
            //float roi_height = fmaxf(roi_end_h - roi_start_h + 1, 0);
            float bin_size_h = roi_height / (pooled_height - 1);
            float bin_size_w = roi_width / (pooled_width - 1);

            float h_ = float(ph) * bin_size_h + roi_start_h;
            float w_ = float(pw) * bin_size_w + roi_start_w;
            int hstart = fminf(floor(h_), height-2);
            int wstart = fminf(floor(w_), width-2);

            if (h_<0 || h_>=height || w_<0 || w_>=width) {
                top_data = 0;
            } else {
                float  h_ratio = h_ - (float)(hstart);
                float  w_ratio = w_ - (float)(wstart);
                int offset = (roi_batch_ind * channels + c) * height * width;
                int upleft = hstart * width + wstart + offset;
                int upright = upleft + 1;
                int downleft = upleft + width;
                int downright = downleft + 1;

                top_data = bottom_data[upleft]*(1.-h_ratio)*(1.-w_ratio)
                         + bottom_data[upright]*(1.-h_ratio)*w_ratio
                         + bottom_data[downleft]*h_ratio*(1.-w_ratio)
                         + bottom_data[downright]*h_ratio*w_ratio;
            }
            ''', 'roi_align_2d_fwd'
        )(bottom_data, self.spatial_scale, channels, height, width,
          self.outh, self.outw, bottom_rois, top_data, self.argmax_data)

        return top_data,

    def backward_gpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            float32 top_diff, raw int32 argmax_data,
            int32 num_rois, float32 spatial_scale,
            int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width, raw float32 bottom_rois
            ''',
            'raw float32 bottom_diff',
            '''
            // (n, c, ph, pw) is an element in the pooled output
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int n = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = bottom_rois[n * 5 + 0];

            if (roi_batch_ind < 0) {
                bottom_diff[i] = 0;
                continue;
            }

            float roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
            float roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
            float roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w, 0);
            //float roi_width = fmaxf(roi_end_w - roi_start_w + 1, 0);
            float roi_height = fmaxf(roi_end_h - roi_start_h, 0);
            //float roi_height = fmaxf(roi_end_h - roi_start_h + 1, 0);
            float bin_size_h = roi_height / (pooled_height - 1);
            float bin_size_w = roi_width / (pooled_width - 1);

            float h_ = float(ph) * bin_size_h + roi_start_h;
            float w_ = float(pw) * bin_size_w + roi_start_w;
            int hstart = fminf(floor(h_), height-2);
            int wstart = fminf(floor(w_), width-2);

            if (h_>=0 && h_<height && w_>=0 && w_<width) {
                float  h_ratio = h_ - (float)(hstart);
                float  w_ratio = w_ - (float)(wstart);
                int offset = (roi_batch_ind * channels + c) * height * width;
                int upleft = hstart * width + wstart + offset;
                int upright = upleft + 1;
                int downleft = upleft + width;
                int downright = downleft + 1;

                atomicAdd(&bottom_diff[upleft],
                          top_diff * (1. - h_ratio) * (1. - w_ratio));
                atomicAdd(&bottom_diff[upright],
                          top_diff * (1. - h_ratio) * w_ratio);
                atomicAdd(&bottom_diff[downleft],
                          top_diff * h_ratio * (1. - w_ratio));
                atomicAdd(&bottom_diff[downright],
                          top_diff * h_ratio * w_ratio);
            }
            ''', 'roi_align_2d_bwd'
        )(gy[0], self.argmax_data, bottom_rois.shape[0],
          self.spatial_scale, channels, height, width, self.outh, self.outw,
          bottom_rois, bottom_diff)

        return bottom_diff, None


def roi_align_2d(x, rois, outh, outw, spatial_scale):
    """Spatial Region of Interest (ROI) align function.

    This function acts similarly to :class:`~functions.MaxPooling2D`, but
    it computes the maximum of input spatial patch for each channel
    with the region of interest.

    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (~chainer.Variable): Input roi variable. The shape is expected to
            be (n: data size, 5), and each datum is set as below:
            (batch_index, x_min, y_min, x_max, y_max).
        outh (int): Height of output image after pooled.
        outw (int): Width of output image after pooled.
        spatial_scale (float): Scale of the roi is resized.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing ROIAlign:
    `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_.

    """
    return ROIAlign2D(outh, outw, spatial_scale)(x, rois)

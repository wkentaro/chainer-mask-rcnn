# Original work by:
# -----------------------------------------------------------------------------
# Copyright (c) 2017 by Contributors
# \file roi_align.cu
# \brief roi align operator
# \author Yuchen Guo, Zehao Shi
# -----------------------------------------------------------------------------

# Modified work by:
# -----------------------------------------------------------------------------
# Copyright (c) 2017 Kentaro Wada <www.kentaro.wada@gmail.com>
# -----------------------------------------------------------------------------

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ROIAlignFine2D(function.Function):

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
        self.argmax_data = cuda.cupy.empty(top_data.shape, numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 bottom_data, float32 spatial_scale, int32 channels,
            int32 height, int32 width, int32 pooled_height, int32 pooled_width,
            raw float32 bottom_rois
            ''',
            'float32 top_data, float32 argmax_data',
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

            float roi_start_w = (bottom_rois[n * 5 + 1]) * spatial_scale;
            float roi_start_h = (bottom_rois[n * 5 + 2]) * spatial_scale;
            float roi_end_w = (bottom_rois[n * 5 + 3]) * spatial_scale;
            float roi_end_h = (bottom_rois[n * 5 + 4]) * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = roi_end_w - roi_start_w;
            float roi_height = roi_end_h - roi_start_h;
            float bin_size_h = static_cast<float>(roi_height)
                               / static_cast<float>(pooled_height);
            float bin_size_w = static_cast<float>(roi_width)
                               / static_cast<float>(pooled_width);

            float hstart = static_cast<float>((ph) * bin_size_h);
            float wstart = static_cast<float>((pw) * bin_size_w);
            float hend = static_cast<float>((ph + 1) * bin_size_h);
            float wend = static_cast<float>((pw + 1) * bin_size_w);

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart + roi_start_h, static_cast<float>(0)), static_cast<float>(height));
            hend = min(max(hend + roi_start_h, static_cast<float>(0)), static_cast<float>(height));
            wstart = min(max(wstart + roi_start_w, static_cast<float>(0)), static_cast<float>(width));
            wend = min(max(wend + roi_start_w, static_cast<float>(0)), static_cast<float>(width));
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Define an empty pooling region to be zero
            float maxval = is_empty ? 0 : -1E+37;
            // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
            int maxidx = -1;
            int bottom_index = 0;
            int offset = (roi_batch_ind * channels + c) * height * width;
            float h_stride = (hend - hstart)/3.0;
            float w_stride = (wend - wstart)/3.0;
            for (float h = hstart+h_stride; h <= hend-h_stride+0.01; h += max(h_stride, 0.01)) {
              for (float w = wstart+w_stride; w <= wend-w_stride+0.01; w += max(w_stride, 0.01)) {
                bottom_index ++;
                int hlow = min(max(static_cast<int>(floor(h)), 0), height-1);
                int hhigh = hlow + 1;
                int wleft = min(max(static_cast<int>(floor(w)), 0), width-1);
                int wright = wleft + 1;
                int topleft = offset + hlow * width + wleft;
                int topright = offset + hlow * width + wright;
                int bottomleft = offset + hhigh * width + wleft;
                int bottomright = offset + hhigh * width + wright;

                float alpha = (hlow == hhigh) ? static_cast<float>(0.5) : (h - hlow) / (hhigh - hlow);
                float beta = (wleft == wright) ? static_cast<float>(0.5) : (w - wleft) / (wright - wleft);
                float value = (1 - alpha) * (1 - beta) * bottom_data[topleft] + alpha * (1 - beta) * bottom_data[bottomleft]
                            + (1 - alpha) * beta * bottom_data[topright] + alpha * beta * bottom_data[bottomright];
                if (value > maxval) {
                  maxval = value;
                  maxidx = bottom_index;
                }
              }
            }
            top_data = maxval;
            argmax_data = (float)maxidx;
            ''', 'roi_align_2d_fwd'  # NOQA
        )(bottom_data, self.spatial_scale, channels, height, width,
          self.outh, self.outw, bottom_rois, top_data, self.argmax_data)

        return top_data,

    def backward_gpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 top_diff,
            raw float32 argmax_data,
            int32 num_rois, float32 spatial_scale,
            int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width, raw float32 bottom_rois
            ''',
            'float32 bottom_diff',
            '''
            // (n, c, h, w) coords in bottom data
            int w = i % width;
            int h = (i / width) % height;
            int c = (i / width / height) % channels;
            int n = i / width / height / channels;

            float gradient = 0;
            // Accumulate gradient over all ROIs that pooled this element
            for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
              int roi_batch_ind = bottom_rois[roi_n * 5 + 0];
              // Skip if ROI's batch index doesn't match n
              if (n != roi_batch_ind) {
                continue;
              }

              float roi_start_w = (bottom_rois[roi_n * 5 + 1]) * spatial_scale;
              float roi_start_h = (bottom_rois[roi_n * 5 + 2]) * spatial_scale;
              float roi_end_w = (bottom_rois[roi_n * 5 + 3]) * spatial_scale;
              float roi_end_h = (bottom_rois[roi_n * 5 + 4]) * spatial_scale;

              // Skip if ROI doesn't include (h, w)
              const bool in_roi = (w > roi_start_w - 1.0 && w < roi_end_w + 1.0 &&
                                   h > roi_start_h - 1.0 && h < roi_end_h + 1.0);
              if (!in_roi) {
                continue;
              }

              int offset = (roi_n * channels + c) * pooled_height * pooled_width;

              // Compute feasible set of pooled units that could have pooled
              // this bottom unit

              // Force malformed ROIs to be 1x1
              float roi_width = roi_end_w - roi_start_w;
              float roi_height = roi_end_h - roi_start_h;

              float bin_size_h = static_cast<float>(roi_height)
                                 / static_cast<float>(pooled_height);
              float bin_size_w = static_cast<float>(roi_width)
                                 / static_cast<float>(pooled_width);

              for (int ph = 0; ph < pooled_height; ++ph) {
                for (int pw = 0; pw < pooled_width; ++pw) {
                  float hstart = static_cast<float>((ph) * bin_size_h);
                  float wstart = static_cast<float>((pw) * bin_size_w);
                  float hend = static_cast<float>((ph + 1) * bin_size_h);
                  float wend = static_cast<float>((pw + 1) * bin_size_w);

                  hstart = min(max(hstart + roi_start_h, static_cast<float>(0)), static_cast<float>(height));
                  hend = min(max(hend + roi_start_h, static_cast<float>(0)), static_cast<float>(height));
                  wstart = min(max(wstart + roi_start_w, static_cast<float>(0)), static_cast<float>(width));
                  wend = min(max(wend + roi_start_w, static_cast<float>(0)), static_cast<float>(width));

                  bool in_bin = (w > wstart - 1.0 && w < wend + 1.0 &&
                              h > hstart - 1.0 && h < hend + 1.0);
                  if (!in_bin) {
                    continue;
                  }

                  const int pool_index = offset + ph * pooled_width + pw;
                  int bottom_index = 0;
                  float h_stride = (hend - hstart)/3.0;
                  float w_stride = (wend - wstart)/3.0;
                  for (float rh = hstart+h_stride; rh <= hend-h_stride+0.01; rh += max(h_stride, 0.01)) {
                    for (float rw = wstart+w_stride; rw <= wend-w_stride+0.01; rw += max(w_stride, 0.01)) {
                      bottom_index ++;
                      if (argmax_data[pool_index] != bottom_index) continue;
                      // compute the integer coordinates around (h, w) for bilinear interpolation
                      int hlow = min(max(static_cast<int>(floor(rh)), 0), height-1);
                      int hhigh = hlow + 1;
                      int wleft = min(max(static_cast<int>(floor(rw)), 0), width-1);
                      int wright = wleft + 1;
                      if (h != hlow && h != hhigh && w != wleft && w != wright) // (w, h) is not around (rw, rh)
                          continue;

                      float alpha = (hlow == hhigh) ? static_cast<float>(0.5) : (rh - hlow) / (hhigh - hlow);
                      float beta = (wleft == wright) ? static_cast<float>(0.5) : (rw - wleft) / (wright - wleft);
                      if (h == hlow && w == wleft) gradient += top_diff[pool_index] * (1 - alpha) * (1 - beta);
                      else if (h == hlow && w == wright) gradient += top_diff[pool_index] * (1 - alpha) * beta;
                      else if (h == hhigh && w == wleft) gradient += top_diff[pool_index] * alpha * (1 - beta);
                      else if (h == hhigh && w == wright) gradient += top_diff[pool_index] * alpha * beta;
                    }
                  }
                }
              }
            }
            bottom_diff += gradient;
            ''', 'roi_align_2d_bwd'  # NOQA
        )(gy[0], self.argmax_data, bottom_rois.shape[0],
          self.spatial_scale, channels, height, width, self.outh, self.outw,
          bottom_rois, bottom_diff)

        return bottom_diff, None


def roi_align_fine_2d(x, rois, outh, outw, spatial_scale):
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
    return ROIAlignFine2D(outh, outw, spatial_scale)(x, rois)

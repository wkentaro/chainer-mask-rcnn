import collections

import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer import functions
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

    def forward_cpu(self, inputs):
        self.retain_inputs((1,))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = numpy.empty((n_rois, channels, self.outh, self.outw),
                               dtype=numpy.float32)
        self.argmax_data_x = numpy.empty(top_data.shape, numpy.float32)
        self.argmax_data_y = numpy.empty(top_data.shape, numpy.float32)

        pooled_width, pooled_height = self.outw, self.outh
        spatial_scale = self.spatial_scale
        argmax_data_x = self.argmax_data_x
        argmax_data_y = self.argmax_data_y

        for i in six.moves.range(top_data.size):
            pw = i % pooled_width
            ph = (i / pooled_width) % pooled_height
            c = (i / pooled_width / pooled_height) % channels
            n = i / pooled_width / pooled_height / channels

            roi_batch_ind = bottom_rois[n, 0]
            if roi_batch_ind < 0:
                top_data[n, c, ph, pw] = 0
                argmax_data_x[n, c, ph, pw] = 0
                argmax_data_y[n, c, ph, pw] = 0
                continue

            roi_start_w = bottom_rois[n, 1] * spatial_scale
            roi_start_h = bottom_rois[n, 2] * spatial_scale
            roi_end_w = bottom_rois[n, 3] * spatial_scale
            roi_end_h = bottom_rois[n, 4] * spatial_scale

            roi_width = roi_end_w - roi_start_w
            roi_height = roi_end_h - roi_start_h
            bin_size_h = 1. * roi_height / pooled_height
            bin_size_w = 1. * roi_width / pooled_width

            hstart = ph * bin_size_h
            wstart = pw * bin_size_w
            hend = hstart + bin_size_h
            wend = wstart + bin_size_w

            # Add roi offsets and clip to input boundaries
            hstart = min(max(hstart + roi_start_h, 0.), float(height))
            hend = min(max(hend + roi_start_h, 0.), float(height))
            wstart = min(max(wstart + roi_start_w, 0.), float(width))
            wend = min(max(wend + roi_start_w, 0.), float(width))
            is_empty = hend <= hstart or wend <= wstart

            # number of grids for bilinear interp
            grids_h = int(numpy.ceil(bin_size_h))
            grids_w = int(numpy.ceil(bin_size_w))

            # Define an empty pooling region to be zero
            maxval = 0 if is_empty else - numpy.inf
            # If nothing is pooled, argmax = -1 causes nothing to be backprop'd
            maxidx_x = -1
            maxidx_y = -1
            offset = int((roi_batch_ind * channels + c) * height * width)
            bottom_data = bottom_data.flatten()

            margin_h = grids_h - bin_size_h
            margin_w = grids_w - bin_size_w
            for i_grid_h in six.moves.range(grids_h):
                h = hstart + margin_h / 2. + i_grid_h
                assert h < hend
                for i_grid_w in six.moves.range(grids_w):
                    w = wstart + margin_w / 2. + i_grid_w
                    assert w < wend

                    # Selecting four regular locations for bilinear interp
                    x_left = int(numpy.floor(w))
                    x_right = int(numpy.ceil(w))
                    y_bottom = int(numpy.floor(h))
                    y_top = int(numpy.ceil(h))

                    top_left_index = offset + y_top * width + x_left
                    top_right_index = offset + y_top * width + x_right
                    bottom_left_index = offset + y_bottom * width + x_left
                    bottom_right_index = offset + y_bottom * width + x_right

                    is_all_in = (0 <= x_left <= (width - 1) and
                                 0 <= x_right <= (width - 1) and
                                 0 <= y_bottom <= (height - 1) and
                                 0 <= y_top <= (height - 1))

                    if is_all_in:
                        w_ratio = w - x_left    # ratio for right
                        h_ratio = h - y_bottom  # ratio for top
                        # bilinear interpolation in x direction
                        val_bottom = (1. - w_ratio) * bottom_data[bottom_left_index] \
                            + w_ratio * bottom_data[bottom_right_index]
                        val_top = (1. - w_ratio) * bottom_data[top_left_index] \
                            + w_ratio * bottom_data[top_right_index]
                        # bilinear interpolation in y direction
                        val = (1. - h_ratio) * val_bottom + h_ratio * val_top

                    if val > maxval:
                        maxval = val
                        maxidx_x = w
                        maxidx_y = h

            top_data[n, c, ph, pw] = maxval
            argmax_data_x[n, c, ph, pw] = maxidx_x
            argmax_data_y[n, c, ph, pw] = maxidx_y

        return top_data,

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.outh,
                                    self.outw), dtype=numpy.float32)
        self.argmax_data_x = cuda.cupy.empty(top_data.shape, numpy.float32)
        self.argmax_data_y = cuda.cupy.empty(top_data.shape, numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 bottom_data, float32 spatial_scale, int32 channels,
            int32 height, int32 width, int32 pooled_height, int32 pooled_width,
            raw float32 bottom_rois
            ''',
            'float32 top_data, float32 argmax_data_x, float32 argmax_data_y',
            '''
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int n = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = bottom_rois[n * 5 + 0];

            if (roi_batch_ind < 0) {
              top_data = 0;
              argmax_data_x = 0;
              argmax_data_y = 0;
              continue;
            }

            float roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
            float roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
            float roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = roi_end_w - roi_start_w;
            float roi_height = roi_end_h - roi_start_h;
            float bin_size_h = static_cast<float>(roi_height)
                               / static_cast<float>(pooled_height);
            float bin_size_w = static_cast<float>(roi_width)
                               / static_cast<float>(pooled_width);

            float hstart = static_cast<float>(ph) * bin_size_h;
            float wstart = static_cast<float>(pw) * bin_size_w;
            float hend = static_cast<float>(ph + 1) * bin_size_h;
            float wend = static_cast<float>(pw + 1) * bin_size_w;

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart + roi_start_h, 0.), static_cast<float>(height));
            hend = min(max(hend + roi_start_h, 0.), static_cast<float>(height));
            wstart = min(max(wstart + roi_start_w, 0.), static_cast<float>(width));
            wend = min(max(wend + roi_start_w, 0.), static_cast<float>(width));
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Define an empty pooling region to be zero
            float maxval = is_empty ? 0 : -1E+37;
            // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
            float maxidx_x = -1;
            float maxidx_y = -1;
            int offset = (roi_batch_ind * channels + c) * height * width;
            for (float h = hstart; h < hend; h += 1.) {
              for (float w = wstart; w < wend; w += 1.) {
                // Selecting four regular locations for bilinear interpolation
                int x_left = floor(w);
                int x_right = x_left + 1;
                int y_bottom = floor(h);
                int y_top = y_bottom + 1;

                int top_left_index = offset + y_top * width + x_left;
                int top_right_index = offset + y_top * width + x_right;
                int bottom_left_index = offset + y_bottom * width + x_left;
                int bottom_right_index = offset + y_bottom * width + x_right;

                //Check whether 4 locations are in bounds
                bool is_top_left_in = x_left >= 0 && x_left <= width - 1
                    && y_top >= 0 && y_top <= height - 1;
                bool is_top_right_in = x_right >= 0 && x_right <= width - 1
                    && y_top >= 0 && y_top <= height - 1;
                bool is_bottom_left_in = x_left >= 0 && x_left <= width - 1
                    && y_bottom >= 0 && y_bottom <= height - 1;
                bool is_bottom_right_in = x_right >= 0 && x_right <= width - 1
                    && y_bottom >= 0 && y_bottom <= height - 1;

                //do bilinear interpolation
                float val = 0;
                if (is_bottom_left_in && is_bottom_right_in &&
                    is_top_left_in && is_top_right_in)
                {
                    float w_ratio = w - x_left;  // ratio for right
                    // bilinear interpolation in x direction
                    float val_bottom = (1 - w_ratio) * bottom_data[bottom_left_index]
                        + w_ratio * bottom_data[bottom_right_index];
                    float val_top = (1 - w_ratio) * bottom_data[top_left_index]
                        + w_ratio * bottom_data[top_right_index];
                    // bilinear interpolation in y direction
                    float h_ratio = h - y_bottom;  // ratio for top
                    val = (1 - h_ratio) * val_bottom + h_ratio * val_top;
                }

                if (val > maxval){
                  maxval = val;
                  maxidx_x = w;
                  maxidx_y = h;
                }
              }
            }
            top_data = maxval;
            argmax_data_x = maxidx_x;
            argmax_data_y = maxidx_y;
            ''', 'roi_align_2d_fwd'
        )(bottom_data, self.spatial_scale, channels, height, width,
          self.outh, self.outw, bottom_rois, top_data,
          self.argmax_data_x, self.argmax_data_y)

        return top_data,

    def backward_cpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, numpy.float32)

        num_rois = bottom_rois.shape[0]
        spatial_scale = self.spatial_scale
        pooled_height = self.outh
        pooled_width = self.outw
        argmax_data_x = self.argmax_data_x.flatten()
        argmax_data_y = self.argmax_data_y.flatten()
        top_diff = gy[0].flatten()

        for i in six.moves.range(bottom_diff.size):
            w = i % width              # x coords of input feature
            h = (i / width) % height   # y coords of input feature
            c = (i / width / height) % channels
            n = i / width / height / channels

            gradient = 0.

            # Accumulate gradient over all ROIs that pooled this element
            for roi_n in six.moves.range(num_rois):
                roi_batch_ind = bottom_rois[roi_n, 0]
                # Skip if ROI's batch index doesn't match n
                if n != roi_batch_ind:
                    continue

                # And it assumes that we don't have any negative offset of course
                roi_start_w = bottom_rois[roi_n, 1] * spatial_scale
                roi_start_h = bottom_rois[roi_n, 2] * spatial_scale
                roi_end_w = bottom_rois[roi_n, 3] * spatial_scale
                roi_end_h = bottom_rois[roi_n, 4] * spatial_scale

                # Skip if ROI doesn't include (h, w)
                in_roi = (roi_start_w <= w <= roi_end_w and
                          roi_start_h <= h <= roi_end_h)
                if not in_roi:
                    continue

                # Compute feasible set of pooled units that could have pooled
                # this bottom unit
                roi_width = roi_end_w - roi_start_w
                roi_height = roi_end_h - roi_start_h

                bin_size_h = roi_height / pooled_height
                bin_size_w = roi_width / pooled_width

                phstart = int(numpy.floor((h - roi_start_h) / bin_size_h))
                phend = int(numpy.ceil((h - roi_start_h + 1) / bin_size_h))
                pwstart = int(numpy.floor((w - roi_start_w) / bin_size_w))
                pwend = int(numpy.ceil((w - roi_start_w + 1) / bin_size_w))

                phstart = min(max(phstart, 0), pooled_height)
                phend = min(max(phend, 0), pooled_height)
                pwstart = min(max(pwstart, 0), pooled_width)
                pwend = min(max(pwend, 0), pooled_width)

                offset = (roi_n * channels + c) * pooled_height * pooled_width
                for ph in six.moves.range(phstart, phend):
                    for pw in six.moves.range(pwstart, pwend):
                        index = offset + ph * pooled_width + pw
                        maxidx_x = argmax_data_x[index]
                        maxidx_y = argmax_data_y[index]

                        x_left = int(numpy.floor(maxidx_x))
                        x_right = int(numpy.ceil(maxidx_x))
                        y_bottom = int(numpy.floor(maxidx_y))
                        y_top = int(numpy.ceil(maxidx_y))

                        is_all_in = (0 <= x_left <= (width - 1) and
                                     0 <= x_right <= (width - 1) and
                                     0 <= y_bottom <= (height - 1) and
                                     0 <= y_top <= (height - 1))

                        if is_all_in:
                            w_ratio = maxidx_x - x_left    # ratio for right
                            h_ratio = maxidx_y - y_bottom  # ratio for top
                            # bilinear interpolation in x direction
                            diff_bottom = (1. - h_ratio) * top_diff[index]
                            diff_top = h_ratio * top_diff[index]
                            diff_bottom_left = (1. - w_ratio) * diff_bottom
                            diff_bottom_right = w_ratio * diff_bottom
                            diff_top_left = (1. - w_ratio) * diff_top
                            diff_top_right = w_ratio * diff_top

                            # if (w, h) is 1 location of the 4 bilinear locations, it can get gradient
                            if w == x_left and h == y_bottom:
                                gradient += diff_bottom_left
                            elif w == x_right and h == y_bottom:
                                gradient += diff_bottom_right
                            elif w == x_left and h == y_top:
                                gradient += diff_top_left
                            elif w == x_right and h == y_top:
                                gradient += diff_top_right
                bottom_diff[n, c, h, w] = gradient

        return bottom_diff, None

    def backward_gpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 top_diff,
            raw float32 argmax_data_x, raw float32 argmax_data_y,
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

              // And it assumes that we don't have any negative offset of course
              int roi_start_w = floor(bottom_rois[roi_n * 5 + 1] * spatial_scale);
              int roi_start_h = floor(bottom_rois[roi_n * 5 + 2] * spatial_scale);
              int roi_end_w = ceil(bottom_rois[roi_n * 5 + 3] * spatial_scale);
              int roi_end_h = ceil(bottom_rois[roi_n * 5 + 4] * spatial_scale);

              // Skip if ROI doesn't include (h, w)
              const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                                   h >= roi_start_h && h <= roi_end_h);
              if (!in_roi) {
                continue;
              }

              // Compute feasible set of pooled units that could have pooled
              // this bottom unit
              float roi_width = roi_end_w - roi_start_w;
              float roi_height = roi_end_h - roi_start_h;

              float bin_size_h = static_cast<float>(roi_height)
                                 / static_cast<float>(pooled_height);
              float bin_size_w = static_cast<float>(roi_width)
                                 / static_cast<float>(pooled_width);

              int phstart = floor(static_cast<float>(h - roi_start_h) / bin_size_h);
              int phend = ceil(static_cast<float>(h - roi_start_h + 1) / bin_size_h);
              int pwstart = floor(static_cast<float>(w - roi_start_w) / bin_size_w);
              int pwend = ceil(static_cast<float>(w - roi_start_w + 1) / bin_size_w);

              phstart = min(max(phstart, 0), pooled_height);
              phend = min(max(phend, 0), pooled_height);
              pwstart = min(max(pwstart, 0), pooled_width);
              pwend = min(max(pwend, 0), pooled_width);

              int offset = (roi_n * channels + c) * pooled_height * pooled_width;
              for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                  int index = offset + ph * pooled_width + pw;
                  float max_x = argmax_data_x[index];
                  float max_y = argmax_data_y[index];

                  int x_left = floor(max_x);
                  // int x_right = ceil(max_x);
                  int x_right = x_left + 1;
                  int y_bottom = floor(max_y);
                  // int y_top = ceil(max_y);
                  int y_top = y_bottom + 1;

                  //Check whether 4 locations are in bounds
                  bool is_top_left_in = x_left >= 0 && x_left <= width - 1
                      && y_top >= 0 && y_top <= height - 1;
                  bool is_top_right_in = x_right >= 0 && x_right <= width - 1
                      && y_top >= 0 && y_top <= height - 1;
                  bool is_bottom_left_in = x_left >= 0 && x_left <= width - 1
                      && y_bottom >= 0 && y_bottom <= height - 1;
                  bool is_bottom_right_in = x_right >= 0 && x_right <= width - 1
                      && y_bottom >= 0 && y_bottom <= height - 1;

                  if (!(is_bottom_left_in && is_bottom_right_in &&
                        is_top_left_in && is_top_right_in)) {
                    continue;
                  }
                  float w_ratio = max_x - x_left;    // ratio for right
                  float h_ratio = max_y - y_bottom;  // ratio for top

                  float diff_bottom = (1. - h_ratio) * top_diff[index];
                  float diff_top = h_ratio * top_diff[index];
                  float diff_bottom_left = (1. - w_ratio) * diff_bottom;
                  float diff_bottom_right = w_ratio * diff_bottom;
                  float diff_top_left = (1. - w_ratio) * diff_top;
                  float diff_top_right = w_ratio * diff_top;

                  // if (w,h) is 1 location of the 4 bilinear locations, it can get gradient
                  if (w == x_left && h == y_bottom)
                      gradient += diff_bottom_left;
                  else if (w == x_right && h == y_bottom)
                      gradient += diff_bottom_right;
                  else if (w == x_left && h == y_top)
                      gradient += diff_top_left;
                  else if (w == x_right && h == y_top)
                      gradient += diff_top_right;
                }
              }
            }
            bottom_diff = gradient;
            ''', 'roi_align_2d_bwd'
        )(gy[0], self.argmax_data_x, self.argmax_data_y, bottom_rois.shape[0],
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


def crop_and_resize(bottom_data, bottom_rois, outh, outw, spatial_scale):
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
        x2 = max(int(spatial_scale * x2), x1 + 1)
        y1 = int(spatial_scale * y1)
        y2 = max(int(spatial_scale * y2), y1 + 1)
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

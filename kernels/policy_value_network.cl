// OpenCL kernels for the policy-value network.
//
// Activations are stored as [batch][channel][kPlane] floats, where
// kPlane = 15 * 15 board cells in row-major order. BatchNorm is already folded
// into every convolution's weight and bias by tools/export_opencl_weights.py.

#define kBoardSize 15
#define kPlane (kBoardSize * kBoardSize)

#define kActivationNone 0
#define kActivationRelu 1
#define kActivationTanh 2

// 3x3 convolution with zero padding 1, plus bias, an optional residual input
// added before ReLU, and an optional ReLU.
// One work-item computes one output value.
// Global size: (kPlane, out_channels, batch).
__kernel void Conv3x3(__global const float* input,
                      __global const float* weight,
                      __global const float* bias,
                      __global const float* residual,
                      __global float* output,
                      const int in_channels,
                      const int out_channels,
                      const int has_residual,
                      const int apply_relu) {
  const int cell = get_global_id(0);
  const int oc = get_global_id(1);
  const int b = get_global_id(2);
  const int row = cell / kBoardSize;
  const int col = cell % kBoardSize;

  __global const float* in_b = input + b * in_channels * kPlane;
  __global const float* w_oc = weight + oc * in_channels * 9;

  float sum = bias[oc];
  for (int ic = 0; ic < in_channels; ++ic) {
    __global const float* plane = in_b + ic * kPlane;
    __global const float* w = w_oc + ic * 9;
    for (int ky = 0; ky < 3; ++ky) {
      const int y = row + ky - 1;
      if (y < 0 || y >= kBoardSize) continue;
      for (int kx = 0; kx < 3; ++kx) {
        const int x = col + kx - 1;
        if (x < 0 || x >= kBoardSize) continue;
        sum += w[ky * 3 + kx] * plane[y * kBoardSize + x];
      }
    }
  }

  const int out_index = (b * out_channels + oc) * kPlane + cell;
  if (has_residual) sum += residual[out_index];
  if (apply_relu) sum = fmax(sum, 0.0f);
  output[out_index] = sum;
}

// 1x1 convolution plus bias and ReLU (used by the policy and value heads).
// Global size: (kPlane, out_channels, batch).
__kernel void Conv1x1Relu(__global const float* input,
                          __global const float* weight,
                          __global const float* bias,
                          __global float* output,
                          const int in_channels,
                          const int out_channels) {
  const int cell = get_global_id(0);
  const int oc = get_global_id(1);
  const int b = get_global_id(2);

  __global const float* in_b = input + b * in_channels * kPlane;
  __global const float* w = weight + oc * in_channels;

  float sum = bias[oc];
  for (int ic = 0; ic < in_channels; ++ic) {
    sum += w[ic] * in_b[ic * kPlane + cell];
  }
  output[(b * out_channels + oc) * kPlane + cell] = fmax(sum, 0.0f);
}

// Fully connected layer: output[b][o] = act(bias[o] + sum_i W[o][i] * in[b][i]).
// Global size: (out_features, batch).
__kernel void Linear(__global const float* input,
                     __global const float* weight,
                     __global const float* bias,
                     __global float* output,
                     const int in_features,
                     const int out_features,
                     const int activation) {
  const int o = get_global_id(0);
  const int b = get_global_id(1);

  __global const float* in_b = input + b * in_features;
  __global const float* w = weight + o * in_features;

  float sum = bias[o];
  for (int i = 0; i < in_features; ++i) sum += w[i] * in_b[i];

  if (activation == kActivationRelu) {
    sum = fmax(sum, 0.0f);
  } else if (activation == kActivationTanh) {
    sum = tanh(sum);
  }
  output[b * out_features + o] = sum;
}

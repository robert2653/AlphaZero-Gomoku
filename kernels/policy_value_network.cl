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

// Register tile of Conv3x3, set by the host when it builds the program:
// each work-item computes TILE_BATCH positions x TILE_CHANNELS output
// channels, so every input value it loads is used TILE_CHANNELS times and
// every weight TILE_BATCH times.
#ifndef TILE_BATCH
#define TILE_BATCH 1
#endif
#ifndef TILE_CHANNELS
#define TILE_CHANNELS 8
#endif

// 3x3 convolution with zero padding 1, plus bias, an optional residual input
// added before ReLU, and an optional ReLU.
// Global size: (kPlane, out_channels / TILE_CHANNELS,
//               ceil(batch / TILE_BATCH)).
__kernel void Conv3x3(__global const float* input,
                      __global const float* weight,
                      __global const float* bias,
                      __global const float* residual,
                      __global float* output,
                      const int in_channels,
                      const int out_channels,
                      const int has_residual,
                      const int apply_relu,
                      const int batch) {
  const int cell = get_global_id(0);
  const int oc0 = get_global_id(1) * TILE_CHANNELS;
  const int b0 = get_global_id(2) * TILE_BATCH;
  const int row = cell / kBoardSize;
  const int col = cell % kBoardSize;

  float sum[TILE_BATCH][TILE_CHANNELS];
  for (int t = 0; t < TILE_BATCH; ++t) {
    for (int o = 0; o < TILE_CHANNELS; ++o) sum[t][o] = bias[oc0 + o];
  }

  for (int ic = 0; ic < in_channels; ++ic) {
    for (int ky = 0; ky < 3; ++ky) {
      const int y = row + ky - 1;
      if (y < 0 || y >= kBoardSize) continue;
      for (int kx = 0; kx < 3; ++kx) {
        const int x = col + kx - 1;
        if (x < 0 || x >= kBoardSize) continue;

        // Load this input cell once per position in the tile...
        float value[TILE_BATCH];
        for (int t = 0; t < TILE_BATCH; ++t) {
          const int b = b0 + t;
          value[t] = (b < batch) ? input[(b * in_channels + ic) * kPlane +
                                         y * kBoardSize + x]
                                 : 0.0f;
        }
        // ...and each weight once per channel, then reuse both.
        for (int o = 0; o < TILE_CHANNELS; ++o) {
          const float w =
              weight[((oc0 + o) * in_channels + ic) * 9 + ky * 3 + kx];
          for (int t = 0; t < TILE_BATCH; ++t) sum[t][o] += w * value[t];
        }
      }
    }
  }

  for (int t = 0; t < TILE_BATCH; ++t) {
    const int b = b0 + t;
    if (b >= batch) break;
    for (int o = 0; o < TILE_CHANNELS; ++o) {
      const int out_index = (b * out_channels + oc0 + o) * kPlane + cell;
      float result = sum[t][o];
      if (has_residual) result += residual[out_index];
      if (apply_relu) result = fmax(result, 0.0f);
      output[out_index] = result;
    }
  }
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

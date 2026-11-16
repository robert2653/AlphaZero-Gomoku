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

// Conv3x3 tiling, set by the host when it builds the program. Each work-item
// keeps TILE_BATCH positions x TILE_CHANNELS output channels in registers, so
// every input value it reads is used TILE_CHANNELS times and every weight
// TILE_BATCH times. TILE_IC input channels are staged in local memory per
// round (see Conv3x3).
#ifndef TILE_BATCH
#define TILE_BATCH 1
#endif
#ifndef TILE_CHANNELS
#define TILE_CHANNELS 8
#endif

#ifndef TILE_IC
#define TILE_IC 4
#endif
#define kPadded (kBoardSize + 2)

// Local-memory 3x3 conv: one work-group of kPlane work-items (one per cell)
// computes TILE_BATCH positions x TILE_CHANNELS channels for the whole board.
// Each round the group stages TILE_IC zero-padded input planes per position
// and the matching weights in local memory, then every work-item reads them.
// Global size: (kPlane, out_channels / TILE_CHANNELS, ceil(batch / TILE_BATCH))
// Local size: (kPlane, 1, 1).
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
  __local float planes[TILE_BATCH][TILE_IC][kPadded * kPadded];
  __local float weights[TILE_IC][TILE_CHANNELS][9];

  const int cell = get_local_id(0);
  const int oc0 = get_group_id(1) * TILE_CHANNELS;
  const int b0 = get_group_id(2) * TILE_BATCH;
  const int row = cell / kBoardSize;
  const int col = cell % kBoardSize;
  const int padded_cell = (row + 1) * kPadded + col + 1;

  // Zero the border once; later rounds only overwrite the interior.
  __local float* flat = &planes[0][0][0];
  for (int i = cell; i < TILE_BATCH * TILE_IC * kPadded * kPadded; i += kPlane)
    flat[i] = 0.0f;

  float sum[TILE_BATCH][TILE_CHANNELS];
  for (int t = 0; t < TILE_BATCH; ++t)
    for (int o = 0; o < TILE_CHANNELS; ++o) sum[t][o] = bias[oc0 + o];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int ic0 = 0; ic0 < in_channels; ic0 += TILE_IC) {
    for (int t = 0; t < TILE_BATCH; ++t) {
      const int b = b0 + t;
      for (int i = 0; i < TILE_IC; ++i) {
        const int ic = ic0 + i;
        planes[t][i][padded_cell] =
            (b < batch && ic < in_channels)
                ? input[(b * in_channels + ic) * kPlane + cell] : 0.0f;
      }
    }
    for (int k = cell; k < TILE_IC * TILE_CHANNELS * 9; k += kPlane) {
      const int i = k / (TILE_CHANNELS * 9);
      const int o = (k / 9) % TILE_CHANNELS;
      const int j = k % 9;
      const int ic = ic0 + i;
      weights[i][o][j] = (ic < in_channels)
          ? weight[((oc0 + o) * in_channels + ic) * 9 + j] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < TILE_IC; ++i) {
      for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
          const int at = padded_cell + (ky - 1) * kPadded + (kx - 1);
          float value[TILE_BATCH];
          for (int t = 0; t < TILE_BATCH; ++t) value[t] = planes[t][i][at];
          for (int o = 0; o < TILE_CHANNELS; ++o) {
            const float w = weights[i][o][ky * 3 + kx];
            for (int t = 0; t < TILE_BATCH; ++t) sum[t][o] += w * value[t];
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
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

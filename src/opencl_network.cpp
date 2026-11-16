#include "opencl_network.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

constexpr int kPlane = kActionCount;
constexpr int kInputChannels = PolicyValueNetwork::kChannels;
constexpr int kNumResBlocks = 4;
constexpr int kActivationNone = 0;
constexpr int kActivationRelu = 1;
constexpr int kActivationTanh = 2;

// Conv3x3 register tile (see kernels/policy_value_network.cl). The program is
// built once per batch tile; RunConv3x3 picks the largest tile that the batch
// fills, since a tile wider than the batch computes empty positions.
constexpr int kTileChannels = 8;
constexpr int kTileBatches[] = {1, 2, 4, 8};
constexpr int kNumTiles = sizeof(kTileBatches) / sizeof(kTileBatches[0]);

void Check(cl_int status, const char* what) {
  if (status != CL_SUCCESS) {
    throw std::runtime_error(std::string("OpenCL error ") +
                             std::to_string(status) + " in " + what);
  }
}

struct Tensor {
  std::vector<std::uint32_t> dims;
  std::vector<float> data;
};

std::uint32_t ReadU32(std::ifstream& in) {
  std::uint32_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  return value;
}

// Reads the weight file written by tools/export_opencl_weights.py.
std::map<std::string, Tensor> LoadWeights(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open weight file '" + path + "'");

  char magic[8];
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, "AZCLWGT1", sizeof(magic)) != 0) {
    throw std::runtime_error("'" + path + "' is not an OpenCL weight file");
  }

  std::map<std::string, Tensor> tensors;
  const std::uint32_t count = ReadU32(in);
  for (std::uint32_t i = 0; i < count; ++i) {
    std::string name(ReadU32(in), '\0');
    in.read(name.data(), name.size());

    Tensor tensor;
    tensor.dims.resize(ReadU32(in));
    size_t size = 1;
    for (std::uint32_t& dim : tensor.dims) {
      dim = ReadU32(in);
      size *= dim;
    }
    tensor.data.resize(size);
    in.read(reinterpret_cast<char*>(tensor.data.data()),
            size * sizeof(float));
    tensors.emplace(std::move(name), std::move(tensor));
  }
  if (!in) throw std::runtime_error("Weight file '" + path + "' is truncated");
  return tensors;
}

std::string ReadText(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open kernel file '" + path + "'");
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

}  // namespace

struct OpenCLNetwork::Impl {
  // Weight and bias of one layer, already on the device.
  struct Layer {
    cl_mem weight = nullptr;
    cl_mem bias = nullptr;
    int in = 0;
    int out = 0;
  };

  struct ResBlock {
    Layer conv1;
    Layer conv2;
    Layer downsample;  // Only the first block changes the channel count.
    bool has_downsample = false;
  };

  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program programs[kNumTiles] = {};  // One build per kTileBatches entry.
  cl_kernel conv3x3[kNumTiles] = {};
  cl_kernel conv1x1 = nullptr;
  cl_kernel linear = nullptr;
  std::string device_name;

  ResBlock blocks[kNumResBlocks];
  Layer p_conv, p_fc, v_conv, v_fc1, v_fc2;
  int channels = 0;

  // Activation buffers, sized for `capacity` positions.
  int capacity = 0;
  cl_mem input = nullptr;
  cl_mem act_a = nullptr;
  cl_mem act_b = nullptr;
  cl_mem act_c = nullptr;
  cl_mem p_head = nullptr;
  cl_mem v_head = nullptr;
  cl_mem logits = nullptr;
  cl_mem v_hidden = nullptr;
  cl_mem value = nullptr;

  ~Impl() {
    ReleaseActivations();
    for (ResBlock& block : blocks) {
      Release(block.conv1);
      Release(block.conv2);
      Release(block.downsample);
    }
    for (Layer* layer : {&p_conv, &p_fc, &v_conv, &v_fc1, &v_fc2}) {
      Release(*layer);
    }
    if (conv1x1) clReleaseKernel(conv1x1);
    if (linear) clReleaseKernel(linear);
    for (int i = 0; i < kNumTiles; ++i) {
      if (conv3x3[i]) clReleaseKernel(conv3x3[i]);
      if (programs[i]) clReleaseProgram(programs[i]);
    }
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
  }

  static void Release(Layer& layer) {
    if (layer.weight) clReleaseMemObject(layer.weight);
    if (layer.bias) clReleaseMemObject(layer.bias);
    layer = Layer{};
  }

  void ReleaseActivations() {
    for (cl_mem* buffer : {&input, &act_a, &act_b, &act_c, &p_head, &v_head,
                           &logits, &v_hidden, &value}) {
      if (*buffer) clReleaseMemObject(*buffer);
      *buffer = nullptr;
    }
    capacity = 0;
  }

  void PickDevice(const std::string& platform_hint) {
    cl_uint num_platforms = 0;
    Check(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs");
    std::vector<cl_platform_id> platforms(num_platforms);
    Check(clGetPlatformIDs(num_platforms, platforms.data(), nullptr),
          "clGetPlatformIDs");

    // Preference: hinted GPU, then any GPU, then any device.
    cl_device_id best = nullptr;
    int best_rank = 0;
    for (cl_platform_id platform : platforms) {
      char name[256] = {};
      clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name,
                        nullptr);
      const bool hinted = std::string(name).find(platform_hint) !=
                          std::string::npos;

      cl_uint num_devices = 0;
      if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr,
                         &num_devices) != CL_SUCCESS) {
        continue;
      }
      std::vector<cl_device_id> devices(num_devices);
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices,
                     devices.data(), nullptr);
      for (cl_device_id candidate : devices) {
        cl_device_type type = 0;
        clGetDeviceInfo(candidate, CL_DEVICE_TYPE, sizeof(type), &type,
                        nullptr);
        const bool gpu = (type & CL_DEVICE_TYPE_GPU) != 0;
        const int rank = gpu ? (hinted ? 3 : 2) : 1;
        if (rank > best_rank) {
          best = candidate;
          best_rank = rank;
        }
      }
    }
    if (!best) throw std::runtime_error("No OpenCL device found");
    device = best;

    char name[256] = {};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    device_name = name;
  }

  void BuildProgram(const std::string& kernel_path) {
    cl_int status = CL_SUCCESS;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    Check(status, "clCreateContext");
    queue = clCreateCommandQueue(context, device, 0, &status);
    Check(status, "clCreateCommandQueue");

    const std::string source = ReadText(kernel_path);
    for (int i = 0; i < kNumTiles; ++i) {
      const std::string options =
          "-DTILE_BATCH=" + std::to_string(kTileBatches[i]) +
          " -DTILE_CHANNELS=" + std::to_string(kTileChannels);
      programs[i] = Compile(source, options);
      conv3x3[i] = clCreateKernel(programs[i], "Conv3x3", &status);
      Check(status, "clCreateKernel(Conv3x3)");
    }

    // The other kernels do not depend on the tile size.
    conv1x1 = clCreateKernel(programs[0], "Conv1x1Relu", &status);
    Check(status, "clCreateKernel(Conv1x1Relu)");
    linear = clCreateKernel(programs[0], "Linear", &status);
    Check(status, "clCreateKernel(Linear)");
  }

  cl_program Compile(const std::string& source, const std::string& options) {
    cl_int status = CL_SUCCESS;
    const char* text = source.c_str();
    const size_t length = source.size();
    cl_program program =
        clCreateProgramWithSource(context, 1, &text, &length, &status);
    Check(status, "clCreateProgramWithSource");

    if (clBuildProgram(program, 1, &device, options.c_str(), nullptr,
                       nullptr) != CL_SUCCESS) {
      size_t log_size = 0;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                            &log_size);
      std::string log(log_size, '\0');
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), nullptr);
      clReleaseProgram(program);
      throw std::runtime_error("OpenCL build failed (" + options + "):\n" +
                               log);
    }
    return program;
  }

  cl_mem Upload(const std::vector<float>& data) {
    cl_int status = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        data.size() * sizeof(float), const_cast<float*>(data.data()), &status);
    Check(status, "clCreateBuffer(weight)");
    return buffer;
  }

  // Uploads `name`.weight / `name`.bias. Conv weights are [out, in, k, k],
  // linear weights are [out, in].
  Layer LoadLayer(const std::map<std::string, Tensor>& tensors,
                  const std::string& name, size_t ndim) {
    const auto weight = tensors.find(name + ".weight");
    const auto bias = tensors.find(name + ".bias");
    if (weight == tensors.end() || bias == tensors.end()) {
      throw std::runtime_error("Weight file is missing layer '" + name + "'");
    }
    const std::vector<std::uint32_t>& dims = weight->second.dims;
    if (dims.size() != ndim || bias->second.dims.size() != 1 ||
        bias->second.dims[0] != dims[0]) {
      throw std::runtime_error("Layer '" + name + "' has unexpected shape");
    }

    Layer layer;
    layer.out = static_cast<int>(dims[0]);
    layer.in = static_cast<int>(dims[1]);
    layer.weight = Upload(weight->second.data);
    layer.bias = Upload(bias->second.data);
    return layer;
  }

  void LoadModel(const std::string& weights_path) {
    const std::map<std::string, Tensor> tensors = LoadWeights(weights_path);

    for (int i = 0; i < kNumResBlocks; ++i) {
      const std::string prefix = "res_layers." + std::to_string(i);
      ResBlock& block = blocks[i];
      block.conv1 = LoadLayer(tensors, prefix + ".conv1", 4);
      block.conv2 = LoadLayer(tensors, prefix + ".conv2", 4);
      block.has_downsample =
          tensors.count(prefix + ".downsample_conv.weight") > 0;
      if (block.has_downsample) {
        block.downsample = LoadLayer(tensors, prefix + ".downsample_conv", 4);
      }
    }
    p_conv = LoadLayer(tensors, "p_conv", 4);
    v_conv = LoadLayer(tensors, "v_conv", 4);
    p_fc = LoadLayer(tensors, "p_fc", 2);
    v_fc1 = LoadLayer(tensors, "v_fc1", 2);
    v_fc2 = LoadLayer(tensors, "v_fc2", 2);

    channels = blocks[0].conv1.out;
    if (blocks[0].conv1.in != kInputChannels ||
        p_fc.in != p_conv.out * kPlane || p_fc.out != kActionCount ||
        v_fc1.in != v_conv.out * kPlane || v_fc2.out != 1) {
      throw std::runtime_error("Weight file does not match the network");
    }
    for (const ResBlock& block : blocks) {
      if (block.conv1.out % kTileChannels != 0 ||
          block.conv2.out % kTileChannels != 0 ||
          (block.has_downsample && block.downsample.out % kTileChannels != 0)) {
        throw std::runtime_error(
            "Convolution channels must be a multiple of " +
            std::to_string(kTileChannels));
      }
    }
  }

  cl_mem CreateBuffer(size_t floats) {
    cl_int status = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   floats * sizeof(float), nullptr, &status);
    Check(status, "clCreateBuffer(activation)");
    return buffer;
  }

  void EnsureCapacity(int batch) {
    if (batch <= capacity) return;
    ReleaseActivations();
    const size_t n = static_cast<size_t>(batch);
    input = CreateBuffer(n * kInputChannels * kPlane);
    act_a = CreateBuffer(n * channels * kPlane);
    act_b = CreateBuffer(n * channels * kPlane);
    act_c = CreateBuffer(n * channels * kPlane);
    p_head = CreateBuffer(n * p_conv.out * kPlane);
    v_head = CreateBuffer(n * v_conv.out * kPlane);
    logits = CreateBuffer(n * kActionCount);
    v_hidden = CreateBuffer(n * v_fc1.out);
    value = CreateBuffer(n);
    capacity = batch;
  }

  void RunConv3x3(const Layer& layer, cl_mem in, cl_mem residual, cl_mem out,
                  bool relu, int batch) {
    // Largest batch tile that the batch fills (kTileBatches is ascending).
    int tile = 0;
    while (tile + 1 < kNumTiles && kTileBatches[tile + 1] <= batch) ++tile;
    cl_kernel kernel = conv3x3[tile];
    const int tile_batch = kTileBatches[tile];

    const int has_residual = residual != nullptr;
    const int apply_relu = relu;
    Check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in), "arg");
    Check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &layer.weight), "arg");
    Check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &layer.bias), "arg");
    // The kernel never reads `residual` when has_residual is 0.
    cl_mem residual_arg = residual ? residual : out;
    Check(clSetKernelArg(kernel, 3, sizeof(cl_mem), &residual_arg), "arg");
    Check(clSetKernelArg(kernel, 4, sizeof(cl_mem), &out), "arg");
    Check(clSetKernelArg(kernel, 5, sizeof(int), &layer.in), "arg");
    Check(clSetKernelArg(kernel, 6, sizeof(int), &layer.out), "arg");
    Check(clSetKernelArg(kernel, 7, sizeof(int), &has_residual), "arg");
    Check(clSetKernelArg(kernel, 8, sizeof(int), &apply_relu), "arg");
    Check(clSetKernelArg(kernel, 9, sizeof(int), &batch), "arg");
    const size_t global[3] = {
        kPlane, static_cast<size_t>(layer.out / kTileChannels),
        static_cast<size_t>((batch + tile_batch - 1) / tile_batch)};
    Check(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, nullptr,
                                 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(Conv3x3)");
  }

  void RunConv1x1(const Layer& layer, cl_mem in, cl_mem out, int batch) {
    Check(clSetKernelArg(conv1x1, 0, sizeof(cl_mem), &in), "arg");
    Check(clSetKernelArg(conv1x1, 1, sizeof(cl_mem), &layer.weight), "arg");
    Check(clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &layer.bias), "arg");
    Check(clSetKernelArg(conv1x1, 3, sizeof(cl_mem), &out), "arg");
    Check(clSetKernelArg(conv1x1, 4, sizeof(int), &layer.in), "arg");
    Check(clSetKernelArg(conv1x1, 5, sizeof(int), &layer.out), "arg");
    const size_t global[3] = {kPlane, static_cast<size_t>(layer.out),
                              static_cast<size_t>(batch)};
    Check(clEnqueueNDRangeKernel(queue, conv1x1, 3, nullptr, global, nullptr,
                                 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(Conv1x1Relu)");
  }

  void RunLinear(const Layer& layer, cl_mem in, cl_mem out, int activation,
                 int batch) {
    Check(clSetKernelArg(linear, 0, sizeof(cl_mem), &in), "arg");
    Check(clSetKernelArg(linear, 1, sizeof(cl_mem), &layer.weight), "arg");
    Check(clSetKernelArg(linear, 2, sizeof(cl_mem), &layer.bias), "arg");
    Check(clSetKernelArg(linear, 3, sizeof(cl_mem), &out), "arg");
    Check(clSetKernelArg(linear, 4, sizeof(int), &layer.in), "arg");
    Check(clSetKernelArg(linear, 5, sizeof(int), &layer.out), "arg");
    Check(clSetKernelArg(linear, 6, sizeof(int), &activation), "arg");
    const size_t global[2] = {static_cast<size_t>(layer.out),
                              static_cast<size_t>(batch)};
    Check(clEnqueueNDRangeKernel(queue, linear, 2, nullptr, global, nullptr,
                                 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel(Linear)");
  }

  std::vector<Evaluation> Forward(const std::vector<Features>& features) {
    const int batch = static_cast<int>(features.size());
    if (batch == 0) return {};
    EnsureCapacity(batch);

    // Features are contiguous std::arrays, so the batch is one flat block.
    Check(clEnqueueWriteBuffer(queue, input, CL_FALSE, 0,
                               features.size() * sizeof(Features),
                               features.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(input)");

    // Residual tower. `act_b` holds the block output.
    cl_mem x = input;
    for (const ResBlock& block : blocks) {
      RunConv3x3(block.conv1, x, nullptr, act_a, true, batch);
      cl_mem shortcut = x;
      if (block.has_downsample) {
        RunConv3x3(block.downsample, x, nullptr, act_c, false, batch);
        shortcut = act_c;
      }
      cl_mem out = (x == act_b) ? act_c : act_b;
      if (shortcut == out) out = act_b;
      RunConv3x3(block.conv2, act_a, shortcut, out, true, batch);
      x = out;
    }

    RunConv1x1(p_conv, x, p_head, batch);
    RunLinear(p_fc, p_head, logits, kActivationNone, batch);
    RunConv1x1(v_conv, x, v_head, batch);
    RunLinear(v_fc1, v_head, v_hidden, kActivationRelu, batch);
    RunLinear(v_fc2, v_hidden, value, kActivationTanh, batch);

    std::vector<float> host_logits(static_cast<size_t>(batch) * kActionCount);
    std::vector<float> host_value(batch);
    Check(clEnqueueReadBuffer(queue, logits, CL_FALSE, 0,
                              host_logits.size() * sizeof(float),
                              host_logits.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(logits)");
    Check(clEnqueueReadBuffer(queue, value, CL_TRUE, 0,
                              host_value.size() * sizeof(float),
                              host_value.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(value)");

    // Softmax over 225 logits is cheap, so it stays on the CPU.
    std::vector<Evaluation> evaluations(batch);
    for (int b = 0; b < batch; ++b) {
      const float* row = host_logits.data() + b * kActionCount;
      const float max_logit = *std::max_element(row, row + kActionCount);
      float sum = 0.f;
      for (int a = 0; a < kActionCount; ++a) {
        evaluations[b].policy[a] = std::exp(row[a] - max_logit);
        sum += evaluations[b].policy[a];
      }
      for (float& prob : evaluations[b].policy) prob /= sum;
      evaluations[b].value = host_value[b];
    }
    return evaluations;
  }
};

OpenCLNetwork::OpenCLNetwork(const std::string& weights_path,
                             const std::string& kernel_path,
                             const std::string& platform_hint)
    : impl_(std::make_unique<Impl>()) {
  impl_->PickDevice(platform_hint);
  impl_->BuildProgram(kernel_path);
  impl_->LoadModel(weights_path);
}

OpenCLNetwork::~OpenCLNetwork() = default;

Evaluation OpenCLNetwork::Evaluate(const GamePosition& position) {
  return EvaluateBatch({position}).front();
}

std::vector<Evaluation> OpenCLNetwork::EvaluateBatch(
    const std::vector<GamePosition>& positions) {
  std::vector<Features> features;
  features.reserve(positions.size());
  for (const GamePosition& position : positions) {
    features.push_back(PolicyValueNetwork::EncodeFeatures(position));
  }
  return EvaluateFeatures(features);
}

std::vector<Evaluation> OpenCLNetwork::EvaluateFeatures(
    const std::vector<Features>& features) {
  std::lock_guard<std::mutex> lock(mutex_);
  return impl_->Forward(features);
}

const std::string& OpenCLNetwork::device_name() const {
  return impl_->device_name;
}

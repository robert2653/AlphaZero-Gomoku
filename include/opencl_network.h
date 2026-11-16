#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "policy_value_network.h"

// Policy-value network evaluated with hand-written OpenCL kernels instead of
// LibTorch. Reads the weight file written by tools/export_opencl_weights.py
// and the kernel source in kernels/policy_value_network.cl.
class OpenCLNetwork final : public PolicyValueEvaluator {
 public:
  using Features = PolicyValueNetwork::Features;

  // Picks the first GPU whose platform name contains `platform_hint`
  // (e.g. "NVIDIA"), falling back to any GPU, then any device.
  OpenCLNetwork(const std::string& weights_path,
                const std::string& kernel_path,
                const std::string& platform_hint = "NVIDIA");
  ~OpenCLNetwork() override;

  Evaluation Evaluate(const GamePosition& position) override;
  std::vector<Evaluation> EvaluateBatch(
      const std::vector<GamePosition>& positions) override;

  // Evaluates already-encoded features; used by tests and benchmarks.
  std::vector<Evaluation> EvaluateFeatures(
      const std::vector<Features>& features);

  // Name of the OpenCL device in use.
  const std::string& device_name() const;

 private:
  struct Impl;

  std::mutex mutex_;
  std::unique_ptr<Impl> impl_;
};

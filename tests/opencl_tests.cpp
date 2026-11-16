// Checks the OpenCL network against PyTorch reference outputs and runs the
// parallel MCTS on top of it.
//
// Usage: opencl_tests [weights.bin] [reference.bin] [kernel.cl]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mcts.h"
#include "opencl_network.h"
#include "test_support.h"

namespace {

using Features = OpenCLNetwork::Features;

constexpr float kTolerance = 1e-4f;

struct Reference {
  std::vector<Features> features;
  std::vector<Evaluation> outputs;
};

// Reads the reference file written by tools/export_opencl_weights.py.
Reference LoadReference(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  char magic[8];
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, "AZCLREF1", sizeof(magic)) != 0) {
    throw std::runtime_error("'" + path + "' is not a reference file");
  }
  std::uint32_t n = 0;
  in.read(reinterpret_cast<char*>(&n), sizeof(n));

  Reference reference;
  reference.features.resize(n);
  reference.outputs.resize(n);
  in.read(reinterpret_cast<char*>(reference.features.data()),
          n * sizeof(Features));
  for (Evaluation& output : reference.outputs) {
    in.read(reinterpret_cast<char*>(output.policy.data()),
            sizeof(output.policy));
  }
  for (Evaluation& output : reference.outputs) {
    in.read(reinterpret_cast<char*>(&output.value), sizeof(float));
  }
  if (!in) throw std::runtime_error("Reference file is truncated");
  return reference;
}

// Largest absolute difference over every probability and value.
float MaxError(const std::vector<Evaluation>& actual,
               const std::vector<Evaluation>& expected, size_t offset = 0) {
  float error = 0.f;
  for (size_t i = 0; i < actual.size(); ++i) {
    const Evaluation& a = actual[i];
    const Evaluation& e = expected[offset + i];
    for (int k = 0; k < kActionCount; ++k) {
      error = std::max(error, std::abs(a.policy[k] - e.policy[k]));
    }
    error = std::max(error, std::abs(a.value - e.value));
  }
  return error;
}

void TestMatchesPyTorch(OpenCLNetwork& network, const Reference& reference) {
  // Small batches first so the activation buffers must grow later.
  for (size_t batch : {size_t{1}, size_t{7}, reference.features.size()}) {
    float error = 0.f;
    for (size_t start = 0; start < reference.features.size(); start += batch) {
      const size_t end = std::min(start + batch, reference.features.size());
      const std::vector<Features> chunk(reference.features.begin() + start,
                                        reference.features.begin() + end);
      error = std::max(error, MaxError(network.EvaluateFeatures(chunk),
                                       reference.outputs, start));
    }
    std::cout << "  batch " << batch << ": max error vs PyTorch = " << error
              << "\n";
    CHECK(error < kTolerance);
  }
}

void TestBatchMatchesSingle(OpenCLNetwork& network) {
  std::vector<GamePosition> positions(1);
  for (Action action : {ToAction(7, 7), ToAction(7, 8), ToAction(8, 8),
                        ToAction(6, 6), ToAction(9, 9)}) {
    positions.push_back(positions.back());
    positions.back().PlayMove(action);
  }

  const std::vector<Evaluation> batch = network.EvaluateBatch(positions);
  CHECK(batch.size() == positions.size());
  for (size_t i = 0; i < positions.size(); ++i) {
    const Evaluation single = network.Evaluate(positions[i]);
    CHECK(MaxError({single}, {batch[i]}) < 1e-5f);

    float sum = 0.f;
    for (float prob : batch[i].policy) {
      CHECK(std::isfinite(prob) && prob >= 0.f);
      sum += prob;
    }
    CHECK(std::abs(sum - 1.f) < 1e-4f);
    CHECK(std::abs(batch[i].value) <= 1.f);
  }
}

void TestParallelSearch(OpenCLNetwork& network) {
  SearchOptions options;
  options.num_playouts = 256;
  options.num_threads = 8;
  options.batch_size = 8;
  options.parallel_mode = ParallelMode::kWuPuct;
  CHECK(options.IsValid());

  GamePosition position;
  position.PlayMove(ToAction(7, 7));

  MCTS mcts(options);
  SearchTiming timing;
  const std::vector<MoveVisits> visits = mcts.Search(position, network, &timing);

  int total = 0;
  for (const MoveVisits& move : visits) total += move.visits;
  std::cout << "  search: " << timing.playouts << " playouts, "
            << timing.batches << " batches, mean batch "
            << static_cast<double>(timing.batched_positions) /
                   std::max<std::int64_t>(timing.batches, 1)
            << "\n";
  CHECK(timing.playouts == options.num_playouts);
  CHECK(total > 0 && total <= options.num_playouts);
  CHECK(mcts.CountInFlight() == 0);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string weights = argc > 1 ? argv[1] : "model/alpha-zero.bin";
  const std::string reference_path =
      argc > 2 ? argv[2] : "model/opencl-reference.bin";
  const std::string kernel =
      argc > 3 ? argv[3] : "kernels/policy_value_network.cl";

  try {
    OpenCLNetwork network(weights, kernel);
    std::cout << "device: " << network.device_name() << "\n";

    std::cout << "TestMatchesPyTorch\n";
    TestMatchesPyTorch(network, LoadReference(reference_path));
    std::cout << "TestBatchMatchesSingle\n";
    TestBatchMatchesSingle(network);
    std::cout << "TestParallelSearch\n";
    TestParallelSearch(network);
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << "\n";
    return 1;
  }
  std::cout << "All OpenCL tests passed\n";
  return 0;
}

// Times the parallel MCTS with either backend over a grid of thread counts
// and batch sizes, so LibTorch and OpenCL are measured the same way.
//
// Usage: search_bench <torch|opencl> [playouts] [repeats]
//   torch:  uses model/alpha-zero.pt on CUDA
//   opencl: uses model/alpha-zero.bin and kernels/policy_value_network.cl

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "mcts.h"
#include "opencl_network.h"
#include "policy_value_network.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Result {
  double ms = 0.0;
  double mean_batch = 0.0;
  std::int64_t collisions = 0;
};

// A short opening so the positions are not all empty boards.
GamePosition Opening() {
  GamePosition position;
  for (Action action : {ToAction(7, 7), ToAction(7, 8), ToAction(8, 7),
                        ToAction(6, 7)}) {
    position.PlayMove(action);
  }
  return position;
}

// Median wall time of `repeats` searches from a fresh tree.
Result Run(PolicyValueEvaluator& evaluator, const SearchOptions& options,
           int repeats) {
  const GamePosition position = Opening();
  std::vector<Result> runs;
  for (int r = 0; r < repeats; ++r) {
    MCTS mcts(options);
    SearchTiming timing;
    const Clock::time_point start = Clock::now();
    mcts.Search(position, evaluator, &timing);
    Result result;
    result.ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    result.mean_batch = static_cast<double>(timing.batched_positions) /
                        std::max<std::int64_t>(timing.batches, 1);
    result.collisions = timing.collisions;
    runs.push_back(result);
  }
  std::sort(runs.begin(), runs.end(),
            [](const Result& a, const Result& b) { return a.ms < b.ms; });
  return runs[runs.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: search_bench <torch|opencl> [playouts] [repeats]\n";
    return 1;
  }
  const std::string backend = argv[1];
  const int playouts = argc > 2 ? std::atoi(argv[2]) : 800;
  const int repeats = argc > 3 ? std::atoi(argv[3]) : 3;

  try {
    std::unique_ptr<PolicyValueEvaluator> evaluator;
    if (backend == "torch") {
      evaluator = std::make_unique<PolicyValueNetwork>("model/alpha-zero.pt");
    } else if (backend == "opencl") {
      evaluator = std::make_unique<OpenCLNetwork>(
          "model/alpha-zero.bin", "kernels/policy_value_network.cl");
    } else {
      std::cerr << "Unknown backend '" << backend << "'\n";
      return 1;
    }

    // Warm-up so one-time initialization is not timed.
    {
      SearchOptions warm;
      warm.num_playouts = 32;
      MCTS mcts(warm);
      mcts.Search(Opening(), *evaluator);
    }

    std::cout << "backend " << backend << ", " << playouts
              << " playouts, WU-PUCT, median of " << repeats << "\n"
              << std::setw(8) << "threads" << std::setw(7) << "batch"
              << std::setw(11) << "ms" << std::setw(13) << "playouts/s"
              << std::setw(12) << "mean batch" << std::setw(12)
              << "collisions" << std::setw(10) << "speedup\n";

    double baseline_ms = 0.0;
    for (int threads : {1, 2, 4, 8, 16, 32}) {
      std::vector<int> batches = {1};
      if (threads > 1) batches.push_back(threads);
      for (int batch : batches) {
        SearchOptions options;
        options.num_playouts = playouts;
        options.num_threads = threads;
        options.batch_size = batch;
        options.parallel_mode = ParallelMode::kWuPuct;

        const Result result = Run(*evaluator, options, repeats);
        if (threads == 1) baseline_ms = result.ms;
        std::cout << std::setw(8) << threads << std::setw(7) << batch
                  << std::setw(11) << std::fixed << std::setprecision(1)
                  << result.ms << std::setw(13) << std::setprecision(0)
                  << playouts * 1000.0 / result.ms << std::setw(12)
                  << std::setprecision(2) << result.mean_batch
                  << std::setw(12) << result.collisions << std::setw(9)
                  << baseline_ms / result.ms << "x\n";
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}

// Times the parallel MCTS with either backend over a grid of thread counts
// and batch sizes, so LibTorch and OpenCL are measured the same way.
//
// Usage: search_bench <torch|opencl|torch-cpu> [playouts] [repeats]
//   torch:     uses model/alpha-zero.pt on CUDA
//   opencl:    uses model/alpha-zero.bin and kernels/policy_value_network.cl
//   torch-cpu: uses model/alpha-zero.pt on the CPU and compares where the
//              cores go: LibTorch threads inside one inference, MCTS workers
//              that each evaluate on their own core, mixes of the two, and
//              batching on one core

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <ATen/Parallel.h>

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
    const bool cpu = backend == "torch-cpu";
    if (backend == "torch") {
      evaluator = std::make_unique<PolicyValueNetwork>("model/alpha-zero.pt");
    } else if (cpu) {
      at::set_num_threads(1);
      evaluator = std::make_unique<PolicyValueNetwork>(
          "model/alpha-zero.pt", PolicyValueNetwork::Device::kCpu);
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
              << std::setw(11) << "source" << std::setw(8) << "threads"
              << std::setw(7) << "torch"
              << std::setw(7) << "batch" << std::setw(11) << "ms"
              << std::setw(13) << "playouts/s" << std::setw(12)
              << "mean batch" << std::setw(12) << "collisions"
              << std::setw(10) << "speedup\n";

    double baseline_ms = 0.0;
    const auto report = [&](const SearchOptions& options) {
      const Result result = Run(*evaluator, options, repeats);
      if (baseline_ms == 0.0) baseline_ms = result.ms;
      std::cout << std::setw(11)
                << kBatchSourceNames[static_cast<int>(options.batch_source)]
                << std::setw(8) << options.num_threads << std::setw(7)
                << at::get_num_threads() << std::setw(7) << options.batch_size << std::setw(11) << std::fixed
                << std::setprecision(1) << result.ms << std::setw(13)
                << std::setprecision(0) << playouts * 1000.0 / result.ms
                << std::setw(12) << std::setprecision(2) << result.mean_batch
                << std::setw(12) << result.collisions << std::setw(9)
                << baseline_ms / result.ms << "x\n";
    };

    if (cpu) {
      // "torch" is the number of cores LibTorch uses inside one inference;
      // "threads" is the number of MCTS workers, each evaluating directly.
      const auto direct = [&](int workers, int torch_threads) {
        at::set_num_threads(torch_threads);
        SearchOptions options;
        options.num_playouts = playouts;
        options.num_threads = workers;
        options.parallel_mode = ParallelMode::kWuPuct;
        options.batch_source = BatchSource::kDirect;
        report(options);
      };
      // One MCTS thread, cores spent inside each inference.
      for (int torch_threads : {1, 2, 4, 8, 12, 16, 20}) direct(1, torch_threads);
      // Cores spent on concurrent playouts, one core per inference.
      for (int workers : {2, 4, 8, 12, 16, 20}) direct(workers, 1);
      // Mixes that use 12 cores in total.
      for (int workers : {2, 3, 4, 6}) direct(workers, 12 / workers);
      // Batching on one core for comparison.
      at::set_num_threads(1);
      for (int batch : {4, 8}) {
        SearchOptions options;
        options.num_playouts = playouts;
        options.batch_size = batch;
        options.parallel_mode = ParallelMode::kWuPuct;
        options.batch_source = BatchSource::kSequential;
        report(options);
      }
      return 0;
    }

    // Worker threads: each thread count without and with batching. The first
    // row (1 thread, batch 1) is the serial baseline for the speedup column.
    for (int threads : {1, 2, 4, 8, 16, 32}) {
      std::vector<int> batches = {1};
      if (threads > 1) batches.push_back(threads);
      for (int batch : batches) {
        SearchOptions options;
        options.num_playouts = playouts;
        options.num_threads = threads;
        options.batch_size = batch;
        options.parallel_mode = ParallelMode::kWuPuct;
        report(options);
      }
    }

    // Twice as many workers as the batch size: while one batch is on the
    // device, the other half of the workers can already fill the next one.
    for (int batch : {2, 4, 8, 16, 32}) {
      SearchOptions options;
      options.num_playouts = playouts;
      options.num_threads = 2 * batch;
      options.batch_size = batch;
      options.parallel_mode = ParallelMode::kWuPuct;
      report(options);
    }

    // One thread filling the same batch sizes by itself.
    for (int batch : {2, 4, 8, 16, 32}) {
      SearchOptions options;
      options.num_playouts = playouts;
      options.batch_size = batch;
      options.parallel_mode = ParallelMode::kWuPuct;
      options.batch_source = BatchSource::kSequential;
      report(options);
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}

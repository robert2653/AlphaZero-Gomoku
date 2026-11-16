// Times the OpenCL network alone and the parallel MCTS that uses it.
//
// Usage: opencl_bench [weights.bin] [kernel.cl] [playouts]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "mcts.h"
#include "opencl_network.h"

namespace {

using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

// A short opening so the positions are not all empty boards.
GamePosition Opening() {
  GamePosition position;
  for (Action action : {ToAction(7, 7), ToAction(7, 8), ToAction(8, 7),
                        ToAction(6, 7)}) {
    position.PlayMove(action);
  }
  return position;
}

void BenchInference(OpenCLNetwork& network) {
  std::cout << "== Network only ==\n"
            << std::setw(6) << "batch" << std::setw(14) << "ms/batch"
            << std::setw(16) << "positions/s\n";

  const GamePosition position = Opening();
  for (int batch : {1, 2, 4, 8, 16, 32, 64}) {
    const std::vector<GamePosition> positions(batch, position);
    network.EvaluateBatch(positions);  // Warm-up and buffer allocation.

    int runs = 0;
    const Clock::time_point start = Clock::now();
    while (runs < 5 || Milliseconds(Clock::now() - start) < 500.0) {
      network.EvaluateBatch(positions);
      ++runs;
    }
    const double ms = Milliseconds(Clock::now() - start) / runs;
    std::cout << std::setw(6) << batch << std::setw(14) << std::fixed
              << std::setprecision(3) << ms << std::setw(15)
              << std::setprecision(0) << batch * 1000.0 / ms << "\n";
  }
}

void BenchSearch(OpenCLNetwork& network, int playouts) {
  std::cout << "\n== MCTS search, " << playouts
            << " playouts, WU-PUCT, batch_size = threads ==\n"
            << std::setw(8) << "threads" << std::setw(12) << "ms"
            << std::setw(14) << "playouts/s" << std::setw(12) << "mean batch"
            << std::setw(10) << "speedup\n";

  const GamePosition position = Opening();
  double baseline_ms = 0.0;
  for (int threads : {1, 2, 4, 8, 16, 32}) {
    SearchOptions options;
    options.num_playouts = playouts;
    options.num_threads = threads;
    options.batch_size = threads;
    options.parallel_mode = ParallelMode::kWuPuct;

    // Fresh tree each time so every run does the same amount of work.
    MCTS mcts(options);
    SearchTiming timing;
    const Clock::time_point start = Clock::now();
    mcts.Search(position, network, &timing);
    const double ms = Milliseconds(Clock::now() - start);
    if (threads == 1) baseline_ms = ms;

    std::cout << std::setw(8) << threads << std::setw(12) << std::fixed
              << std::setprecision(1) << ms << std::setw(14)
              << std::setprecision(0) << timing.playouts * 1000.0 / ms
              << std::setw(12) << std::setprecision(2)
              << static_cast<double>(timing.batched_positions) /
                     std::max<std::int64_t>(timing.batches, 1)
              << std::setw(9) << baseline_ms / ms << "x\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string weights = argc > 1 ? argv[1] : "model/alpha-zero.bin";
  const std::string kernel =
      argc > 2 ? argv[2] : "kernels/policy_value_network.cl";
  const int playouts = argc > 3 ? std::atoi(argv[3]) : 800;

  try {
    OpenCLNetwork network(weights, kernel);
    std::cout << "device: " << network.device_name() << "\n\n";
    BenchInference(network);
    BenchSearch(network, playouts);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}

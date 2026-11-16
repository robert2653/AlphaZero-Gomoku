// Repeats search timings to tell small differences from noise.
//
// Runs every configuration once per round, for `rounds` rounds, so slow
// drift (clock boost, heat, background load) hits all of them alike. Prints
// mean, standard deviation and a 95% confidence interval per configuration,
// then paired comparisons: within each round, time(a) / time(b), and how many
// rounds `b` was faster. Every run is also written to a CSV file.
//
// Usage: search_stats [rounds] [playouts] [csv]
// Uses model/alpha-zero.pt on CUDA.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "mcts.h"
#include "policy_value_network.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Config {
  std::string name;
  SearchOptions options;
};

struct Stats {
  double mean = 0.0;
  double sd = 0.0;
  double half_ci = 0.0;  // 95% confidence half-width of the mean.
};

// Two-sided 95% t quantile for n - 1 degrees of freedom (n >= 2).
double TQuantile(size_t n) {
  static const double kTable[] = {12.706, 4.303, 3.182, 2.776, 2.571, 2.447,
                                  2.365,  2.306, 2.262, 2.228, 2.201, 2.179,
                                  2.160,  2.145, 2.131, 2.120, 2.110, 2.101,
                                  2.093,  2.086, 2.080, 2.074, 2.069, 2.064,
                                  2.060,  2.056, 2.052, 2.048, 2.045};
  const size_t df = n - 1;
  return df <= std::size(kTable) ? kTable[df - 1] : 1.96;
}

Stats Summarize(const std::vector<double>& values) {
  Stats stats;
  const double n = static_cast<double>(values.size());
  for (double v : values) stats.mean += v / n;
  double squares = 0.0;
  for (double v : values) squares += (v - stats.mean) * (v - stats.mean);
  stats.sd = values.size() > 1 ? std::sqrt(squares / (n - 1.0)) : 0.0;
  stats.half_ci =
      values.size() > 1 ? TQuantile(values.size()) * stats.sd / std::sqrt(n)
                        : 0.0;
  return stats;
}

Config Workers(int threads, int batch, int playouts) {
  Config config{"workers " + std::to_string(threads) + "/" +
                    std::to_string(batch),
                {}};
  config.options.num_threads = threads;
  config.options.batch_size = batch;
  config.options.num_playouts = playouts;
  config.options.parallel_mode = ParallelMode::kWuPuct;
  return config;
}

Config Sequential(int batch, int playouts) {
  Config config{"sequential 1/" + std::to_string(batch), {}};
  config.options.batch_size = batch;
  config.options.num_playouts = playouts;
  config.options.parallel_mode = ParallelMode::kWuPuct;
  config.options.batch_source = BatchSource::kSequential;
  return config;
}

GamePosition Opening() {
  GamePosition position;
  for (Action action : {ToAction(7, 7), ToAction(7, 8), ToAction(8, 7),
                        ToAction(6, 7)}) {
    position.PlayMove(action);
  }
  return position;
}

}  // namespace

int main(int argc, char** argv) {
  const int rounds = argc > 1 ? std::atoi(argv[1]) : 30;
  const int playouts = argc > 2 ? std::atoi(argv[2]) : 800;
  const std::string csv_path = argc > 3 ? argv[3] : "search_stats.csv";

  std::vector<Config> configs = {
      Workers(1, 1, playouts),  Workers(2, 1, playouts),
      Workers(4, 1, playouts),  Workers(8, 1, playouts),
      Workers(8, 8, playouts),  Sequential(8, playouts),
      Workers(16, 16, playouts), Sequential(16, playouts),
      Workers(32, 32, playouts), Sequential(32, playouts),
      Workers(32, 32, playouts),  // Patched below: long batch timeout.
  };
  // Same as workers 32/32, but the queue waits up to 100 ms for a full batch
  // instead of 1 ms, to test whether the timeout splits the batches.
  configs.back().name = "workers 32/32 t100ms";
  configs.back().options.batch_timeout_us = 100'000;

  // Paired comparisons: {a, b} reports time(a) / time(b).
  const std::vector<std::pair<int, int>> pairs = {
      {0, 1}, {0, 2}, {0, 3}, {4, 5}, {6, 7}, {8, 9}, {8, 10}, {10, 9}};

  try {
    PolicyValueNetwork network("model/alpha-zero.pt");
    const GamePosition position = Opening();

    // Warm-up so one-time initialization is not timed.
    for (const Config& config : configs) {
      MCTS mcts(config.options);
      mcts.Search(position, network);
    }

    std::ofstream csv(csv_path);
    csv << "round,config,threads,batch,source,ms,mean_batch,collisions,"
           "selection_ms,expansion_ms,evaluation_ms,inference_ms,backup_ms\n";

    std::vector<std::vector<double>> times(configs.size());
    std::vector<SearchTiming> totals(configs.size());
    for (int round = 0; round < rounds; ++round) {
      for (size_t c = 0; c < configs.size(); ++c) {
        const SearchOptions& options = configs[c].options;
        MCTS mcts(options);
        SearchTiming timing;
        const Clock::time_point start = Clock::now();
        mcts.Search(position, network, &timing);
        const double ms =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
        times[c].push_back(ms);
        totals[c] += timing;
        csv << round << ",\"" << configs[c].name << "\","
            << options.num_threads << "," << options.batch_size << ","
            << kBatchSourceNames[static_cast<int>(options.batch_source)]
            << "," << ms << ","
            << static_cast<double>(timing.batched_positions) /
                   std::max<std::int64_t>(timing.batches, 1)
            << "," << timing.collisions << "," << timing.selection_ms << ","
            << timing.expansion_ms << "," << timing.evaluation_ms << ","
            << timing.inference_ms << "," << timing.backup_ms << "\n";
      }
      std::cerr << "round " << round + 1 << "/" << rounds << "\r";
    }
    std::cerr << "\n";

    std::cout << std::fixed << "LibTorch CUDA, " << playouts
              << " playouts, WU-PUCT, " << rounds << " interleaved rounds\n\n"
              << std::left << std::setw(20) << "config" << std::right
              << std::setw(10) << "mean ms" << std::setw(9) << "sd"
              << std::setw(16) << "95% CI\n";
    for (size_t c = 0; c < configs.size(); ++c) {
      const Stats s = Summarize(times[c]);
      std::cout << std::left << std::setw(20) << configs[c].name << std::right
                << std::setprecision(1) << std::setw(10) << s.mean
                << std::setw(9) << s.sd << "   [" << s.mean - s.half_ci
                << ", " << s.mean + s.half_ci << "]\n";
    }

    // Phase times are summed over all workers, so with several threads they
    // are CPU time and can exceed the wall time. "wait" is the time workers
    // spend waiting for their evaluation; "model" is time inside the model.
    std::cout << "\nper search, mean over rounds (ms; phases summed over "
                 "workers)\n"
              << std::left << std::setw(20) << "config" << std::right
              << std::setw(8) << "batches" << std::setw(9) << "collide"
              << std::setw(9) << "select" << std::setw(9) << "expand"
              << std::setw(9) << "backup" << std::setw(10) << "wait"
              << std::setw(9) << "model\n";
    for (size_t c = 0; c < configs.size(); ++c) {
      const SearchTiming& t = totals[c];
      const double n = rounds;
      std::cout << std::left << std::setw(20) << configs[c].name << std::right
                << std::setprecision(1) << std::setw(8) << t.batches / n
                << std::setw(9) << t.collisions / n << std::setw(9)
                << t.selection_ms / n << std::setw(9) << t.expansion_ms / n
                << std::setw(9) << t.backup_ms / n << std::setw(10)
                << t.evaluation_ms / n << std::setw(9) << t.inference_ms / n
                << "\n";
    }

    std::cout << "\npaired, per round: time(a) / time(b); > 1 means b is "
                 "faster\n"
              << std::left << std::setw(42) << "a vs b" << std::right
              << std::setw(8) << "mean" << std::setw(20) << "95% CI"
              << std::setw(14) << "b faster in\n";
    for (const auto& [a, b] : pairs) {
      std::vector<double> ratios;
      int b_faster = 0;
      for (int r = 0; r < rounds; ++r) {
        ratios.push_back(times[a][r] / times[b][r]);
        if (times[b][r] < times[a][r]) ++b_faster;
      }
      const Stats s = Summarize(ratios);
      std::cout << std::left << std::setw(42)
                << configs[a].name + " vs " + configs[b].name << std::right
                << std::setprecision(3) << std::setw(8) << s.mean << "   ["
                << s.mean - s.half_ci << ", " << s.mean + s.half_ci << "]"
                << std::setw(6) << b_faster << "/" << rounds << "\n";
    }
    std::cout << "\nraw runs: " << csv_path << "\n";
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}

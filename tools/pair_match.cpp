// Plays paired AI-vs-AI games that cancel out the first-move advantage.
//
// Black wins most freestyle games between equal players, so with plain
// alternating colors the result mostly reflects who had black. Here every
// opening is played twice with the colors swapped: player A takes black in
// one game and white in the other. Openings are the first `plies` moves
// sampled from the network's policy with a fixed seed, so every matchup run
// with the same seed uses the same openings. After the opening both players
// always play their most visited move.
//
// Players come from gomoku.cfg in the working directory (player 1 = A,
// player 2 = B), as for `gomoku --simulation`.
//
// Usage: pair_match <pairs> <csv> [plies] [seed]
// Writes one CSV row per game: pair, opening moves, A's color, winner, moves.

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "ai_player.h"
#include "configuration.h"
#include "policy_value_network.h"

namespace {

// Samples `plies` moves from the network policy over legal moves. Returns
// std::nullopt if the opening already ended the game.
std::optional<std::vector<Action>> SampleOpening(PolicyValueEvaluator& network,
                                                 int plies,
                                                 std::mt19937& random) {
  GamePosition position;
  std::vector<Action> moves;
  for (int ply = 0; ply < plies; ++ply) {
    const Evaluation evaluation = network.Evaluate(position);
    const std::vector<Action> legal = position.LegalActions();
    std::vector<double> weights;
    weights.reserve(legal.size());
    for (Action action : legal) weights.push_back(evaluation.policy[action]);
    std::discrete_distribution<size_t> pick(weights.begin(), weights.end());
    const Action action = legal[pick(random)];
    position.PlayMove(action);
    moves.push_back(action);
    if (position.is_terminal()) return std::nullopt;
  }
  return moves;
}

struct GameResult {
  std::optional<Player> winner;
  int moves = 0;
};

// Plays one game from `opening` and returns the winner's color.
GameResult PlayGame(const std::vector<Action>& opening, AIPlayer& black,
                    AIPlayer& white) {
  GamePosition position;
  for (Action action : opening) position.PlayMove(action);
  black.Reset();
  white.Reset();

  while (!position.is_terminal()) {
    AIPlayer& mover =
        position.current_player() == Player::kBlack ? black : white;
    AIPlayer& other = &mover == &black ? white : black;
    const std::optional<Action> action = mover.PickMove(position);
    if (!action) break;
    position.PlayMove(*action);
    mover.ObserveMove(*action);
    other.ObserveMove(*action);
  }
  return {position.winner(), position.board().stone_count()};
}

std::string MovesText(const std::vector<Action>& moves) {
  std::string text;
  for (Action action : moves) {
    if (!text.empty()) text += ' ';
    text += std::to_string(ActionRow(action)) + ':' +
            std::to_string(ActionCol(action));
  }
  return text;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: pair_match <pairs> <csv> [plies] [seed]\n";
    return 1;
  }
  const int pairs = std::atoi(argv[1]);
  const std::string csv_path = argv[2];
  const int plies = argc > 3 ? std::atoi(argv[3]) : 6;
  const std::uint32_t seed =
      argc > 4 ? static_cast<std::uint32_t>(std::atoll(argv[4])) : 2026;

  try {
    const Configuration config;
    SearchOptions options_a = config.ai_options(AIPlayerId::kPlayer1);
    SearchOptions options_b = config.ai_options(AIPlayerId::kPlayer2);
    // The opening supplies the variety; afterwards always play the best move.
    options_a.num_sampling_moves = 0;
    options_b.num_sampling_moves = 0;

    auto network = std::make_shared<PolicyValueNetwork>("model/alpha-zero.pt");
    AIPlayer a(network, options_a);
    AIPlayer b(network, options_b);

    // Distinct openings from a fixed seed.
    std::mt19937 random(seed);
    std::vector<std::vector<Action>> openings;
    std::set<std::vector<Action>> seen;
    while (static_cast<int>(openings.size()) < pairs) {
      std::optional<std::vector<Action>> opening =
          SampleOpening(*network, plies, random);
      if (opening && seen.insert(*opening).second) {
        openings.push_back(*opening);
      }
    }

    std::ofstream csv(csv_path);
    csv << "pair,opening,a_color,winner,moves\n";
    int a_points2 = 0;  // Twice A's score, so draws stay integral.
    for (int pair = 0; pair < pairs; ++pair) {
      for (Player a_color : {Player::kBlack, Player::kWhite}) {
        const bool a_black = a_color == Player::kBlack;
        const GameResult game =
            a_black ? PlayGame(openings[pair], a, b)
                    : PlayGame(openings[pair], b, a);
        std::string winner = "draw";
        if (game.winner) winner = *game.winner == a_color ? "a" : "b";
        a_points2 += winner == "a" ? 2 : winner == "draw" ? 1 : 0;
        csv << pair << ',' << MovesText(openings[pair]) << ','
            << (a_black ? "black" : "white") << ',' << winner << ','
            << game.moves << '\n';
        csv.flush();
      }
      std::cerr << "pair " << pair + 1 << "/" << pairs << "  A score "
                << a_points2 / 2.0 << "/" << 2 * (pair + 1) << "\r";
    }
    std::cerr << "\n";
    std::cout << "A scored " << a_points2 / 2.0 << " of " << 2 * pairs
              << " games\n";
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}

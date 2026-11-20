"""Plays AI-vs-AI matches with `gomoku --simulation` and reports win rates.

Each matchup writes gomoku.cfg (player 1 = candidate, player 2 = opponent),
plays the games with colors alternating, keeps the simulation logs, and
scores player 1 as in the proposal:

    win rate = (wins + 0.5 * draws) / games

with a 95% confidence interval (normal approximation). Results go to
results/matches/<matchup>/ and results/matches/summary.csv.

Usage (from the repository root, after building):
    python3 tools/run_matches.py [games] [matchup ...]
    python3 tools/run_matches.py --paired [pairs] [matchup ...]

--paired plays the PAIRED matchups with ./build/pair_match: every opening
twice with colors swapped, scored per opening (0, 0.5 or 1 for player 1),
which cancels the first-move advantage. Results go to
results/paired/<matchup>.csv and results/paired/summary.csv.
"""

import csv
import math
import pathlib
import shutil
import subprocess
import sys
import time

GAMES = 100
SAMPLING_MOVES = 8   # Opening moves sampled by visit count, for variety.
PLAYOUTS = 800       # Fixed-simulation matches.
THINK_MS = 300       # Fixed-time matches.

SERIAL = {}
WORKERS16 = {"num_threads": 16, "batch_size": 16}
SEQUENTIAL16 = {"batch_source": "sequential", "batch_size": 16}


def playouts(**options):
    return {"search_limit": "playouts", "num_playouts": PLAYOUTS, **options}


def think_time(**options):
    return {"search_limit": "think_time", "think_time_ms": THINK_MS, **options}


# name -> (player 1 options, player 2 options)
MATCHUPS = {
    "fixed_wu16_vs_serial": (
        playouts(parallel_mode="wu_puct", **WORKERS16), playouts(**SERIAL)),
    "fixed_vl16_vs_serial": (
        playouts(parallel_mode="virtual_loss", **WORKERS16),
        playouts(**SERIAL)),
    "fixed_none16_vs_serial": (
        playouts(parallel_mode="none", **WORKERS16), playouts(**SERIAL)),
    "fixed_seq16_vs_serial": (
        playouts(parallel_mode="wu_puct", **SEQUENTIAL16), playouts(**SERIAL)),
    "time_wu16_vs_serial": (
        think_time(parallel_mode="wu_puct", **WORKERS16),
        think_time(**SERIAL)),
    "time_seq16_vs_serial": (
        think_time(parallel_mode="wu_puct", **SEQUENTIAL16),
        think_time(**SERIAL)),
    "time_wu16_vs_seq16": (
        think_time(parallel_mode="wu_puct", **WORKERS16),
        think_time(parallel_mode="wu_puct", **SEQUENTIAL16)),
}


PAIRED_THINK_MS = 100

# Fixed think time per move; player 1 is A, player 2 is B.
PAIRED = {
    # Control: identical players must score 50%.
    "serial_vs_serial": ({}, {}),
    "wu16_vs_serial": ({"parallel_mode": "wu_puct", **WORKERS16}, {}),
    "seq16_vs_serial": ({"parallel_mode": "wu_puct", **SEQUENTIAL16}, {}),
    "wu16_vs_seq16": ({"parallel_mode": "wu_puct", **WORKERS16},
                      {"parallel_mode": "wu_puct", **SEQUENTIAL16}),
}


def write_config(player1, player2):
    lines = []
    for prefix, options in (("ai_player_1_", player1),
                            ("ai_player_2_", player2)):
        merged = {"num_sampling_moves": SAMPLING_MOVES, **options}
        lines += [f"{prefix}{key} = {value}" for key, value in merged.items()]
    pathlib.Path("gomoku.cfg").write_text("\n".join(lines) + "\n")
    return lines


def score(games_csv):
    """Player 1's wins, draws, losses overall and by color."""
    totals = {"all": [0, 0, 0], "black": [0, 0, 0], "white": [0, 0, 0]}
    with open(games_csv, newline="") as f:
        for row in csv.DictReader(f):
            color = "black" if row["black_player_id"] == "1" else "white"
            winner = row["winner_player_id"]
            index = 1 if winner in ("", "0", "draw") else (
                0 if winner == "1" else 2)
            for key in ("all", color):
                totals[key][index] += 1
    return totals


def win_rate(wins, draws, losses):
    games = wins + draws + losses
    if games == 0:
        return 0.0, 0.0
    rate = (wins + 0.5 * draws) / games
    return rate, 1.96 * math.sqrt(rate * (1 - rate) / games)


def run(name, games):
    player1, player2 = MATCHUPS[name]
    config = write_config(player1, player2)
    start = time.time()
    result = subprocess.run(["./build/gomoku", "--simulation", str(games)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{name}: {result.stderr.strip()}")
    log_dir = pathlib.Path(result.stdout.strip().split("Logs: ")[-1])

    out = pathlib.Path("results/matches") / name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(log_dir, out)
    (out / "gomoku.cfg").write_text("\n".join(config) + "\n")

    totals = score(out / "games.csv")
    rate, half = win_rate(*totals["all"])
    print(f"{name}: {totals['all'][0]}W {totals['all'][1]}D "
          f"{totals['all'][2]}L  win rate {rate:.1%} +- {half:.1%}  "
          f"({time.time() - start:.0f} s)", flush=True)
    return totals


def t95(n):
    """Two-sided 95% t quantile for n - 1 degrees of freedom."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228,
             20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980}
    df = n - 1
    for limit in sorted(table):
        if df <= limit:
            return table[limit]
    return 1.96


def run_paired(name, pairs):
    player1, player2 = PAIRED[name]
    config = write_config(
        {"search_limit": "think_time", "think_time_ms": PAIRED_THINK_MS,
         **player1},
        {"search_limit": "think_time", "think_time_ms": PAIRED_THINK_MS,
         **player2})
    out = pathlib.Path("results/paired")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{name}.csv"
    (out / f"{name}.cfg").write_text("\n".join(config) + "\n")
    start = time.time()
    result = subprocess.run(["./build/pair_match", str(pairs), str(csv_path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{name}: {result.stderr.strip()}")

    points = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            value = {"a": 1.0, "draw": 0.5, "b": 0.0}[row["winner"]]
            points.setdefault(row["pair"], []).append(value)
    scores = [sum(p) / len(p) for p in points.values()]
    n = len(scores)
    mean = sum(scores) / n
    sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1))
    half = t95(n) * sd / math.sqrt(n)
    both_a = sum(s == 1.0 for s in scores)
    both_b = sum(s == 0.0 for s in scores)
    print(f"{name}: score {mean:.1%} +- {half:.1%} over {n} openings "
          f"(A won both {both_a}, split {n - both_a - both_b}, "
          f"B won both {both_b})  ({time.time() - start:.0f} s)", flush=True)
    return n, mean, half, both_a, n - both_a - both_b, both_b


def main_paired(args):
    pairs = int(args[0]) if args and args[0].isdigit() else 150
    names = [a for a in args if not a.isdigit()] or list(PAIRED)
    summary = pathlib.Path("results/paired/summary.csv")
    summary.parent.mkdir(parents=True, exist_ok=True)
    new_file = not summary.exists()
    with open(summary, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["matchup", "openings", "score", "ci95",
                             "a_won_both", "split", "b_won_both",
                             "think_ms"])
        for name in names:
            n, mean, half, both_a, split, both_b = run_paired(name, pairs)
            writer.writerow([name, n, f"{mean:.4f}", f"{half:.4f}", both_a,
                             split, both_b, PAIRED_THINK_MS])
            f.flush()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--paired":
        main_paired(sys.argv[2:])
        return
    games = int(sys.argv[1]) if len(sys.argv) > 1 else GAMES
    names = sys.argv[2:] or list(MATCHUPS)
    summary = pathlib.Path("results/matches/summary.csv")
    summary.parent.mkdir(parents=True, exist_ok=True)
    new_file = not summary.exists()
    with open(summary, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["matchup", "games", "wins", "draws", "losses",
                             "win_rate", "ci95", "black_win_rate",
                             "white_win_rate"])
        for name in names:
            totals = run(name, games)
            rate, half = win_rate(*totals["all"])
            black, _ = win_rate(*totals["black"])
            white, _ = win_rate(*totals["white"])
            writer.writerow([name, sum(totals["all"]), *totals["all"],
                             f"{rate:.4f}", f"{half:.4f}", f"{black:.4f}",
                             f"{white:.4f}"])
            f.flush()


if __name__ == "__main__":
    main()

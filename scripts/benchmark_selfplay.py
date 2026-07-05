#!/usr/bin/env python3
"""Benchmark self-play wall time and MCTS phase breakdown."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from nnmcts.cli_utils import default_device
from play_matches import run_matches


def build_parser():
  parser = argparse.ArgumentParser(
    description="Benchmark self-play throughput and MCTS phase timings.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--game-type", choices=("TTT", "UTTT"), default="UTTT")
  parser.add_argument("--num-games", type=int, default=5)
  parser.add_argument("--device", default=default_device())
  parser.add_argument("--workers", type=int, default=1)
  parser.add_argument("--output", help="Optional JSON path for benchmark results.")

  for player_idx in (1, 2):
    parser.add_argument(f"--player{player_idx}-type", choices=("random", "mcts", "nmcts"), default="nmcts")
    parser.add_argument(f"--player{player_idx}-iters", type=int, default=50)
    parser.add_argument(f"--player{player_idx}-model", help="Checkpoint path for NMCTS players.")

  return parser


def main():
  args = build_parser().parse_args()

  total_start = perf_counter()
  summary = run_matches(
    game_type=args.game_type,
    num_games=args.num_games,
    player_one_type=args.player1_type,
    player_one_iters=args.player1_iters,
    player_one_model=args.player1_model,
    player_two_type=args.player2_type,
    player_two_iters=args.player2_iters,
    player_two_model=args.player2_model,
    device=args.device,
    workers=args.workers,
    collect_benchmark=True,
  )
  total_wall_time = perf_counter() - total_start

  benchmark = summary.get("benchmark", {})
  report = {
    "game_type": args.game_type,
    "num_games": args.num_games,
    "workers": args.workers,
    "device": args.device,
    "player1_type": args.player1_type,
    "player2_type": args.player2_type,
    "player1_iters": args.player1_iters,
    "player2_iters": args.player2_iters,
    "total_wall_time": total_wall_time,
    "avg_game_wall_time": benchmark.get("avg_game_wall_time", 0.0),
    "per_game_wall_time": benchmark.get("per_game_wall_time", []),
    "mcts_phase_avg": benchmark.get("mcts_phase_avg"),
  }

  print(json.dumps(report, indent=2))

  if args.output:
    with open(args.output, "w", encoding="utf-8") as handle:
      json.dump(report, handle, indent=2)
    print(f"Wrote benchmark report to {args.output}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())

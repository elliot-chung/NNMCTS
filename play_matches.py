import argparse
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

from tqdm import tqdm

from nnmcts.arena.Arena import Arena
from nnmcts.cli_utils import (
  create_environment,
  create_player,
  format_ratio,
  default_device,
  save_records_file,
  summarize_results,
)
from nnmcts.selfplay.worker import play_game_worker


def build_parser():
  parser = argparse.ArgumentParser(
    description="Pit two players against each other and optionally record the resulting dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--game-type", choices=("TTT", "UTTT"), required=True, help="Type of game to play.")
  parser.add_argument("--num-games", type=int, required=True, help="Number of games to play.")
  parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device(), help="Default device when --play-device is omitted.")
  parser.add_argument("--play-device", choices=("cpu", "cuda"), help="Device for self-play, including NMCTS inference.")
  parser.add_argument("--workers", type=int, default=1, help="Parallel self-play worker processes.")
  parser.add_argument("--record-output", help="Optional path to save recorded game data.")
  parser.add_argument("--show-mcts-timing", action="store_true", help="Print MCTS phase breakdown per move.")

  for player_idx in (1, 2):
    parser.add_argument(f"--player{player_idx}-type", choices=("random", "mcts", "nmcts"), required=True, help="Type of player to use.")
    parser.add_argument(f"--player{player_idx}-iters", type=int, default=100, help="Number of iterations for the player. (Does nothing for random players.)")
    parser.add_argument(f"--player{player_idx}-model", help="Checkpoint path for NMCTS players.")

  return parser


def _build_worker_args(
  game_index: int,
  game_type: str,
  player_one_type: str,
  player_one_iters: int,
  player_one_model: str | None,
  player_two_type: str,
  player_two_iters: int,
  player_two_model: str | None,
  device: str,
  record: bool,
  show_mcts_timing: bool,
  collect_mcts_timing: bool,
) -> dict:
  return {
    "game_index": game_index,
    "game_type": game_type,
    "player_one_type": player_one_type,
    "player_one_iters": player_one_iters,
    "player_one_model": player_one_model,
    "player_two_type": player_two_type,
    "player_two_iters": player_two_iters,
    "player_two_model": player_two_model,
    "device": device,
    "record": record,
    "show_mcts_timing": show_mcts_timing,
    "collect_mcts_timing": collect_mcts_timing,
  }


def run_matches(
  game_type: str,
  num_games: int,
  player_one_type: str,
  player_one_iters: int,
  player_one_model: str | None,
  player_two_type: str,
  player_two_iters: int,
  player_two_model: str | None,
  play_device: str | None = None,
  device: str | None = None,
  record_output: str | None = None,
  workers: int = 1,
  show_mcts_timing: bool = False,
  collect_benchmark: bool = False,
):
  play_device = play_device or device or default_device()
  workers = max(1, workers)
  results = Counter()
  records = []
  game_timings = []
  mcts_timings = []
  record_games = record_output is not None

  if workers == 1:
    player_one = None
    player_two = None
    match_iterator = tqdm(range(num_games), desc="Playing matches", unit="game", ascii=True)
    for _ in match_iterator:
      environment = create_environment(game_type)
      if player_one is None:
        player_one = create_player(
          environment,
          game_type,
          player_one_type,
          True,
          player_one_iters,
          player_one_model,
          play_device,
          "player_one",
          "--player1-model",
          show_mcts_timing=show_mcts_timing,
        )
        player_two = create_player(
          environment,
          game_type,
          player_two_type,
          False,
          player_two_iters,
          player_two_model,
          play_device,
          "player_two",
          "--player2-model",
          show_mcts_timing=show_mcts_timing,
        )
      else:
        player_one.environment = environment
        player_two.environment = environment

      game_start = perf_counter()
      arena = Arena(environment, player_one, player_two)
      if record_games:
        winner, record = arena.play_game(record=True)
        records.append(record)
      else:
        winner = arena.play_game(record=False)
      results[winner] += 1
      if collect_benchmark:
        game_timings.append(perf_counter() - game_start)

      summary = summarize_results(results, sum(results.values()))
      match_iterator.set_postfix({
        "p1": summary["player_one_wins"],
        "draw": summary["draws"],
        "p2": summary["player_two_wins"],
      })
  else:
    worker_args = [
      _build_worker_args(
        game_index=index,
        game_type=game_type,
        player_one_type=player_one_type,
        player_one_iters=player_one_iters,
        player_one_model=player_one_model,
        player_two_type=player_two_type,
        player_two_iters=player_two_iters,
        player_two_model=player_two_model,
        device=play_device,
        record=record_games,
        show_mcts_timing=show_mcts_timing,
        collect_mcts_timing=collect_benchmark,
      )
      for index in range(num_games)
    ]

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
      futures = [executor.submit(play_game_worker, args) for args in worker_args]
      match_iterator = tqdm(as_completed(futures), total=num_games, desc="Playing matches", unit="game", ascii=True)
      for future in match_iterator:
        game_result = future.result()
        results[game_result["winner"]] += 1
        if game_result["record"] is not None:
          records.append(game_result["record"])
        if collect_benchmark:
          game_timings.append(game_result["wall_time"])
          if game_result.get("mcts_timing"):
            mcts_timings.append(game_result["mcts_timing"])

        summary = summarize_results(results, sum(results.values()))
        match_iterator.set_postfix({
          "p1": summary["player_one_wins"],
          "draw": summary["draws"],
          "p2": summary["player_two_wins"],
        })

  summary = summarize_results(results, num_games)
  if record_output:
    metadata = {
      "num_games": num_games,
      "workers": workers,
      "player_one": {
        "type": player_one_type,
        "iters": player_one_iters,
        "model": player_one_model,
      },
      "player_two": {
        "type": player_two_type,
        "iters": player_two_iters,
        "model": player_two_model,
      },
    }
    save_records_file(record_output, game_type, records, metadata)

  if collect_benchmark:
    summary["benchmark"] = _summarize_benchmark(game_timings, mcts_timings)

  return summary


def _summarize_benchmark(game_timings: list[float], mcts_timings: list[dict[str, float]]) -> dict:
  benchmark = {
    "per_game_wall_time": game_timings,
    "total_wall_time": sum(game_timings),
    "avg_game_wall_time": sum(game_timings) / len(game_timings) if game_timings else 0.0,
  }

  if mcts_timings:
    keys = mcts_timings[0].keys()
    benchmark["mcts_phase_avg"] = {
      key: sum(timing[key] for timing in mcts_timings) / len(mcts_timings)
      for key in keys
    }

  return benchmark


def main():
  args = build_parser().parse_args()
  play_device = args.play_device or args.device
  summary = run_matches(
    game_type=args.game_type,
    num_games=args.num_games,
    player_one_type=args.player1_type,
    player_one_iters=args.player1_iters,
    player_one_model=args.player1_model,
    player_two_type=args.player2_type,
    player_two_iters=args.player2_iters,
    player_two_model=args.player2_model,
    play_device=play_device,
    record_output=args.record_output,
    workers=args.workers,
    show_mcts_timing=args.show_mcts_timing,
  )

  print(f"Player One wins: {summary['player_one_wins']} ({format_ratio(summary['player_one_win_rate'])})")
  print(f"Draws: {summary['draws']} ({format_ratio(summary['draw_rate'])})")
  print(f"Player Two wins: {summary['player_two_wins']} ({format_ratio(summary['player_two_win_rate'])})")
  if args.record_output:
    print(f"Recorded dataset written to: {args.record_output}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())

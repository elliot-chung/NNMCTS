import argparse
from collections import Counter

from tqdm import tqdm

from nnmcts.arena.Arena import Arena
from nnmcts.cli_utils import (
  create_environment,
  create_player,
  format_ratio,
  save_records_file,
  summarize_results,
)


def build_parser():
  parser = argparse.ArgumentParser(
    description="Pit two players against each other and optionally record the resulting dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--game-type", choices=("TTT", "UTTT"), required=True)
  parser.add_argument("--num-games", type=int, required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--record-output", help="Optional path to save recorded game data.")

  for player_idx in (1, 2):
    parser.add_argument(f"--player{player_idx}-type", choices=("random", "mcts", "nmcts"), required=True)
    parser.add_argument(f"--player{player_idx}-iters", type=int, default=100)
    parser.add_argument(f"--player{player_idx}-model", help="Checkpoint path for NMCTS players.")

  return parser


def run_matches(
  game_type: str,
  num_games: int,
  player_one_type: str,
  player_one_iters: int,
  player_one_model: str | None,
  player_two_type: str,
  player_two_iters: int,
  player_two_model: str | None,
  device: str,
  record_output: str | None = None,
):
  results = Counter()
  records = []

  match_iterator = tqdm(range(num_games), desc="Playing matches", unit="game")
  for _ in match_iterator:
    environment = create_environment(game_type)
    player_one = create_player(
      environment,
      game_type,
      player_one_type,
      True,
      player_one_iters,
      player_one_model,
      device,
      "player_one",
      "--player1-model",
    )
    player_two = create_player(
      environment,
      game_type,
      player_two_type,
      False,
      player_two_iters,
      player_two_model,
      device,
      "player_two",
      "--player2-model",
    )

    arena = Arena(environment, player_one, player_two)
    if record_output:
      winner, record = arena.play_game(record=True)
      records.append(record)
    else:
      winner = arena.play_game(record=False)
    results[winner] += 1

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

  return summary


def main():
  args = build_parser().parse_args()
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
    record_output=args.record_output,
  )

  print(f"Player One wins: {summary['player_one_wins']} ({format_ratio(summary['player_one_win_rate'])})")
  print(f"Draws: {summary['draws']} ({format_ratio(summary['draw_rate'])})")
  print(f"Player Two wins: {summary['player_two_wins']} ({format_ratio(summary['player_two_win_rate'])})")
  if args.record_output:
    print(f"Recorded dataset written to: {args.record_output}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())

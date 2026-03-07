import argparse
from pathlib import Path

from nnmcts.cli_utils import load_records_file, save_records_file
from play_matches import run_matches
from train_model import run_training


def build_parser():
  parser = argparse.ArgumentParser(
    description="Alternate between match generation and model training.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--game-type", choices=("TTT", "UTTT"), required=True)
  parser.add_argument("--rounds", type=int, required=True)
  parser.add_argument("--games-per-round", type=int, required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--initial-checkpoint", help="Optional starting checkpoint for training and NMCTS players.")
  parser.add_argument("--accumulate-records", action="store_true")

  for player_idx in (1, 2):
    parser.add_argument(f"--player{player_idx}-type", choices=("random", "mcts", "nmcts"), required=True)
    parser.add_argument(f"--player{player_idx}-iters", type=int, default=100)
    parser.add_argument(
      f"--player{player_idx}-model",
      help="Optional fixed checkpoint for this player. If omitted for NMCTS, the latest trained checkpoint is used.",
    )

  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=0.0)
  parser.add_argument("--value-loss-weight", type=float, default=0.1)
  parser.add_argument("--policy-loss-weight", type=float, default=0.9)
  parser.add_argument("--val-split", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--log-every", type=int, default=10)
  parser.add_argument("--augment-train", action="store_true")
  parser.add_argument("--augment-val", action="store_true")
  parser.add_argument("--deduplicate-train", action="store_true")
  parser.add_argument("--deduplicate-val", action="store_true")
  return parser


def resolve_round_player(player_type: str, static_model: str | None, latest_checkpoint: str | None):
  if player_type != "nmcts":
    return player_type, None
  if static_model is not None:
    return "nmcts", static_model
  if latest_checkpoint is not None:
    return "nmcts", latest_checkpoint
  return "random", None


def main():
  args = build_parser().parse_args()

  output_dir = Path(args.output_dir)
  datasets_dir = output_dir / "datasets"
  checkpoints_dir = output_dir / "checkpoints"
  datasets_dir.mkdir(parents=True, exist_ok=True)
  checkpoints_dir.mkdir(parents=True, exist_ok=True)

  latest_checkpoint = args.initial_checkpoint
  cumulative_records = []

  for round_idx in range(1, args.rounds + 1):
    print(f"=== Round {round_idx}/{args.rounds}: Match Generation ===")
    round_dataset_path = datasets_dir / f"round_{round_idx:03d}.pkl"

    player_one_type, player_one_model = resolve_round_player(args.player1_type, args.player1_model, latest_checkpoint)
    player_two_type, player_two_model = resolve_round_player(args.player2_type, args.player2_model, latest_checkpoint)

    if round_idx == 1:
      bootstrap_players = []
      if args.player1_type == "nmcts" and player_one_type == "random":
        bootstrap_players.append("player1")
      if args.player2_type == "nmcts" and player_two_type == "random":
        bootstrap_players.append("player2")
      if bootstrap_players:
        joined = ", ".join(bootstrap_players)
        print(
          f"Bootstrapping {joined} with random play in round 1 because no checkpoint was provided. "
          "Later rounds will use the newly trained model."
        )

    summary = run_matches(
      game_type=args.game_type,
      num_games=args.games_per_round,
      player_one_type=player_one_type,
      player_one_iters=args.player1_iters,
      player_one_model=player_one_model,
      player_two_type=player_two_type,
      player_two_iters=args.player2_iters,
      player_two_model=player_two_model,
      device=args.device,
      record_output=str(round_dataset_path),
    )

    print(
      f"Round {round_idx} results: "
      f"P1 {summary['player_one_wins']} | Draw {summary['draws']} | P2 {summary['player_two_wins']}"
    )

    training_dataset_path = round_dataset_path
    if args.accumulate_records:
      round_payload = load_records_file(round_dataset_path)
      cumulative_records.extend(round_payload["records"])
      training_dataset_path = datasets_dir / f"round_{round_idx:03d}_cumulative.pkl"
      save_records_file(
        training_dataset_path,
        args.game_type,
        cumulative_records,
        {"source_rounds": round_idx},
      )

    print(f"=== Round {round_idx}/{args.rounds}: Training ===")
    checkpoint_output = checkpoints_dir / f"round_{round_idx:03d}.pt"
    train_result = run_training(
      game_type=args.game_type,
      dataset_path=str(training_dataset_path),
      output_model=str(checkpoint_output),
      checkpoint_path=latest_checkpoint,
      device=args.device,
      epochs=args.epochs,
      batch_size=args.batch_size,
      learning_rate=args.learning_rate,
      weight_decay=args.weight_decay,
      value_loss_weight=args.value_loss_weight,
      policy_loss_weight=args.policy_loss_weight,
      val_split=args.val_split,
      seed=args.seed + round_idx - 1,
      log_every=args.log_every,
      augment_train=args.augment_train,
      augment_val=args.augment_val,
      deduplicate_train=args.deduplicate_train,
      deduplicate_val=args.deduplicate_val,
    )

    latest_checkpoint = str(checkpoint_output)
    print(
      f"Completed round {round_idx}: "
      f"train_loss={train_result['final_train_loss']:.4f} "
      f"val_loss={train_result['final_val_loss'] if train_result['final_val_loss'] is not None else 'n/a'} "
      f"checkpoint={latest_checkpoint}"
    )

  print(f"Pipeline complete. Latest checkpoint: {latest_checkpoint}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

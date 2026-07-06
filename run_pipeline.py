import argparse
import re
import warnings
from pathlib import Path

from tqdm import tqdm

from nnmcts.cli_utils import load_records_file, save_records_file, default_device, add_non_interactive_logging_arg, tqdm_kwargs
from play_matches import run_matches
from train_model import run_training


def build_parser():
  parser = argparse.ArgumentParser(
    description="Alternate between match generation and model training.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--game-type", choices=("TTT", "UTTT"), required=True, help="Type of game to play.")
  parser.add_argument("--rounds", type=int, required=True, help="Number of rounds to run the pipeline.")
  parser.add_argument("--games-per-round", type=int, required=True, help="Number of games to play in each round.")
  parser.add_argument("--output-dir", required=True, help="Directory to save the output.")
  parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device(), help="Default device when --play-device or --train-device are omitted.")
  parser.add_argument("--play-device", choices=("cpu", "cuda"), help="Device for self-play, including NMCTS inference. Defaults to cpu when --train-device is cuda, otherwise --device.")
  parser.add_argument("--train-device", choices=("cpu", "cuda"), help="Device for model training. Defaults to --device.")
  parser.add_argument("--initial-checkpoint", help="Optional starting checkpoint for training and NMCTS players.")
  parser.add_argument(
    "--start-round",
    type=int,
    help="First pipeline round number. Defaults to one past the round encoded in --initial-checkpoint, or 1.",
  )
  parser.add_argument("--accumulate-records", action="store_true", help="Accumulate records from each round into a single dataset.")
  parser.add_argument(
    "--num-eval-games",
    type=int,
    default=0,
    help="Head-to-head eval games after each training round. 0 disables eval gating.",
  )
  parser.add_argument(
    "--winrate-threshold",
    type=float,
    default=0.55,
    help="Candidate must exceed this win rate vs the champion to be promoted.",
  )

  for player_idx in (1, 2):
    parser.add_argument(f"--player{player_idx}-type", choices=("random", "mcts", "nmcts"), required=True, help="Type of player to use.")
    parser.add_argument(f"--player{player_idx}-iters", type=int, default=100, help="Number of iterations for the player. (Does nothing for random players.)")
    parser.add_argument(
      f"--player{player_idx}-model",
      help="Optional fixed checkpoint for this player. If omitted for NMCTS, the latest trained checkpoint is used.",
    )

  parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
  parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
  parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for training.")
  parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for training.")
  parser.add_argument("--value-loss-weight", type=float, default=0.1, help="Weight for value loss in training.")
  parser.add_argument("--policy-loss-weight", type=float, default=0.9, help="Weight for policy loss in training.")
  parser.add_argument("--val-split", type=float, default=0.2, help="Validation split for training.")
  parser.add_argument("--seed", type=int, default=0, help="Random seed for training.")
  parser.add_argument("--augment-train", action="store_true", help="Augment training dataset.")
  parser.add_argument("--augment-val", action="store_true", help="Augment validation dataset.")
  parser.add_argument("--deduplicate-train", action="store_true", help="Deduplicate training dataset.")
  parser.add_argument("--deduplicate-val", action="store_true", help="Deduplicate validation dataset.")
  parser.add_argument("--self-play-workers", type=int, default=1, help="Parallel self-play worker processes.")
  parser.add_argument("--show-mcts-timing", action="store_true", help="Print MCTS phase breakdown per move.")
  parser.add_argument("--amp", action="store_true", help="Enable mixed-precision training on CUDA.")
  add_non_interactive_logging_arg(parser)
  return parser


def eval_enabled(args) -> bool:
  return args.num_eval_games > 0


def has_dynamic_nmcts_player(args) -> bool:
  for player_idx in (1, 2):
    if getattr(args, f"player{player_idx}_type") == "nmcts" and getattr(args, f"player{player_idx}_model") is None:
      return True
  return False


def validate_eval_args(args) -> None:
  if args.num_eval_games < 0:
    raise ValueError("--num-eval-games must be >= 0")
  if not (0.0 < args.winrate_threshold < 1.0):
    raise ValueError("--winrate-threshold must be in the range (0.0, 1.0)")
  if not eval_enabled(args):
    return
  if not has_dynamic_nmcts_player(args):
    raise ValueError(
      "Evaluation gating requires at least one dynamic NMCTS player "
      "(nmcts without --playerN-model)."
    )
  if args.accumulate_records:
    warnings.warn(
      "--accumulate-records is ignored when evaluation gating is enabled; "
      "champion-streak accumulation is used instead.",
      stacklevel=2,
    )


def should_promote(candidate_win_rate: float, threshold: float) -> bool:
  return candidate_win_rate > threshold


def aggregate_candidate_wins(
  candidate_as_player_one: dict,
  candidate_as_player_two: dict,
) -> tuple[int, float, int]:
  candidate_wins = candidate_as_player_one["player_one_wins"] + candidate_as_player_two["player_two_wins"]
  num_games = (
    candidate_as_player_one["player_one_wins"]
    + candidate_as_player_one["player_two_wins"]
    + candidate_as_player_one["draws"]
    + candidate_as_player_two["player_one_wins"]
    + candidate_as_player_two["player_two_wins"]
    + candidate_as_player_two["draws"]
  )
  candidate_win_rate = candidate_wins / num_games if num_games else 0.0
  return candidate_wins, candidate_win_rate, num_games


def run_candidate_eval(
  game_type: str,
  num_games: int,
  candidate_ckpt: str,
  champion_ckpt: str,
  iters: int,
  play_device: str,
  workers: int,
  non_interactive_logging: bool,
) -> dict:
  half = num_games // 2
  second_half = num_games - half

  candidate_as_p1 = run_matches(
    game_type=game_type,
    num_games=half,
    player_one_type="nmcts",
    player_one_iters=iters,
    player_one_model=candidate_ckpt,
    player_two_type="nmcts",
    player_two_iters=iters,
    player_two_model=champion_ckpt,
    play_device=play_device,
    record_output=None,
    workers=workers,
    non_interactive_logging=non_interactive_logging,
  )
  candidate_as_p2 = run_matches(
    game_type=game_type,
    num_games=second_half,
    player_one_type="nmcts",
    player_one_iters=iters,
    player_one_model=champion_ckpt,
    player_two_type="nmcts",
    player_two_iters=iters,
    player_two_model=candidate_ckpt,
    play_device=play_device,
    record_output=None,
    workers=workers,
    non_interactive_logging=non_interactive_logging,
  )

  candidate_wins, candidate_win_rate, _ = aggregate_candidate_wins(candidate_as_p1, candidate_as_p2)
  return {
    "candidate_wins": candidate_wins,
    "candidate_win_rate": candidate_win_rate,
    "num_games": num_games,
    "candidate_as_player_one": candidate_as_p1,
    "candidate_as_player_two": candidate_as_p2,
  }


def append_round_records(champion_records: list[dict], round_dataset_path: Path) -> list[dict]:
  round_payload = load_records_file(round_dataset_path)
  champion_records.extend(round_payload["records"])
  return champion_records


def write_champion_streak_dataset(
  dataset_path: Path,
  game_type: str,
  champion_records: list[dict],
  round_idx: int,
) -> Path:
  save_records_file(
    dataset_path,
    game_type,
    champion_records,
    {"source_rounds": round_idx, "champion_streak": True},
  )
  return dataset_path


def resolve_devices(args) -> tuple[str, str]:
  train_device = args.train_device if args.train_device is not None else args.device
  if args.play_device is not None:
    play_device = args.play_device
  elif args.train_device is not None and train_device.startswith("cuda"):
    play_device = "cpu"
  else:
    play_device = args.device
  return play_device, train_device


def resolve_start_round(args) -> int:
  if args.start_round is not None:
    return args.start_round
  if args.initial_checkpoint:
    match = re.match(r"round_(\d+)\.pt$", Path(args.initial_checkpoint).name, re.IGNORECASE)
    if match:
      return int(match.group(1)) + 1
  return 1


def resolve_round_player(player_type: str, static_model: str | None, latest_checkpoint: str | None):
  if player_type != "nmcts":
    return player_type, None
  if static_model is not None:
    return "nmcts", static_model
  if latest_checkpoint is not None:
    return "nmcts", latest_checkpoint
  return "random", None


def run_training_round(
  args,
  offset: int,
  round_idx: int,
  training_dataset_path: Path,
  checkpoint_output: Path,
  checkpoint_path: str | None,
  train_device: str,
) -> dict:
  return run_training(
    game_type=args.game_type,
    dataset_path=str(training_dataset_path),
    output_model=str(checkpoint_output),
    checkpoint_path=checkpoint_path,
    device=train_device,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    value_loss_weight=args.value_loss_weight,
    policy_loss_weight=args.policy_loss_weight,
    val_split=args.val_split,
    seed=args.seed + offset,
    augment_train=args.augment_train,
    augment_val=args.augment_val,
    deduplicate_train=args.deduplicate_train,
    deduplicate_val=args.deduplicate_val,
    use_amp=args.amp,
    non_interactive_logging=args.non_interactive_logging,
  )


def run_legacy_pipeline(args, output_dir: Path, play_device: str, train_device: str, start_round: int) -> str:
  datasets_dir = output_dir / "datasets"
  checkpoints_dir = output_dir / "checkpoints"
  latest_checkpoint = args.initial_checkpoint
  cumulative_records = []
  round_numbers = range(start_round, start_round + args.rounds)

  previous_avg_game_length = None
  round_iterator = tqdm(
    round_numbers,
    desc="Pipeline rounds",
    unit="round",
    **tqdm_kwargs(args.non_interactive_logging),
  )
  for offset, round_idx in enumerate(round_iterator):
    round_dataset_path = datasets_dir / f"round_{round_idx:03d}.pkl"

    player_one_type, player_one_model = resolve_round_player(args.player1_type, args.player1_model, latest_checkpoint)
    player_two_type, player_two_model = resolve_round_player(args.player2_type, args.player2_model, latest_checkpoint)

    if round_idx == start_round:
      bootstrap_players = []
      if args.player1_type == "nmcts" and player_one_type == "random":
        bootstrap_players.append("player1")
      if args.player2_type == "nmcts" and player_two_type == "random":
        bootstrap_players.append("player2")
      if bootstrap_players:
        joined = ", ".join(bootstrap_players)
        tqdm.write(
          f"Bootstrapping {joined} with random play in round {round_idx} because no checkpoint was provided. "
          "Later rounds will use the newly trained model."
        )

    match_summary = run_matches(
      game_type=args.game_type,
      num_games=args.games_per_round,
      player_one_type=player_one_type,
      player_one_iters=args.player1_iters,
      player_one_model=player_one_model,
      player_two_type=player_two_type,
      player_two_iters=args.player2_iters,
      player_two_model=player_two_model,
      play_device=play_device,
      record_output=str(round_dataset_path),
      workers=args.self_play_workers,
      show_mcts_timing=args.show_mcts_timing,
      previous_avg_game_length=previous_avg_game_length,
      non_interactive_logging=args.non_interactive_logging,
    )
    previous_avg_game_length = match_summary.get("avg_game_length")

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

    checkpoint_output = checkpoints_dir / f"round_{round_idx:03d}.pt"
    train_result = run_training_round(
      args,
      offset,
      round_idx,
      training_dataset_path,
      checkpoint_output,
      latest_checkpoint,
      train_device,
    )

    latest_checkpoint = str(checkpoint_output)
    round_iterator.set_postfix({
      "latest": Path(latest_checkpoint).name,
      "train_loss": f"{train_result['final_train_loss']:.4f}",
      "val_loss": f"{train_result['final_val_loss']:.4f}" if train_result["final_val_loss"] is not None else "n/a",
    })

  return latest_checkpoint


def run_eval_gated_pipeline(args, output_dir: Path, play_device: str, train_device: str, start_round: int) -> str:
  datasets_dir = output_dir / "datasets"
  checkpoints_dir = output_dir / "checkpoints"
  champion_checkpoint = args.initial_checkpoint
  champion_records: list[dict] = []
  champion_round: int | None = None
  round_numbers = range(start_round, start_round + args.rounds)

  previous_avg_game_length = None
  round_iterator = tqdm(
    round_numbers,
    desc="Pipeline rounds",
    unit="round",
    **tqdm_kwargs(args.non_interactive_logging),
  )
  for offset, round_idx in enumerate(round_iterator):
    round_dataset_path = datasets_dir / f"round_{round_idx:03d}.pkl"

    player_one_type, player_one_model = resolve_round_player(args.player1_type, args.player1_model, champion_checkpoint)
    player_two_type, player_two_model = resolve_round_player(args.player2_type, args.player2_model, champion_checkpoint)

    if round_idx == start_round:
      bootstrap_players = []
      if args.player1_type == "nmcts" and player_one_type == "random":
        bootstrap_players.append("player1")
      if args.player2_type == "nmcts" and player_two_type == "random":
        bootstrap_players.append("player2")
      if bootstrap_players:
        joined = ", ".join(bootstrap_players)
        tqdm.write(
          f"Bootstrapping {joined} with random play in round {round_idx} because no checkpoint was provided. "
          "Later rounds will use the newly trained model."
        )

    match_summary = run_matches(
      game_type=args.game_type,
      num_games=args.games_per_round,
      player_one_type=player_one_type,
      player_one_iters=args.player1_iters,
      player_one_model=player_one_model,
      player_two_type=player_two_type,
      player_two_iters=args.player2_iters,
      player_two_model=player_two_model,
      play_device=play_device,
      record_output=str(round_dataset_path),
      workers=args.self_play_workers,
      show_mcts_timing=args.show_mcts_timing,
      previous_avg_game_length=previous_avg_game_length,
      non_interactive_logging=args.non_interactive_logging,
    )
    previous_avg_game_length = match_summary.get("avg_game_length")

    append_round_records(champion_records, round_dataset_path)
    training_dataset_path = write_champion_streak_dataset(
      datasets_dir / f"round_{round_idx:03d}_champion.pkl",
      args.game_type,
      champion_records,
      round_idx,
    )

    checkpoint_output = checkpoints_dir / f"round_{round_idx:03d}.pt"
    train_result = run_training_round(
      args,
      offset,
      round_idx,
      training_dataset_path,
      checkpoint_output,
      champion_checkpoint,
      train_device,
    )

    candidate_checkpoint = str(checkpoint_output)
    postfix = {
      "champion": Path(champion_checkpoint).name if champion_checkpoint else "none",
      "train_loss": f"{train_result['final_train_loss']:.4f}",
      "val_loss": f"{train_result['final_val_loss']:.4f}" if train_result["final_val_loss"] is not None else "n/a",
    }

    if champion_checkpoint is None:
      champion_checkpoint = candidate_checkpoint
      champion_records = []
      champion_round = round_idx
      postfix["eval_wr"] = "n/a"
      postfix["status"] = "promoted"
      tqdm.write(
        f"Round {round_idx:03d} eval: skipped (no prior champion) — auto-promoted {Path(candidate_checkpoint).name}"
      )
    else:
      eval_iters = args.player1_iters if args.player1_type == "nmcts" else args.player2_iters
      eval_summary = run_candidate_eval(
        game_type=args.game_type,
        num_games=args.num_eval_games,
        candidate_ckpt=candidate_checkpoint,
        champion_ckpt=champion_checkpoint,
        iters=eval_iters,
        play_device=play_device,
        workers=args.self_play_workers,
        non_interactive_logging=args.non_interactive_logging,
      )
      candidate_wins = eval_summary["candidate_wins"]
      candidate_win_rate = eval_summary["candidate_win_rate"]
      postfix["eval_wr"] = f"{candidate_win_rate:.1%}"

      if should_promote(candidate_win_rate, args.winrate_threshold):
        champion_checkpoint = candidate_checkpoint
        champion_records = []
        champion_round = round_idx
        postfix["status"] = "promoted"
        tqdm.write(
          f"Round {round_idx:03d} eval: candidate {candidate_wins}/{args.num_eval_games} vs champion "
          f"({candidate_win_rate:.1%}) — promoted"
        )
      else:
        postfix["status"] = "rejected"
        tqdm.write(
          f"Round {round_idx:03d} eval: candidate {candidate_wins}/{args.num_eval_games} vs champion "
          f"({candidate_win_rate:.1%}) — rejected, keeping {Path(champion_checkpoint).name}"
        )

    round_iterator.set_postfix(postfix)

  return champion_checkpoint


def main():
  args = build_parser().parse_args()
  validate_eval_args(args)

  output_dir = Path(args.output_dir)
  datasets_dir = output_dir / "datasets"
  checkpoints_dir = output_dir / "checkpoints"
  datasets_dir.mkdir(parents=True, exist_ok=True)
  checkpoints_dir.mkdir(parents=True, exist_ok=True)

  play_device, train_device = resolve_devices(args)
  start_round = resolve_start_round(args)

  if eval_enabled(args):
    final_checkpoint = run_eval_gated_pipeline(args, output_dir, play_device, train_device, start_round)
    tqdm.write(f"Pipeline complete. Champion checkpoint: {final_checkpoint}")
  else:
    final_checkpoint = run_legacy_pipeline(args, output_dir, play_device, train_device, start_round)
    tqdm.write(f"Pipeline complete. Latest checkpoint: {final_checkpoint}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())

from pathlib import Path
from time import perf_counter
from typing import Any

from nnmcts.arena.Arena import Arena
from nnmcts.cli_utils import (
  build_model,
  create_environment,
  create_nmcts_player,
  create_player,
  get_cached_model,
  get_game_spec,
)
from nnmcts.mcts.mcts import collect_mcts_timing
from nnmcts.mcts.nodes import NeuralNode


def _same_model_path(path_a: str | None, path_b: str | None) -> bool:
  if path_a is None or path_b is None:
    return path_a == path_b
  return Path(path_a).resolve() == Path(path_b).resolve()


def _create_worker_player(
  environment,
  game_type: str,
  player_type: str,
  is_first: bool,
  iter_count: int,
  model_path: str | None,
  device: str,
  player_name: str,
  model_arg_name: str,
  show_mcts_timing: bool,
  shared_model=None,
):
  normalized_type = player_type.lower()
  if normalized_type != "nmcts":
    return create_player(
      environment,
      game_type,
      player_type,
      is_first,
      iter_count,
      model_path,
      device,
      player_name,
      model_arg_name,
      show_mcts_timing=show_mcts_timing,
    )

  if model_path is None:
    return create_player(
      environment,
      game_type,
      player_type,
      is_first,
      iter_count,
      model_path,
      device,
      player_name,
      model_arg_name,
      show_mcts_timing=show_mcts_timing,
    )

  model = shared_model if shared_model is not None else get_cached_model(game_type, model_path, device)
  return create_nmcts_player(
    environment,
    game_type,
    is_first,
    iter_count,
    model,
    player_name,
    show_mcts_timing=show_mcts_timing,
  )


def play_game_worker(args: dict[str, Any]) -> dict[str, Any]:
  game_type = args["game_type"]
  record = args.get("record", False)
  collect_timing = args.get("collect_mcts_timing", False)
  show_mcts_timing = args.get("show_mcts_timing", False)
  player_one_model_path = args.get("player_one_model")
  player_two_model_path = args.get("player_two_model")
  shared_nmcts_model = None
  if (
    args["player_one_type"].lower() == "nmcts"
    and args["player_two_type"].lower() == "nmcts"
    and _same_model_path(player_one_model_path, player_two_model_path)
    and player_one_model_path is not None
  ):
    shared_nmcts_model = get_cached_model(game_type, player_one_model_path, args["device"])

  start = perf_counter()
  environment = create_environment(game_type)

  player_one = _create_worker_player(
    environment,
    game_type,
    args["player_one_type"],
    True,
    args["player_one_iters"],
    player_one_model_path,
    args["device"],
    "player_one",
    "--player1-model",
    show_mcts_timing=show_mcts_timing and collect_timing,
    shared_model=shared_nmcts_model,
  )
  player_two = _create_worker_player(
    environment,
    game_type,
    args["player_two_type"],
    False,
    args["player_two_iters"],
    player_two_model_path,
    args["device"],
    "player_two",
    "--player2-model",
    show_mcts_timing=show_mcts_timing and collect_timing,
    shared_model=shared_nmcts_model,
  )

  arena = Arena(environment, player_one, player_two)
  mcts_timing = None
  progress_queue = args.get("progress_queue")
  game_index = args.get("game_index")
  turn_count = 0

  def on_turn(turn: int) -> None:
    nonlocal turn_count
    turn_count = turn
    if progress_queue is not None:
      progress_queue.put((game_index, turn))

  if record:
    winner, game_record = arena.play_game(record=True, on_turn=on_turn)
  else:
    winner = arena.play_game(record=False, on_turn=on_turn)
    game_record = None

  wall_time = perf_counter() - start

  if collect_timing and not mcts_timing and (
    args["player_one_type"] == "nmcts" or args["player_two_type"] == "nmcts"
  ):
    mcts_timing = _sample_mcts_timing(
      game_type,
      args["device"],
      player_one_model_path or player_two_model_path,
      max(args["player_one_iters"], args["player_two_iters"]),
    )

  return {
    "winner": winner,
    "record": game_record,
    "wall_time": wall_time,
    "mcts_timing": mcts_timing,
    "turn_count": turn_count,
  }


def _sample_mcts_timing(
  game_type: str,
  device: str,
  model_path: str | None,
  iters: int,
) -> dict[str, float]:
  environment = create_environment(game_type)
  spec = get_game_spec(game_type)

  model, _ = build_model(game_type, checkpoint_path=model_path, device=device)
  model.eval()
  node_cls = type("BenchmarkNeuralNode", (NeuralNode,), {})
  node_cls.set_model(model, spec.build_tensor, uses_mask=spec.uses_mask)

  env_copy = environment.copy()
  node = node_cls(env_copy, env_copy.is_terminal(), None, None)
  sample_iters = min(iters, 10)
  return collect_mcts_timing(node, sample_iters)

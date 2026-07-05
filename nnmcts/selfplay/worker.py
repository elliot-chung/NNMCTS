from time import perf_counter
from typing import Any

from nnmcts.arena.Arena import Arena
from nnmcts.cli_utils import build_model, create_environment, create_player, get_game_spec
from nnmcts.mcts.mcts import collect_mcts_timing
from nnmcts.mcts.nodes import NeuralNode


def play_game_worker(args: dict[str, Any]) -> dict[str, Any]:
  game_type = args["game_type"]
  record = args.get("record", False)
  collect_timing = args.get("collect_mcts_timing", False)
  show_mcts_timing = args.get("show_mcts_timing", False)

  start = perf_counter()
  environment = create_environment(game_type)

  player_one = create_player(
    environment,
    game_type,
    args["player_one_type"],
    True,
    args["player_one_iters"],
    args.get("player_one_model"),
    args["device"],
    "player_one",
    "--player1-model",
    show_mcts_timing=show_mcts_timing and collect_timing,
  )
  player_two = create_player(
    environment,
    game_type,
    args["player_two_type"],
    False,
    args["player_two_iters"],
    args.get("player_two_model"),
    args["device"],
    "player_two",
    "--player2-model",
    show_mcts_timing=show_mcts_timing and collect_timing,
  )

  arena = Arena(environment, player_one, player_two)
  mcts_timing = None
  progress_queue = args.get("progress_queue")
  game_index = args.get("game_index")

  def on_turn(turn: int) -> None:
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
      args.get("player_one_model") or args.get("player_two_model"),
      max(args["player_one_iters"], args["player_two_iters"]),
    )

  return {
    "winner": winner,
    "record": game_record,
    "wall_time": wall_time,
    "mcts_timing": mcts_timing,
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

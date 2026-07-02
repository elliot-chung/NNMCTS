from time import perf_counter
from typing import Any

from nnmcts.arena.Arena import Arena
from nnmcts.cli_utils import create_environment, create_player, get_game_spec
from nnmcts.inference.client import InferenceClient
from nnmcts.mcts.mcts import collect_mcts_timing
from nnmcts.mcts.nodes import NeuralNode


def _configure_neural_player(player, game_type: str, inference_client: InferenceClient | None):
  if inference_client is None:
    return

  if not hasattr(player, "node_cls"):
    return

  spec = get_game_spec(game_type)
  NeuralNode.set_inference_client(inference_client, spec.build_tensor, uses_mask=spec.uses_mask)
  player.node_cls.inference_client = inference_client
  player.node_cls.build_tensor = spec.build_tensor
  player.node_cls.uses_mask = spec.uses_mask
  player.node_cls.model = None


def play_game_worker(args: dict[str, Any]) -> dict[str, Any]:
  game_type = args["game_type"]
  record = args.get("record", False)
  collect_timing = args.get("collect_mcts_timing", False)
  show_mcts_timing = args.get("show_mcts_timing", False)

  inference_client = None
  if args.get("use_inference_server"):
    inference_client = InferenceClient(args["request_queue"], args["results_dict"])

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
    inference_client=inference_client,
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
    inference_client=inference_client,
    show_mcts_timing=show_mcts_timing and collect_timing,
  )

  _configure_neural_player(player_one, game_type, inference_client)
  _configure_neural_player(player_two, game_type, inference_client)

  arena = Arena(environment, player_one, player_two)
  mcts_timing = None

  if record:
    winner, game_record = arena.play_game(record=True)
  else:
    winner = arena.play_game(record=False)
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
      inference_client,
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
  inference_client: InferenceClient | None,
) -> dict[str, float]:
  environment = create_environment(game_type)
  spec = get_game_spec(game_type)

  if inference_client is not None:
    node_cls = type("BenchmarkNeuralNode", (NeuralNode,), {})
    node_cls.set_inference_client(inference_client, spec.build_tensor, uses_mask=spec.uses_mask)
  else:
    from nnmcts.cli_utils import build_model

    model, _ = build_model(game_type, checkpoint_path=model_path, device=device)
    model.eval()
    node_cls = type("BenchmarkNeuralNode", (NeuralNode,), {})
    node_cls.set_model(model, spec.build_tensor, uses_mask=spec.uses_mask)

  env_copy = environment.copy()
  node = node_cls(env_copy, env_copy.is_terminal(), None, None)
  sample_iters = min(iters, 10)
  return collect_mcts_timing(node, sample_iters)

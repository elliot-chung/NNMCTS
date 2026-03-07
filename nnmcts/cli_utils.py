import pickle
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from nnmcts.games.TicTacToe.TTT import TTTGame
from nnmcts.games.TicTacToe.TTTNet import TTTNet, build_tensor as ttt_build_tensor
from nnmcts.games.TicTacToe.TTTRecordDataset import TTTRecordDataset
from nnmcts.games.UltimateTicTacToe.UTTT import UTTTGame
from nnmcts.games.UltimateTicTacToe.UTTTNet import UTTTNet, build_tensor as uttt_build_tensor
from nnmcts.games.UltimateTicTacToe.UTTTRecordDataset import UTTTRecordDataset
from nnmcts.mcts.mcts import mcts
from nnmcts.mcts.nodes import NeuralNode
from nnmcts.players.players import MCTS_Player, Player, Random_Player


@dataclass(frozen=True)
class GameSpec:
  game_type: str
  environment_cls: type
  model_cls: type[torch.nn.Module]
  dataset_cls: type
  build_tensor: Any
  uses_mask: bool


GAME_SPECS = {
  "TTT": GameSpec("TTT", TTTGame, TTTNet, TTTRecordDataset, ttt_build_tensor, False),
  "UTTT": GameSpec("UTTT", UTTTGame, UTTTNet, UTTTRecordDataset, uttt_build_tensor, True),
}


def normalize_game_type(game_type: str) -> str:
  normalized = game_type.upper()
  if normalized not in GAME_SPECS:
    raise ValueError(f"Unsupported game type: {game_type}")
  return normalized


def get_game_spec(game_type: str) -> GameSpec:
  return GAME_SPECS[normalize_game_type(game_type)]


def create_environment(game_type: str):
  spec = get_game_spec(game_type)
  return spec.environment_cls()


def default_device() -> str:
  return "cuda" if torch.cuda.is_available() else "cpu"


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
  log_probs = torch.log_softmax(logits, dim=1)
  return -(target_probs * log_probs).sum(dim=1).mean()


def ensure_parent_dir(path: str | Path):
  Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_records_file(path: str | Path, game_type: str, records: list[dict], metadata: dict | None = None):
  ensure_parent_dir(path)
  payload = {
    "game_type": normalize_game_type(game_type),
    "records": records,
    "metadata": metadata or {},
  }
  with open(path, "wb") as handle:
    pickle.dump(payload, handle)


def load_records_file(path: str | Path) -> dict:
  with open(path, "rb") as handle:
    payload = pickle.load(handle)

  if isinstance(payload, list):
    return {"game_type": None, "records": payload, "metadata": {}}
  if not isinstance(payload, dict) or "records" not in payload:
    raise ValueError("Dataset file must contain a records payload")
  return payload


def load_checkpoint_file(path: str | Path, map_location: str | torch.device = "cpu"):
  payload = torch.load(path, map_location=map_location)
  if isinstance(payload, dict) and "model_state_dict" in payload:
    return payload
  if isinstance(payload, dict):
    return {"model_state_dict": payload}
  raise ValueError("Checkpoint must be a state dict or a checkpoint dictionary")


def build_model(game_type: str, checkpoint_path: str | None = None, device: str | torch.device = "cpu"):
  spec = get_game_spec(game_type)
  model = spec.model_cls()
  metadata = {}

  if checkpoint_path is not None:
    checkpoint = load_checkpoint_file(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = checkpoint

  model.to(device)
  return model, metadata


def save_model_checkpoint(
  path: str | Path,
  model: torch.nn.Module,
  game_type: str,
  metadata: dict | None = None,
):
  ensure_parent_dir(path)
  payload = {
    "game_type": normalize_game_type(game_type),
    "model_state_dict": model.state_dict(),
  }
  if metadata:
    payload.update(metadata)
  torch.save(payload, path)


class ScriptNeuralMCTSPlayer(Player):
  def __init__(self, environment, is_first, iter_count: int, model: torch.nn.Module, build_tensor, player_name: str):
    super().__init__(environment, is_first)
    self.iter_count = iter_count
    self.player_name = player_name

    node_name = f"{player_name}NeuralNode"
    self.node_cls = type(node_name, (NeuralNode,), {})
    self.node_cls.model = model
    self.node_cls.build_tensor = build_tensor

  def on_my_turn(self):
    env_copy = self.environment.copy()
    node = self.node_cls(env_copy, env_copy.is_terminal(), None, None)
    node, policy = mcts(node, self.iter_count)
    return node.action, policy


def create_player(
  environment,
  game_type: str,
  player_type: str,
  is_first: bool,
  iter_count: int,
  model_path: str | None,
  device: str,
  player_name: str,
  model_arg_name: str | None = None,
):
  normalized_type = player_type.lower()
  if normalized_type == "random":
    return Random_Player(environment, is_first)
  if normalized_type == "mcts":
    return MCTS_Player(environment, is_first, iter_count)
  if normalized_type != "nmcts":
    raise ValueError(f"Unsupported player type: {player_type}")
  if model_path is None:
    required_arg = model_arg_name or f"--{player_name.replace('_', '-')}-model"
    raise ValueError(f"{player_name} requires {required_arg} when using nmcts")

  spec = get_game_spec(game_type)
  model, _ = build_model(game_type, checkpoint_path=model_path, device=device)
  model.eval()
  return ScriptNeuralMCTSPlayer(environment, is_first, iter_count, model, spec.build_tensor, player_name)


def summarize_results(results: dict[int, int], game_count: int):
  player_one_wins = results.get(1, 0)
  draws = results.get(0, 0)
  player_two_wins = results.get(-1, 0)
  return {
    "player_one_wins": player_one_wins,
    "draws": draws,
    "player_two_wins": player_two_wins,
    "player_one_win_rate": player_one_wins / game_count if game_count else 0.0,
    "draw_rate": draws / game_count if game_count else 0.0,
    "player_two_win_rate": player_two_wins / game_count if game_count else 0.0,
  }


def split_records(records: list[dict], val_split: float, seed: int):
  if not 0 <= val_split < 1:
    raise ValueError("val_split must be in the range [0, 1)")

  shuffled = records.copy()
  random.Random(seed).shuffle(shuffled)
  val_count = int(len(shuffled) * val_split)
  if val_split > 0 and len(shuffled) > 1:
    val_count = max(1, val_count)
  train_count = len(shuffled) - val_count
  if train_count <= 0:
    raise ValueError("Training split is empty; reduce val_split or provide more records")
  return shuffled[:train_count], shuffled[train_count:]


def deduplicate_supervised_dataset(dataset):
  aggregator = OrderedDict()

  for sample in dataset:
    *features, policy, reward = sample
    key = tuple(feature.detach().cpu().numpy().tobytes() for feature in features)
    if key not in aggregator:
      aggregator[key] = {
        "features": [feature.clone() for feature in features],
        "policy_sum": policy.clone(),
        "reward_sum": reward.clone(),
        "count": 1,
      }
      continue

    aggregator[key]["policy_sum"] += policy
    aggregator[key]["reward_sum"] += reward
    aggregator[key]["count"] += 1

  if not aggregator:
    raise ValueError("Cannot deduplicate an empty dataset")

  feature_lists = [[] for _ in range(len(next(iter(aggregator.values()))["features"]))]
  policies = []
  rewards = []

  for entry in aggregator.values():
    count = entry["count"]
    for feature_list, feature in zip(feature_lists, entry["features"]):
      feature_list.append(feature)
    policies.append(entry["policy_sum"] / count)
    rewards.append(entry["reward_sum"] / count)

  tensors = [torch.stack(feature_list) for feature_list in feature_lists]
  tensors.append(torch.stack(policies))
  tensors.append(torch.stack(rewards))
  return TensorDataset(*tensors)


def build_record_datasets(
  game_type: str,
  records: list[dict],
  val_split: float,
  seed: int,
  augment_train: bool,
  augment_val: bool,
  deduplicate_train: bool,
  deduplicate_val: bool,
):
  spec = get_game_spec(game_type)
  train_records, val_records = split_records(records, val_split, seed)

  train_dataset = spec.dataset_cls(train_records)
  if augment_train:
    train_dataset.augment_data()
  if deduplicate_train:
    train_dataset = deduplicate_supervised_dataset(train_dataset)

  val_dataset = None
  if val_records:
    val_dataset = spec.dataset_cls(val_records)
    if augment_val:
      val_dataset.augment_data()
    if deduplicate_val:
      val_dataset = deduplicate_supervised_dataset(val_dataset)

  return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size: int):
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = None
  if val_dataset is not None:
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  return train_loader, val_loader


def format_ratio(value: float) -> str:
  return f"{value:.3f}"


def prepare_inputs(game_type: str, batch):
  spec = get_game_spec(game_type)
  if spec.uses_mask:
    states, masks, policies, rewards = batch
    return torch.stack((states, masks), dim=1), policies, rewards

  states, policies, rewards = batch
  return states, policies, rewards

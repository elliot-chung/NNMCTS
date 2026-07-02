#!/usr/bin/env python3
"""Export golden UTTT inference fixtures for TypeScript tests."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nnmcts.cli_utils import build_model, ensure_parent_dir  # noqa: E402
from nnmcts.games.UltimateTicTacToe.UTTT import UTTTGame  # noqa: E402


def _load_export_wrapper():
  import importlib.util

  export_path = Path(__file__).resolve().parent / "export_onnx.py"
  spec = importlib.util.spec_from_file_location("nnmcts_export_onnx", export_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.UTTTNetOnnxExport


def tensor_from_game(game: UTTTGame) -> torch.Tensor:
  state, mask = game.get_canonical_state()
  return torch.tensor([state, mask], dtype=torch.float32).unsqueeze(0)


def named_scenarios() -> list[tuple[str, UTTTGame]]:
  scenarios: list[tuple[str, UTTTGame]] = []

  initial = UTTTGame()
  scenarios.append(("initial", initial))

  center = UTTTGame()
  center.make_move((4, 4))
  scenarios.append(("center_opening", center))

  forced = UTTTGame()
  forced.make_move((0, 3))
  scenarios.append(("forced_board_3", forced))

  top_left_win = UTTTGame()
  for move in [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]:
    top_left_win.make_move(move)
  scenarios.append(("top_left_board_won", top_left_win))

  return scenarios


def random_scenarios(rng: random.Random, count: int) -> list[tuple[str, UTTTGame]]:
  scenarios: list[tuple[str, UTTTGame]] = []
  for index in range(count):
    game = UTTTGame()
    move_count = rng.randint(1, 15)
    for _ in range(move_count):
      if game.is_terminal():
        break
      game.make_move(rng.choice(game.valid_moves()))
    scenarios.append((f"random_{index:02d}", game))
  return scenarios


def fixture_from_game(name: str, game: UTTTGame, export_model) -> dict:
  state, mask = game.get_canonical_state()
  tensor = tensor_from_game(game)
  with torch.no_grad():
    policy_logits, value = export_model(tensor)

  return {
    "name": name,
    "state": state,
    "mask": mask,
    "expected_logits": policy_logits.squeeze(0).tolist(),
    "expected_value": float(value.squeeze().item()),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate golden UTTT inference fixtures")
  parser.add_argument("--checkpoint", default="artifacts/gpu-20260701-192839/checkpoints/round_020.pt")
  parser.add_argument(
    "--output-dir",
    default="demo/src/lib/__fixtures__/inference",
    help="Directory for JSON fixture files",
  )
  parser.add_argument("--random-count", type=int, default=8)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  checkpoint_path = Path(args.checkpoint)
  if not checkpoint_path.exists():
    print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
    print("Run scripts/download_checkpoint.ps1 first, or download manually (see script comments).", file=sys.stderr)
    sys.exit(1)

  uttt_onnx_export = _load_export_wrapper()
  model, _ = build_model("UTTT", checkpoint_path=str(checkpoint_path), device="cpu")
  export_model = uttt_onnx_export(model)
  export_model.eval()

  rng = random.Random(args.seed)
  scenarios = named_scenarios() + random_scenarios(rng, args.random_count)

  fixtures = [fixture_from_game(name, game, export_model) for name, game in scenarios]

  output_dir = Path(args.output_dir)
  ensure_parent_dir(output_dir / "index.json")
  index_path = output_dir / "index.json"
  index_path.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")

  for fixture in fixtures:
    fixture_path = output_dir / f"{fixture['name']}.json"
    fixture_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")

  print(f"Wrote {len(fixtures)} fixtures to {output_dir}")


if __name__ == "__main__":
  main()

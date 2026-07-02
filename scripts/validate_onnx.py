#!/usr/bin/env python3
"""Parity check PyTorch vs ONNX Runtime on random UTTT positions."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nnmcts.cli_utils import build_model  # noqa: E402
from nnmcts.games.UltimateTicTacToe.UTTT import UTTTGame  # noqa: E402


def _load_export_wrapper():
  import importlib.util

  export_path = Path(__file__).resolve().parent / "export_onnx.py"
  spec = importlib.util.spec_from_file_location("nnmcts_export_onnx", export_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.UTTTNetOnnxExport


def random_tensor(rng: random.Random) -> torch.Tensor:
  game = UTTTGame()
  move_count = rng.randint(0, 12)
  for _ in range(move_count):
    if game.is_terminal():
      break
    move = rng.choice(game.valid_moves())
    game.make_move(move)

  state, mask = game.get_canonical_state()
  tensor = torch.tensor([state, mask], dtype=torch.float32).unsqueeze(0)
  return tensor


def validate(
  checkpoint_path: Path,
  onnx_path: Path,
  samples: int,
  max_abs_diff: float,
  seed: int,
) -> bool:
  uttt_onnx_export = _load_export_wrapper()
  model, _ = build_model("UTTT", checkpoint_path=str(checkpoint_path), device="cpu")
  export_model = uttt_onnx_export(model)
  export_model.eval()

  if not onnx_path.exists():
    print(f"ONNX file not found: {onnx_path}", file=sys.stderr)
    return False

  session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  rng = random.Random(seed)

  worst_policy = 0.0
  worst_value = 0.0
  ok = True

  for index in range(samples):
    tensor = random_tensor(rng)
    with torch.no_grad():
      pt_policy, pt_value = export_model(tensor)

    onnx_policy, onnx_value = session.run(None, {"input": tensor.numpy().astype(np.float32)})

    policy_diff = float(np.max(np.abs(pt_policy.numpy() - onnx_policy)))
    value_diff = float(np.max(np.abs(pt_value.numpy() - onnx_value)))
    worst_policy = max(worst_policy, policy_diff)
    worst_value = max(worst_value, value_diff)

    if policy_diff >= max_abs_diff or value_diff >= max_abs_diff:
      ok = False
      print(f"Sample {index}: policy diff={policy_diff:.6e}, value diff={value_diff:.6e} (FAIL)")

  print(f"Checked {samples} random positions (seed={seed})")
  print(f"Worst policy max abs diff: {worst_policy:.6e}")
  print(f"Worst value max abs diff:  {worst_value:.6e}")
  print(f"Threshold:                 {max_abs_diff:.6e}")
  print(f"Result: {'PASS' if ok else 'FAIL'}")

  return ok


def main() -> None:
  parser = argparse.ArgumentParser(description="Validate ONNX export against PyTorch")
  parser.add_argument("--checkpoint", default="artifacts/gpu-20260701-192839/checkpoints/round_020.pt")
  parser.add_argument("--onnx", default="demo/public/models/uttt-v1.onnx")
  parser.add_argument("--samples", type=int, default=32)
  parser.add_argument("--max-abs-diff", type=float, default=1e-4)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  checkpoint_path = Path(args.checkpoint)
  if not checkpoint_path.exists():
    print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
    sys.exit(1)

  ok = validate(
    checkpoint_path,
    Path(args.onnx),
    samples=args.samples,
    max_abs_diff=args.max_abs_diff,
    seed=args.seed,
  )
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()

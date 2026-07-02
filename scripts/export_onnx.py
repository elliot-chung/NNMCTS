#!/usr/bin/env python3
"""Export a UTTT PyTorch checkpoint to ONNX for browser inference.

Requires: pip install onnx onnxscript onnxruntime
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nnmcts.cli_utils import build_model, ensure_parent_dir  # noqa: E402
from nnmcts.games.UltimateTicTacToe.UTTTNet import UTTTNet  # noqa: E402


class UTTTNetOnnxExport(nn.Module):
  """ONNX export wrapper: flat [B, 2, 81] in, explicit reshape to [B, 2, 9, 9]."""

  def __init__(self, model: UTTTNet):
    super().__init__()
    self.model = model

  @staticmethod
  def reshape_input(x: torch.Tensor) -> torch.Tensor:
    # [B, 2, 81] -> [B, 2, 9, 9] using positive perm indices for ONNX compatibility.
    batch_size = x.shape[0]
    x = x.view(batch_size, 2, 3, 3, 3, 3)
    x = x.permute(0, 1, 2, 4, 3, 5)
    return x.reshape(batch_size, 2, 9, 9)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x.shape[0]
    x = self.reshape_input(x)

    x = self.model.conv1(x)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.conv2(x)
    x = self.model.bn2(x)
    x = self.model.relu(x)
    x = self.model.conv3(x)
    x = self.model.bn3(x)
    x = self.model.relu(x)

    x = x.view(batch_size, -1)

    policy_logits = self.model.policy3(self.model.relu(self.model.policy2(self.model.relu(self.model.policy1(x)))))
    value = self.model.tanh(self.model.value3(self.model.relu(self.model.value2(self.model.relu(self.model.value1(x))))))

    return policy_logits, value


def export_onnx(
  checkpoint_path: Path,
  output_path: Path,
  opset_version: int = 17,
) -> None:
  model, _ = build_model("UTTT", checkpoint_path=str(checkpoint_path), device="cpu")
  model.eval()

  export_model = UTTTNetOnnxExport(model)
  export_model.eval()

  dummy = torch.zeros(1, 2, 81, dtype=torch.float32)
  ensure_parent_dir(output_path)

  # dynamo=False avoids Windows console encoding issues with the dynamo exporter.
  torch.onnx.export(
    export_model,
    dummy,
    str(output_path),
    input_names=["input"],
    output_names=["policy_logits", "value"],
    dynamic_axes={
      "input": {0: "batch"},
      "policy_logits": {0: "batch"},
      "value": {0: "batch"},
    },
    opset_version=opset_version,
    do_constant_folding=True,
    dynamo=False,
  )

  print(f"Exported ONNX model to {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Export UTTT checkpoint to ONNX")
  parser.add_argument(
    "--checkpoint",
    default="artifacts/gpu-20260701-192839/checkpoints/round_020.pt",
    help="Path to PyTorch checkpoint",
  )
  parser.add_argument(
    "--output",
    default="demo/public/models/uttt-v1.onnx",
    help="Output ONNX file path",
  )
  parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
  args = parser.parse_args()

  checkpoint_path = Path(args.checkpoint)
  if not checkpoint_path.exists():
    print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
    print("Run scripts/download_checkpoint.ps1 first, or download manually (see script comments).", file=sys.stderr)
    sys.exit(1)

  export_onnx(checkpoint_path, Path(args.output), opset_version=args.opset)


if __name__ == "__main__":
  main()

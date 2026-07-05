import argparse

import torch
from torch import nn
from tqdm import tqdm

from nnmcts.cli_utils import (
  add_non_interactive_logging_arg,
  build_model,
  build_record_datasets,
  create_dataloaders,
  default_device,
  get_game_spec,
  load_records_file,
  prepare_inputs,
  save_model_checkpoint,
  soft_cross_entropy,
  tqdm_kwargs,
)


def build_parser():
  parser = argparse.ArgumentParser(
    description="Train a policy/value model from a recorded dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--game-type", choices=("TTT", "UTTT"), required=True)
  parser.add_argument("--dataset-path", required=True, help="Path to the recorded dataset to train on.")
  parser.add_argument("--output-model", required=True, help="Path to save the trained model.")
  parser.add_argument("--checkpoint-path", help="Optional checkpoint to resume from.")
  parser.add_argument("--device", choices=("cpu", "cuda"), default=default_device(), help="Device to use for training.")
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
  parser.add_argument("--amp", action="store_true", help="Enable mixed-precision training on CUDA.")
  add_non_interactive_logging_arg(parser)
  return parser


def evaluate(model, game_type: str, dataloader, device: str, policy_weight: float, value_weight: float):
  model.eval()
  criterion_value = nn.MSELoss()

  total_loss = 0.0
  total_samples = 0
  with torch.no_grad():
    for batch in dataloader:
      inputs, policies, rewards = prepare_inputs(game_type, batch)
      inputs = inputs.to(device)
      policies = policies.to(device)
      rewards = rewards.to(device)

      predicted_policy_logits, predicted_values = model(inputs)
      policy_loss = soft_cross_entropy(predicted_policy_logits, policies)
      value_loss = criterion_value(predicted_values.squeeze(-1), rewards)
      loss = (policy_weight * policy_loss) + (value_weight * value_loss)

      batch_size = rewards.shape[0]
      total_loss += loss.item() * batch_size
      total_samples += batch_size

  return total_loss / total_samples if total_samples else 0.0


def run_training(
  game_type: str,
  dataset_path: str,
  output_model: str,
  checkpoint_path: str | None,
  device: str,
  epochs: int,
  batch_size: int,
  learning_rate: float,
  weight_decay: float,
  value_loss_weight: float,
  policy_loss_weight: float,
  val_split: float,
  seed: int,
  augment_train: bool,
  augment_val: bool,
  deduplicate_train: bool,
  deduplicate_val: bool,
  use_amp: bool = False,
  non_interactive_logging: bool = False,
):
  torch.manual_seed(seed)

  dataset_payload = load_records_file(dataset_path)
  dataset_game_type = dataset_payload.get("game_type")
  if dataset_game_type and dataset_game_type != game_type:
    raise ValueError(f"Dataset game type {dataset_game_type} does not match requested game {game_type}")

  records = dataset_payload["records"]
  if not records:
    raise ValueError("Dataset contains no records")

  train_dataset, val_dataset = build_record_datasets(
    game_type=game_type,
    records=records,
    val_split=val_split,
    seed=seed,
    augment_train=augment_train,
    augment_val=augment_val,
    deduplicate_train=deduplicate_train,
    deduplicate_val=deduplicate_val,
  )
  train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size, device=device)

  model, checkpoint_metadata = build_model(game_type, checkpoint_path=checkpoint_path, device=device)
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  criterion_value = nn.MSELoss()
  spec = get_game_spec(game_type)
  amp_enabled = use_amp and str(device).startswith("cuda")
  scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

  history = []
  epoch_iterator = tqdm(
    range(epochs),
    desc="Training epochs",
    unit="epoch",
    **tqdm_kwargs(non_interactive_logging),
  )
  for epoch in epoch_iterator:
    model.train()
    total_loss = 0.0

    for batch in train_loader:
      inputs, policies, rewards = prepare_inputs(spec.game_type, batch)
      inputs = inputs.to(device, non_blocking=amp_enabled)
      policies = policies.to(device, non_blocking=amp_enabled)
      rewards = rewards.to(device, non_blocking=amp_enabled)

      optimizer.zero_grad(set_to_none=True)
      with torch.amp.autocast("cuda", enabled=amp_enabled):
        predicted_policy_logits, predicted_values = model(inputs)
        policy_loss = soft_cross_entropy(predicted_policy_logits, policies)
        value_loss = criterion_value(predicted_values.squeeze(-1), rewards)
        loss = (policy_loss_weight * policy_loss) + (value_loss_weight * value_loss)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      total_loss += loss.item() * rewards.shape[0]

    avg_train_loss = total_loss / len(train_dataset)
    avg_val_loss = None
    if val_loader is not None:
      avg_val_loss = evaluate(
        model,
        spec.game_type,
        val_loader,
        device,
        policy_loss_weight,
        value_loss_weight,
      )

    history.append({
      "epoch": epoch + 1,
      "train_loss": avg_train_loss,
      "val_loss": avg_val_loss,
    })

    epoch_iterator.set_postfix({
      "train_loss": f"{avg_train_loss:.4f}",
      "val_loss": f"{avg_val_loss:.4f}" if avg_val_loss is not None else "n/a",
    })

    

  metadata = {
    "epochs": epochs,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "batch_size": batch_size,
    "dataset_path": dataset_path,
    "history": history,
    "base_checkpoint": checkpoint_path,
    "loaded_checkpoint_metadata": checkpoint_metadata,
    "use_amp": amp_enabled,
  }
  save_model_checkpoint(output_model, model, spec.game_type, metadata)

  return {
    "output_model": output_model,
    "train_size": len(train_dataset),
    "val_size": 0 if val_dataset is None else len(val_dataset),
    "final_train_loss": history[-1]["train_loss"],
    "final_val_loss": history[-1]["val_loss"],
  }


def main():
  args = build_parser().parse_args()
  result = run_training(
    game_type=args.game_type,
    dataset_path=args.dataset_path,
    output_model=args.output_model,
    checkpoint_path=args.checkpoint_path,
    device=args.device,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    value_loss_weight=args.value_loss_weight,
    policy_loss_weight=args.policy_loss_weight,
    val_split=args.val_split,
    seed=args.seed,
    augment_train=args.augment_train,
    augment_val=args.augment_val,
    deduplicate_train=args.deduplicate_train,
    deduplicate_val=args.deduplicate_val,
    use_amp=args.amp,
    non_interactive_logging=args.non_interactive_logging,
  )

  print(f"Training dataset size: {result['train_size']}")
  print(f"Validation dataset size: {result['val_size']}")
  print(f"Final train loss: {result['final_train_loss']:.4f}")
  if result["final_val_loss"] is not None:
    print(f"Final val loss: {result['final_val_loss']:.4f}")
  print(f"Saved checkpoint to: {result['output_model']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

import multiprocessing as mp
import time
from queue import Empty

import torch

from nnmcts.cli_utils import build_model, get_game_spec


def _server_main(
  request_queue: mp.Queue,
  results_dict,
  game_type: str,
  checkpoint_path: str | None,
  batch_size: int,
  max_wait_ms: float,
  device: str,
):
  model, _ = build_model(game_type, checkpoint_path=checkpoint_path, device=device)
  model.eval()
  spec = get_game_spec(game_type)
  uses_mask = spec.uses_mask

  while True:
    batch = []
    deadline = time.perf_counter() + (max_wait_ms / 1000.0)

    while len(batch) < batch_size:
      timeout = None if not batch else max(0.0, deadline - time.perf_counter())
      try:
        item = request_queue.get(timeout=timeout)
      except Empty:
        break

      if item is None:
        return

      batch.append(item)

    if not batch:
      continue

    req_ids = [item[0] for item in batch]
    tensors = torch.cat([item[1] for item in batch], dim=0).to(device)
    masks = None
    if batch[0][2] is not None:
      masks = torch.cat([item[2] for item in batch], dim=0).to(device)

    with torch.inference_mode():
      policy_logits, values = model(tensors)

      if masks is not None:
        masked_logits = policy_logits.masked_fill(masks == 0, float("-inf"))
        zero_mask_rows = masks.sum(dim=1) == 0
        policies = torch.softmax(masked_logits, dim=1)
        if zero_mask_rows.any():
          policies = policies.clone()
          policies[zero_mask_rows] = masks[zero_mask_rows]
      elif uses_mask:
        tensor_masks = tensors[:, 1, :]
        masked_logits = policy_logits.masked_fill(tensor_masks == 0, float("-inf"))
        zero_mask_rows = tensor_masks.sum(dim=1) == 0
        policies = torch.softmax(masked_logits, dim=1)
        if zero_mask_rows.any():
          policies = policies.clone()
          policies[zero_mask_rows] = tensor_masks[zero_mask_rows]
      else:
        policies = torch.softmax(policy_logits, dim=1)

    policies_cpu = policies.detach().cpu()
    values_cpu = values.detach().cpu().squeeze(-1)

    for index, req_id in enumerate(req_ids):
      results_dict[req_id] = (
        policies_cpu[index].numpy(),
        float(values_cpu[index].item()),
      )


class InferenceServer:
  def __init__(
    self,
    game_type: str,
    checkpoint_path: str | None = None,
    device: str = "cuda",
    batch_size: int = 32,
    max_wait_ms: float = 5.0,
  ):
    self.game_type = game_type
    self.checkpoint_path = checkpoint_path
    self.device = device
    self.batch_size = batch_size
    self.max_wait_ms = max_wait_ms
    self._ctx = mp.get_context("spawn")
    self._manager = self._ctx.Manager()
    self.results_dict = self._manager.dict()
    self.request_queue = self._ctx.Queue()
    self._process: mp.Process | None = None

  def start(self):
    if self._process is not None and self._process.is_alive():
      return

    self._process = self._ctx.Process(
      target=_server_main,
      args=(
        self.request_queue,
        self.results_dict,
        self.game_type,
        self.checkpoint_path,
        self.batch_size,
        self.max_wait_ms,
        self.device,
      ),
      daemon=True,
    )
    self._process.start()

  def stop(self):
    if self._process is None:
      return

    self.request_queue.put(None)
    self._process.join(timeout=10)
    if self._process.is_alive():
      self._process.terminate()
    self._process = None
    self._manager.shutdown()

  def create_client(self):
    from nnmcts.inference.client import InferenceClient

    return InferenceClient(self.request_queue, self.results_dict)

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc, tb):
    self.stop()
    return False

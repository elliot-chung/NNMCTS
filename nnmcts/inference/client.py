import itertools
import threading
import time

import torch


class InferenceClient:
  _id_counter = itertools.count()
  _id_lock = threading.Lock()

  def __init__(self, request_queue, results_dict):
    self.request_queue = request_queue
    self.results_dict = results_dict

  @classmethod
  def _next_request_id(cls) -> int:
    with cls._id_lock:
      return next(cls._id_counter)

  def evaluate(self, tensor: torch.Tensor, mask: torch.Tensor | None = None, uses_mask: bool = False) -> tuple:
    request_id = self._next_request_id()
    payload = tensor.detach().cpu()
    if payload.dim() == 3:
      payload = payload.unsqueeze(0)
    mask_payload = None
    if mask is not None:
      mask_payload = mask.detach().cpu()
      if mask_payload.dim() == 1:
        mask_payload = mask_payload.unsqueeze(0)
    self.request_queue.put((request_id, payload, mask_payload))

    while request_id not in self.results_dict:
      time.sleep(0.0005)

    policy, value = self.results_dict.pop(request_id)
    return policy, value

  def evaluate_node(self, node, build_tensor, uses_mask: bool = False) -> tuple:
    tensor = build_tensor(node)
    mask = None
    if uses_mask:
      mask = tensor[:, 1, :]
    else:
      mask = torch.tensor(node.environment.get_mask(), dtype=tensor.dtype).unsqueeze(0)
    policy, value = self.evaluate(tensor, mask=mask, uses_mask=uses_mask)
    return policy, value

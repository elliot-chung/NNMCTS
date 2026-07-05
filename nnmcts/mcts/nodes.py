import random
import weakref
from math import sqrt, log
from time import time

NODE_LIST = []


def clear_node_list():
  NODE_LIST.clear()


class Node:
  def __init__(self, environment, terminal, parent, action):
    self.total_reward = 0
    self.visit_count = 0

    self.child = None
    self.environment = environment
    self.terminal = terminal
    self.parent = weakref.ref(parent) if parent else None
    self.action = action

    ref = weakref.ref(self)
    NODE_LIST.append(ref)

  def _ucb(self):
    if self.visit_count == 0:
      return float('inf')

    parent_node = self.parent()
    return (self.total_reward / self.visit_count) + sqrt(1 * log(parent_node.visit_count) / self.visit_count)

  def _expand_child(self, action):
    if self.child is None:
      self.child = {}

    if action in self.child:
      return self.child[action]

    environment = self.environment.copy()
    environment.make_move(action)
    node_cls = type(self)
    child = node_cls(environment, environment.is_terminal(), self, action)
    self.child[action] = child
    return child

  def _create_child(self):
    if self.terminal:
      return

    actions = self.environment.valid_moves()
    if self.child is None:
      self.child = {}

    for action in actions:
      if action not in self.child:
        self._expand_child(action)

  def _rollout(self):
    new_env = self.environment.copy()
    while not new_env.is_terminal():
      actions = new_env.valid_moves()
      action = random.choice(actions)
      new_env.make_move(action)
    reward = new_env.get_winner() * self.environment.current_turn()
    return -reward

  def _traverse_to_leaf(node):
    while not node.terminal:
      if isinstance(node, NeuralNode) and node.neural_policy is None:
        node._evaluate()

      if node.child is None:
        node.child = {}

      actions = node.environment.valid_moves()
      ucb_scores = {}
      for action in actions:
        if action in node.child:
          ucb_scores[action] = node.child[action]._ucb()
        else:
          ucb_scores[action] = float('inf')

      max_ucb = max(ucb_scores.values())
      best_actions = [action for action, score in ucb_scores.items() if score == max_ucb]
      action = random.choice(best_actions)

      if action not in node.child:
        node._expand_child(action)

      node = node.child[action]

    return node

  def _update_parents(node, reward):
    flip = -1
    curr = node
    while curr.parent:
      curr = curr.parent()
      curr.visit_count += 1
      curr.total_reward += reward * flip
      flip *= -1

  def explore(self, perf=None):
    root = self

    start = time()
    current = Node._traverse_to_leaf(root)
    end = time()
    traverse_time = end - start

    start = time()
    reward = current._rollout()
    end = time()
    rollout_time = end - start

    current.total_reward += reward
    current.visit_count += 1

    start = time()
    Node._update_parents(current, reward)
    end = time()
    update_time = end - start

    if perf is not None:
      perf["traverse_time"] = traverse_time
      perf["rollout_time"] = rollout_time
      perf["update_time"] = update_time
      perf["create_time"] = 0.0

  def get_policy(self):
    if self.terminal:
      raise ValueError("Terminal node")

    if not self.child:
      raise ValueError("No children")

    policy = [0] * len(self.environment.get_state())
    for node in self.child.values():
      policy[self.environment.translate(node.action)] = node.visit_count

    sum_p = sum(policy)
    policy = [p / sum_p for p in policy]

    return policy

  def get_most_visited(self):
    if self.terminal:
      raise ValueError("Terminal node")

    if not self.child:
      raise ValueError("No children")

    visit_list = [node.visit_count for node in self.child.values()]
    max_visit = max(visit_list)

    most_visited_nodes = [child for child in self.child.values() if child.visit_count == max_visit]
    return random.choice(most_visited_nodes)

  def detach_parent(self):
    self.parent = None


def print_tree(root_node, indent=0):
  print('  ' * indent + f"- Environment: {root_node.environment.get_state()}, Visits: {root_node.visit_count}, Reward: {root_node.total_reward:.2f}")
  if root_node.child:
    for action, child_node in root_node.child.items():
      print('  ' * (indent + 1) + f"Action: {action}")
      print_tree(child_node, indent + 2)


import torch


class NeuralNode(Node):
  model = None
  build_tensor = None
  uses_mask = False

  def __init__(self, environment, terminal, parent, action):
    super().__init__(environment, terminal, parent, action)
    self.neural_policy = None

  @classmethod
  def set_model(cls, model, closure, uses_mask: bool = False):
    if not isinstance(model, torch.nn.Module):
      raise ValueError("Model is not a PyTorch module")

    cls.model = model
    cls.build_tensor = closure
    cls.uses_mask = uses_mask

  def _evaluate(self):
    if self.environment.is_terminal():
      return -(self.environment.get_winner() * self.environment.current_turn())

    node_cls = type(self)
    device = next(node_cls.model.parameters()).device
    tensor = node_cls.build_tensor(self).to(device)

    with torch.inference_mode():
      node_cls.model.eval()
      policy_logits, value = node_cls.model(tensor)

      if node_cls.uses_mask:
        mask = tensor[:, 1, :]
        masked_logits = policy_logits.masked_fill(mask == 0, float('-inf'))
        if mask.sum().item() == 0:
          masked_policy = mask
        else:
          masked_policy = torch.softmax(masked_logits, dim=1)
      else:
        mask = torch.tensor(
          self.environment.get_mask(),
          dtype=policy_logits.dtype,
          device=policy_logits.device,
        ).unsqueeze(0)
        masked_logits = policy_logits.masked_fill(mask == 0, float('-inf'))
        if mask.sum().item() == 0:
          masked_policy = mask
        else:
          masked_policy = torch.softmax(masked_logits, dim=1)

      self.neural_policy = masked_policy.detach().cpu().numpy()[0]

    reward = value.detach().cpu().item()
    return -reward

  def _rollout(self):
    return self._evaluate()

  def _ucb(self):
    if self.visit_count == 0:
      return float('inf')

    parent_node = self.parent()
    value_score = self.total_reward / self.visit_count
    action_index = self.environment.translate(self.action)
    exploration_score = parent_node.neural_policy[action_index] * sqrt(log(parent_node.visit_count) / self.visit_count)
    return value_score + exploration_score

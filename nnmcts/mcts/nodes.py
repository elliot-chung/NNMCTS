import random
import weakref
from math import sqrt, log
from time import time

NODE_LIST = []

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

  def _create_child(self):
    if self.terminal:
      return

    actions = self.environment.valid_moves()
    environments = [ self.environment.copy() for action in actions]
    for action, environment in zip(actions, environments):
      environment.make_move(action)

    # Get this class type
    NodeVariant = type(self)

    self.child = { action: NodeVariant(environment, environment.is_terminal(), self, action) for action, environment in zip(actions, environments)}

  def _rollout(self):
    new_env = self.environment.copy()
    while not new_env.is_terminal():
      actions = new_env.valid_moves()
      action = random.choice(actions)
      new_env.make_move(action)
    reward = new_env.get_winner() * self.environment.current_turn()
    return -reward

  def _traverse_to_leaf(node):
    while node.child:
      ucb_scores = {a: c._ucb() for a, c in node.child.items()}
      max_ucb = max(ucb_scores.values())
      actions = [a for a, s in ucb_scores.items() if s == max_ucb]

      action = random.choice(actions)
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

    start = time()
    current._create_child()
    end = time()
    create_time = end - start

    if perf is not None:
      perf["traverse_time"] = traverse_time
      perf["rollout_time"] = rollout_time
      perf["update_time"] = update_time
      perf["create_time"] = create_time

  def get_policy(self):
    if self.terminal:
      raise ValueError("Terminal node")

    if not self.child:
      raise ValueError("No children")

    policy = [0] * len(self.environment.get_state())
    for node in self.child.values():
      policy[self.environment.translate(node.action)] = node.visit_count

    # modified softmax
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

    most_visited_nodes = [c for c in self.child.values() if c.visit_count == max_visit]
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

  def __init__(self, environment, terminal, parent, action):
    super().__init__(environment, terminal, parent, action)
    self.neural_policy = None

  # Sets Node to use pytorch model
  # closure should take a node as input and return the tensor that should be
  # fed into the model for this node
  @classmethod
  def set_model(cls, model, closure):
    if not isinstance(model, torch.nn.Module):
      raise ValueError("Model is not a PyTorch module")

    cls.model = model
    cls.build_tensor = closure

  def _rollout(self):
    if self.environment.is_terminal():
      return -(self.environment.get_winner() * self.environment.current_turn())

    tensor = NeuralNode.build_tensor(self)
    with torch.no_grad():
      NeuralNode.model.eval()
      policy, value = NeuralNode.model(tensor)
      policy = policy.detach().numpy()[0]
      value = value.detach().numpy()[0][0]


    self.neural_policy = policy

    reward = value * self.environment.current_turn()
    return -reward

  def _ucb(self):
    if self.visit_count == 0:
      return float('inf')

    parent_node = self.parent()
    value_score = self.total_reward / self.visit_count
    action_index = self.environment.translate(self.action)
    exploration_score = parent_node.neural_policy[action_index] * sqrt(log(parent_node.visit_count) / self.visit_count)
    return value_score + exploration_score
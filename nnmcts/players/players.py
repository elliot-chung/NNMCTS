class Player:
  def __init__(self, environment, is_first=True):
    self.environment = environment
    self.is_first = is_first

  def is_my_turn(self):
    return self.environment.current_turn() == (1 if self.is_first else -1)

  def play_turn(self):
    if not self.is_my_turn(): return None, None

    return self.on_my_turn()

  # Implement this func on subclasses
  # Should return a tuple (action, policy) for this node
  def on_my_turn(self):
    raise NotImplementedError

import random

class Random_Player(Player):
  def __init__(self, environment, is_first=True):
    super().__init__(environment, is_first)

  def on_my_turn(self):
    action = random.choice(self.environment.valid_moves())
    policy = [0] * len(self.environment.get_state())
    for move in self.environment.valid_moves():
      policy[self.environment.translate(move)] = 1 / len(self.environment.valid_moves())
    return action, policy

class Human_Player(Player):
  def __init__(self, environment, is_first=True):
    super().__init__(environment, is_first)

  def on_my_turn(self):
    move = (-1, -1)
    print("Valid Moves:", self.environment.valid_moves())
    while not self.environment.is_valid(move):
      action = input("Enter a move: ")
      try:
        move = eval(action)
      except:
        pass
    return move, None

from nnmcts.mcts.mcts import mcts
from nnmcts.mcts.nodes import Node, NeuralNode

class MCTS_Player(Player):
  def __init__(self, environment, is_first=True, iter_count=100):
    super().__init__(environment, is_first)
    self.iter_count = iter_count

  def on_my_turn(self):
    env_copy = self.environment.copy()
    n = Node(env_copy, env_copy.is_terminal(), None, None)
    n, policy = mcts(n, self.iter_count)
    return n.action, policy

class Neural_MCTS_Player(Player):
  def __init__(self, environment, is_first=True, iter_count=100):
    super().__init__(environment, is_first)
    self.iter_count = iter_count

  def on_my_turn(self):
    env_copy = self.environment.copy()
    n = NeuralNode(env_copy, env_copy.is_terminal(), None, None)
    n, policy = mcts(n, self.iter_count)
    return n.action, policy
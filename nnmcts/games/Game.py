import weakref
from abc import ABC, abstractmethod

"""
Game Abstract Class

This module defines an abstract class for a game as well as a list of weak references to all games that can be used to check how many games are still in memory.

This file contains:
- ENV_LIST: A list of weak references to all games
- clear_env_history(): Clears the ENV_LIST
- check_env_history(): Collects the games still in memory
- Game: An abstract class for a game

"""

ENV_LIST = [] # History of all environments

class Game(ABC):
  """
  Abstract class for a game

  This class defines the interface for a game. It handles the game logic and provides methods for interacting with the game.

  Attributes:
    ENV_LIST (list): A history of all games made since the last clear_env_history() call

  Methods:
    translate(move): Translates a move to a state index
    current_turn(): Returns the current turn of the game
    get_state(): Returns the current state of the game
    is_valid(move): Checks if a move is valid
    valid_moves(): Returns a list of valid moves
    is_terminal(): Checks if the game is terminal
    get_winner(): Returns the winner of the game, 0 if no winner, 1 if player one won, -1 if player two won
    make_move(move): Makes a move in the game
    get_canonical_state(): Returns the canonical form of the state
    __repr__(): Returns a string representation of the game
    __str__(): Returns a string representation of the game
    copy(): Returns a copy of the game
"""
  def __init__(self):
    ref = weakref.ref(self)
    ENV_LIST.append(ref)

  # Translates a move to a state index
  # Moves can be represented as any type but the state array must be indexable by a single integer
  @abstractmethod
  def translate(self, move):
    """
    Translates a move to a state index
    
    Args:
      move (any): The move to be translated
    
    Returns:
      int: The state index
    """
    pass

  # Returns a value representing the player that is currently to move
  @abstractmethod
  def current_turn(self):
    pass

  # Returns a a flat array representing the current state of the game
  @abstractmethod
  def get_state(self):
    pass

  # Returns true if the move is valid, false otherwise
  @abstractmethod
  def is_valid(self, move):
    pass

  # Returns a list of valid moves
  @abstractmethod
  def valid_moves(self):
    pass

  # Returns true if the game is terminal, false otherwise
  @abstractmethod
  def is_terminal(self):
    pass

  # Returns the winner of the game, 0 if no winner, 1 if player one won, -1 if player two won
  @abstractmethod
  def get_winner(self):
    pass

  # Changes the state of the game to the given move
  @abstractmethod
  def make_move(self, move):
    pass

  # Returns the canonical form of the state
  # The canonical form is a representation of the state that is independent of the player
  # The canonical form should hold all the information needed to make a move
  # (This means that the canonical form should have all the information necessary to make an input tensor for a model)
  @abstractmethod
  def get_canonical_state(self):
    pass

  @abstractmethod
  def __repr__(self):
    pass

  @abstractmethod
  def __str__(self):
    pass

  @abstractmethod
  def copy(self):
    pass

def clear_env_history():
  ENV_LIST.clear()
  
def check_env_history():
  num_env = len(ENV_LIST)
  live = []
  
  for ref in ENV_LIST:
    if ref() is not None:
      live.append(ref)
  
  num_dead = num_env - len(live)
  return num_dead, live
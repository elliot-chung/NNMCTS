from abc import ABC, abstractmethod

"""
Game Abstract Class

This module defines an abstract class for a game.

This file contains:
- Game: An abstract class for a game

"""

class Game(ABC):
  """
  Abstract class for a game

  This class defines the interface for a game. It handles the game logic and provides methods for interacting with the game.

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

  # Returns a flat list representing the current state of the game
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
  
  # Returns a list the same shape as the state, where each valid position has a 1 and all other positions have a 0
  @abstractmethod
  def get_mask(self):
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

from typing import Tuple
from enum import IntEnum
from nnmcts.games.Game import Game
import random

class UTTTGame(Game):
  class Position(IntEnum):
    TOPLEFT = 0
    TOPCENTER = 1
    TOPRIGHT = 2
    MIDDLELEFT = 3
    MIDDLECENTER = 4
    MIDDLERIGHT = 5
    BOTTOMLEFT = 6
    BOTTOMCENTER = 7
    BOTTOMRIGHT = 8

  def translate(self, move: Tuple[Position, Position]) -> int:
    board_id, action = UTTTGame.Position(move[0]), UTTTGame.Position(move[1])
    return board_id.value * 9 + action.value

  def get_winner_small_board(state):
    occupant = state[UTTTGame.Position.TOPLEFT]
    if occupant != 0 and occupant != 2:
      if state[UTTTGame.Position.TOPCENTER] == occupant and state[UTTTGame.Position.TOPRIGHT] == occupant:
        return occupant
      if state[UTTTGame.Position.MIDDLELEFT] == occupant and state[UTTTGame.Position.BOTTOMLEFT] == occupant:
        return occupant

    occupant = state[UTTTGame.Position.BOTTOMRIGHT]
    if occupant != 0 and occupant != 2:
      if state[UTTTGame.Position.BOTTOMCENTER] == occupant and state[UTTTGame.Position.BOTTOMLEFT] == occupant:
        return occupant
      if state[UTTTGame.Position.MIDDLERIGHT] == occupant and state[UTTTGame.Position.TOPRIGHT] == occupant:
        return occupant

    occupant = state[UTTTGame.Position.MIDDLECENTER]
    if occupant != 0 and occupant !=2:
      if state[UTTTGame.Position.MIDDLELEFT] == occupant and state[UTTTGame.Position.MIDDLERIGHT] == occupant:
        return occupant
      if state[UTTTGame.Position.TOPLEFT] == occupant and state[UTTTGame.Position.BOTTOMRIGHT] == occupant:
        return occupant
      if state[UTTTGame.Position.TOPCENTER] == occupant and state[UTTTGame.Position.BOTTOMCENTER] == occupant:
        return occupant
      if state[UTTTGame.Position.TOPRIGHT] == occupant and state[UTTTGame.Position.BOTTOMLEFT] == occupant:
        return occupant

    empty_slots = [pos for pos in UTTTGame.Position if state[pos] == 0]
    if len(empty_slots) == 0:
      return 2

    return 0

  def __init__(self, state=None, player=1, previous_move=None, meta_state=None):
    super().__init__()
    if player != 1 and player != -1:
      raise ValueError("Invalid player")

    self.turn = player
    self.previous_move = previous_move

    if state is None:
      self.state = [0] * 81
      self.meta_state = [0] * 9
    else:
      if len(state) != 81: raise ValueError("Invalid state")
      self.state = state
      self.meta_state = self.calculate_meta_state() if meta_state == None else meta_state


  def calculate_meta_state(self):
    meta_state = [0] * 9
    for i in range(9):
      meta_state[i] = UTTTGame.get_winner_small_board(self.state[i*9:i*9+9])
    return meta_state

  def update_meta_state(self, move):
    board_id = move[0]
    if self.meta_state[board_id] != 0:
      raise ValueError("Board already won")

    self.meta_state[board_id] = UTTTGame.get_winner_small_board(self.state[board_id*9:board_id*9+9])

  def current_turn(self):
    return self.turn

  def get_state(self):
    return self.state

  def is_valid(self, move: Tuple[Position, Position]) -> bool:
    try:
      idx = self.translate(move)
    except ValueError:
      return False
    if self.previous_move is None:
      return self.state[idx] == 0
    if move[0] == self.previous_move[1]:
      return self.state[idx] == 0
    return False

  def valid_moves(self) -> list[Tuple[Position, Position]]:
    finished_boards = [i for i, e in enumerate(self.meta_state) if e != 0 ]
    board_id = self.previous_move[1] if self.previous_move is not None else None

    if self.previous_move is None or self.meta_state[board_id] != 0:
      return [(i, j) for i in range(9) for j in range(9) if i not in finished_boards and self.state[self.translate((i, j))] == 0 ]
    moves = [(board_id, i) for i in range(9) if self.state[self.translate((board_id, i))] == 0]
    if len(moves) == 0:
      raise ValueError("No valid moves")
    return moves

  def is_terminal(self) -> bool:
    return self.get_winner() != 0 or len(self.valid_moves()) == 0

  def get_winner(self):
    res = UTTTGame.get_winner_small_board(self.meta_state)
    return 0 if res == 2 else res

  def make_move(self, move: Tuple[Position, Position]):
    self.state[self.translate(move)] = self.turn
    self.previous_move = move
    self.turn = -self.turn

    self.update_meta_state(move)
    return self

  def make_random_move(self):
    moves = self.valid_moves()
    move = random.choice(moves)
    self.make_move(move)
    
  def get_mask(self) -> list[int]:
    pos_mask = [0] * len(self.state)
    for mov in self.valid_moves():
      ind = self.translate(mov)
      pos_mask[ind] = 1
    return pos_mask

  def get_canonical_state(self) -> list[Position]:
    norm_state = [s * self.turn for s in self.state]
    pos_mask = self.get_mask()
    return norm_state, pos_mask

  def state_to_string(state):
    if state == 0:
      return "□"
    elif state == 1:
      return "X"
    elif state == -1:
      return "O"
    else:
      return str(state)


  def __repr__(self):
    return "\n".join(("State: " + str(self.state),
                      "Previous Move: " + str(self.previous_move),
                      "Current Player: " + str(self.turn)))

  def __str__(self):
    state_copy = self.state.copy()
    index_order = [0, 1, 2, 9, 10, 11, 18, 19, 20, 3, 4, 5, 12, 13, 14, 21, 22, 23, 6, 7, 8, 15, 16, 17, 24, 25, 26,
                   27, 28, 29, 36, 37, 38, 45, 46, 47, 30, 31, 32, 39, 40, 41, 48, 49, 50, 33, 34, 35, 42, 43, 44, 51, 52, 53,
                   54, 55, 56, 63, 64, 65, 72, 73, 74, 57, 58, 59, 66, 67, 68, 75, 76, 77, 60, 61, 62, 69, 70, 71, 78, 79, 80]
    x_pattern = ["▪", " ", "▪", " ", "▪", " ", "▪", " ", "▪"]
    o_pattern = ["▪", "▪", "▪", "▪", " ", "▪", "▪", "▪", "▪"]
    t_pattern = [" ", " ", " ", " ", "▪", " ", " ", " ", " "]

    for i, board_result in enumerate(self.meta_state):
      if board_result == 1:
        state_copy[i*9:i*9+9] = x_pattern
      elif board_result == -1:
        state_copy[i*9:i*9+9] = o_pattern
      elif board_result == 2:
        state_copy[i*9:i*9+9] = t_pattern

    state_strings = [UTTTGame.state_to_string(state_copy[i]) for i in index_order]

    out = ""
    i = 0

    for w in range(3):
      for z in range(3):
        for y in range(3):
          for x in range(3):
            out += state_strings[i]
            i += 1
          out += " "
        out += "\n"
      out += "\n"
    return out

  def copy(self):
    return UTTTGame(self.state.copy(), self.turn, self.previous_move, self.meta_state.copy())
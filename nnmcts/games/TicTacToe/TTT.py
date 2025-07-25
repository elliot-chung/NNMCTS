from enum import IntEnum
from nnmcts.games.Game import Game

class TTTGame(Game):
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

  def translate(self, move: Position) -> int:
    move = TTTGame.Position(move)
    return move.value

  def __init__(self, state=None, player_one_to_move=True, previous_move=None):
    super().__init__()
    if state is None:
      state = [0] * 9

    self.state = state
    self.turn = 1 if player_one_to_move else -1
    self.previous_move = previous_move

  def is_valid(self, move: Position) -> bool:
    try:
      TTTGame.Position(move)
    except ValueError:
      return False
    return self.state[move] == 0

  def valid_moves(self) -> list[Position]:
    return [i for i in range(9) if self.is_valid(i)]

  def is_terminal(self) -> bool:
    return self.get_winner() != 0 or len(self.valid_moves()) == 0

  def current_turn(self):
    return self.turn

  def get_state(self):
    return self.state

  def get_winner(self):
    occupant = self.state[self.Position.TOPLEFT]
    if occupant != 0:
      if self.state[self.Position.TOPCENTER] == occupant and self.state[self.Position.TOPRIGHT] == occupant:
        return occupant
      if self.state[self.Position.MIDDLELEFT] == occupant and self.state[self.Position.BOTTOMLEFT] == occupant:
        return occupant

    occupant = self.state[self.Position.BOTTOMRIGHT]
    if occupant != 0:
      if self.state[self.Position.BOTTOMCENTER] == occupant and self.state[self.Position.BOTTOMLEFT] == occupant:
        return occupant
      if self.state[self.Position.MIDDLERIGHT] == occupant and self.state[self.Position.TOPRIGHT] == occupant:
        return occupant

    occupant = self.state[self.Position.MIDDLECENTER]
    if occupant != 0:
      if self.state[self.Position.MIDDLELEFT] == occupant and self.state[self.Position.MIDDLERIGHT] == occupant:
        return occupant
      if self.state[self.Position.TOPLEFT] == occupant and self.state[self.Position.BOTTOMRIGHT] == occupant:
        return occupant
      if self.state[self.Position.TOPCENTER] == occupant and self.state[self.Position.BOTTOMCENTER] == occupant:
        return occupant
      if self.state[self.Position.TOPRIGHT] == occupant and self.state[self.Position.BOTTOMLEFT] == occupant:
        return occupant

    return 0

  def make_move(self, move: Position):
    self.state[move] = self.turn
    self.turn = -self.turn
    self.previous_move = move
    return self
  
  def get_mask(self) -> list[int]:
    pos_mask = [0] * len(self.state)
    for mov in self.valid_moves():
      ind = self.translate(mov)
      pos_mask[ind] = 1
    return pos_mask

  def get_canonical_state(self) -> list[Position]:
    norm_state = [s * self.turn for s in self.state]
    return norm_state


  def state_to_string(state):
    if state == 0:
      return "□"
    elif state == 1:
      return "X"
    elif state == -1:
      return "O"
    else:
      raise ValueError("Invalid state")

  def __repr__(self):
    return str(self)

  def __str__(self):
    state_strings = [TTTGame.state_to_string(s) for s in self.state]
    out = ""
    i = 0

    for _ in range(3):
      for _ in range(3):
        out += state_strings[i]
        i += 1
      out += "\n"
    return out

  def copy(self):
    return TTTGame(self.state.copy(), (self.turn == 1), self.previous_move)

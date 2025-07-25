from nnmcts.games.Game import Game
from nnmcts.players.players import Player

class Arena:
  def __init__(self, environment: Game, player_one: Player, player_two: Player):
    self.environment = environment

    self.player_one = player_one
    self.player_two = player_two

    self.record = {
      "player_one": {
          "states": [],
          "policies": []
      },
      "player_two": {
          "states": [],
          "policies": []
      },
      "winner": None
    }

  def play_game(self, show=False, record=False):
    if show: print(self.environment)
    while not self.environment.is_terminal():
      action1, policy1 = self.player_one.play_turn()
      action2, policy2 = self.player_two.play_turn()

      action = action1 if action1 != None else action2
      policy = policy1 if policy1 != None else policy2

      if record:
        player = "player_one" if action1 != None else "player_two"
        self.record[player]["states"].append(self.environment.get_canonical_state())
        self.record[player]["policies"].append(policy)

      self.environment.make_move(action)
      if show: print(self.environment)

    winner = self.environment.get_winner()
    if show: print(f"Winner: {winner}")
    if record:
      self.record["winner"] = winner
      return winner, self.record
    
    return winner





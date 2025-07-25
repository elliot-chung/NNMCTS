from nnmcts.arena import create_arena
from nnmcts.mcts.nodes import NeuralNode

from functools import reduce

winners = []

# NeuralNode.set_model(model, build_tensor)

for i in range(100):
  print(f"Game {i+1}")
  arena = create_arena('TTT', 'mcts_1000', 'mcts_1000')
  winner = arena.play_game(show=False, record=False)
  winners.append(winner)


winrate1 = reduce(lambda x, y: x + int(y ==  1), winners, 0) / len(winners)
tierate  = reduce(lambda x, y: x + int(y ==  0), winners, 0) / len(winners)
winrate2 = reduce(lambda x, y: x + int(y == -1), winners, 0) / len(winners)

print(f"Player One winrate: {winrate1}")
print(f"Average tierate: {tierate}")
print(f"Player Two Winrate: {winrate2}")


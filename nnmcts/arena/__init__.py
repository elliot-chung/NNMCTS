

from nnmcts.arena.Arena import Arena

def create_arena(game_type='UTTT', player_one_type='human', player_two_type='mcts'):
  if game_type == 'UTTT':
    from nnmcts.games.UltimateTicTacToe.UTTT import UTTTGame
    environment = UTTTGame()
  elif game_type == 'TTT':
    from nnmcts.games.TicTacToe.TTT import TTTGame
    environment = TTTGame()
  else:
    raise ValueError("Game type does not exist")

  if player_one_type == 'random':
    from nnmcts.players.players import Random_Player
    player_one = Random_Player(environment, True)
  elif player_one_type.startswith('mcts'):
    from nnmcts.players.players import MCTS_Player
    iter_count = int(player_one_type.split('_')[1]) if player_one_type != 'mcts' else 100
    player_one = MCTS_Player(environment, True, iter_count)
  elif player_one_type == 'human':
    from nnmcts.players.players import Human_Player
    player_one = Human_Player(environment, True)
  elif player_one_type.startswith('nmcts'):
    from nnmcts.players.players import Neural_MCTS_Player
    iter_count = int(player_one_type.split('_')[1]) if player_one_type != 'nmcts' else 100
    player_one = Neural_MCTS_Player(environment, True, iter_count)
  else:
    raise ValueError("Player one type does not exist")

  if player_two_type == 'random':
    from nnmcts.players.players import Random_Player
    player_two = Random_Player(environment, False)
  elif player_two_type.startswith('mcts'):
    from nnmcts.players.players import MCTS_Player
    iter_count = int(player_two_type.split('_')[1]) if player_two_type != 'mcts' else 100
    player_two = MCTS_Player(environment, False, iter_count)
  elif player_two_type == 'human':
    from nnmcts.players.players import Human_Player
    player_two = Human_Player(environment, False)
  elif player_two_type.startswith('nmcts'):
    from nnmcts.players.players import Neural_MCTS_Player
    iter_count = int(player_two_type.split('_')[1]) if player_two_type != 'nmcts' else 100
    player_two = Neural_MCTS_Player(environment, False, iter_count)
  else:
    raise ValueError("Player two type does not exist")

  return Arena(environment, player_one, player_two)
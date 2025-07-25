
from nnmcts.mcts.nodes import Node, print_tree
      
def mcts(node: Node, iters=100, show_execution_time=False, pause_after_iter=None):
  execution_times = []

  for i in range(iters):
    perf = {}
    node.explore(perf)
    execution_times.append(perf)
    if pause_after_iter != None and i >= pause_after_iter - 1:
      print_tree(node)
      input("Press Enter to continue")

  if show_execution_time:
    from functools import reduce
    total_times = reduce(lambda x, y: {"traverse_time": x["traverse_time"] + y["traverse_time"],
                         "rollout_time": x["rollout_time"] + y["rollout_time"],
                         "update_time": x["update_time"] + y["update_time"],
                         "create_time": x["create_time"] + y["create_time"]}, execution_times, {"traverse_time": 0, "rollout_time": 0, "update_time": 0, "create_time": 0})
    print(f"Traverse time: {total_times['traverse_time'] / iters:.4f}")
    print(f"Rollout time: {total_times['rollout_time'] / iters:.4f}")
    print(f"Update time: {total_times['update_time'] / iters:.4f}")
    print(f"Create time: {total_times['create_time'] / iters:.4f}")

  policy = node.get_policy()
  next_node = node.get_most_visited()
  # next_node.detach_parent()

  return next_node, policy

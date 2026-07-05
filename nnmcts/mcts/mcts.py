
from functools import reduce

from nnmcts.mcts.nodes import Node


def mcts(node: Node, iters=100, show_execution_time=False):
  execution_times = []

  for _ in range(iters):
    perf = {}
    node.explore(perf)
    execution_times.append(perf)

  if show_execution_time:
    total_times = reduce(
      lambda x, y: {
        "traverse_time": x["traverse_time"] + y["traverse_time"],
        "rollout_time": x["rollout_time"] + y["rollout_time"],
        "update_time": x["update_time"] + y["update_time"],
      },
      execution_times,
      {"traverse_time": 0, "rollout_time": 0, "update_time": 0},
    )
    print(f"Traverse time: {total_times['traverse_time'] / iters:.4f}")
    print(f"Rollout time: {total_times['rollout_time'] / iters:.4f}")
    print(f"Update time: {total_times['update_time'] / iters:.4f}")

  policy = node.get_policy()
  next_node = node.get_most_visited()

  return next_node, policy


def collect_mcts_timing(node: Node, iters: int) -> dict[str, float]:
  totals = {"traverse_time": 0.0, "rollout_time": 0.0, "update_time": 0.0}

  for _ in range(iters):
    perf = {}
    node.explore(perf)
    for key in totals:
      totals[key] += perf.get(key, 0.0)

  count = max(iters, 1)
  return {key: value / count for key, value in totals.items()}

import { NeuralNode, Node } from "./node";
import type { MctsEnvironment, MctsOptions, MctsResult } from "./types";

export function createRootNode<TMove, TEnv extends MctsEnvironment<TMove>>(
  environment: TEnv,
  useNeural = false,
): Node<TMove, TEnv> {
  const terminal = environment.isTerminal();
  return useNeural
    ? new NeuralNode<TMove, TEnv>(environment, terminal, null, null)
    : new Node<TMove, TEnv>(environment, terminal, null, null);
}

export async function mcts<TMove>(
  root: Node<TMove, MctsEnvironment<TMove>>,
  options: MctsOptions = {},
): Promise<MctsResult<TMove>> {
  const { iters = 100, showExecutionTime = false } = options;
  const executionTimes: Array<{
    traverseTime: number;
    rolloutTime: number;
    updateTime: number;
    createTime: number;
  }> = [];

  for (let i = 0; i < iters; i++) {
    const perf = {
      traverseTime: 0,
      rolloutTime: 0,
      updateTime: 0,
      createTime: 0,
    };
    await root.explore(perf);
    executionTimes.push(perf);
  }

  if (showExecutionTime) {
    const totals = executionTimes.reduce(
      (acc, perf) => ({
        traverseTime: acc.traverseTime + perf.traverseTime,
        rolloutTime: acc.rolloutTime + perf.rolloutTime,
        updateTime: acc.updateTime + perf.updateTime,
        createTime: acc.createTime + perf.createTime,
      }),
      {
        traverseTime: 0,
        rolloutTime: 0,
        updateTime: 0,
        createTime: 0,
      },
    );

    console.log(`Traverse time: ${(totals.traverseTime / iters).toFixed(4)}`);
    console.log(`Rollout time: ${(totals.rolloutTime / iters).toFixed(4)}`);
    console.log(`Update time: ${(totals.updateTime / iters).toFixed(4)}`);
    console.log(`Create time: ${(totals.createTime / iters).toFixed(4)}`);
  }

  const policy = root.getPolicy();
  const nextNode = root.getMostVisited();

  if (nextNode.action === null) {
    throw new Error("Most visited node has no action");
  }

  return {
    move: nextNode.action,
    policy,
  };
}

export { Node, NeuralNode } from "./node";
export type {
  ExplorePerf,
  MctsEnvironment,
  MctsOptions,
  MctsResult,
  UTTTEnvironment,
} from "./types";

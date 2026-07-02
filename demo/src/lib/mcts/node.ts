import { buildTensor } from "@/lib/model/build-tensor";
import { maskedSoftmax } from "@/lib/model/masked-softmax";
import { predict } from "@/lib/model/inference";
import type { ExplorePerf, MctsEnvironment } from "./types";

function pickRandom<T>(items: T[]): T {
  return items[Math.floor(Math.random() * items.length)];
}

export class Node<TMove, TEnv extends MctsEnvironment<TMove>> {
  totalReward = 0;
  visitCount = 0;
  child: Map<TMove, Node<TMove, TEnv>> | null = null;
  neuralPolicy: Float32Array | null = null;

  constructor(
    readonly environment: TEnv,
    readonly terminal: boolean,
    readonly parent: Node<TMove, TEnv> | null,
    readonly action: TMove | null,
  ) {}

  ucb(): number {
    if (this.visitCount === 0) {
      return Number.POSITIVE_INFINITY;
    }

    const parentNode = this.parent;
    if (!parentNode) {
      throw new Error("Root node has no parent for UCB");
    }

    return (
      this.totalReward / this.visitCount +
      Math.sqrt(Math.log(parentNode.visitCount) / this.visitCount)
    );
  }

  createChild(): void {
    if (this.terminal) {
      return;
    }

    const actions = this.environment.validMoves();
    const NodeVariant = this.constructor as typeof Node;
    const children = new Map<TMove, Node<TMove, TEnv>>();

    for (const action of actions) {
      const nextEnvironment = this.environment.copy().makeMove(action) as TEnv;
      children.set(
        action,
        new NodeVariant(
          nextEnvironment,
          nextEnvironment.isTerminal(),
          this,
          action,
        ),
      );
    }

    this.child = children;
  }

  async rollout(): Promise<number> {
    let newEnv = this.environment.copy();
    while (!newEnv.isTerminal()) {
      const actions = newEnv.validMoves();
      const action = pickRandom(actions);
      newEnv = newEnv.makeMove(action) as TEnv;
    }

    const reward = newEnv.getWinner() * this.environment.currentTurn();
    return -reward;
  }

  async explore(perf?: Partial<ExplorePerf>): Promise<void> {
    const traverseStart = performance.now();
    const current = Node.traverseToLeaf(this);
    const traverseTime = performance.now() - traverseStart;

    const rolloutStart = performance.now();
    const reward = await current.rollout();
    const rolloutTime = performance.now() - rolloutStart;

    current.totalReward += reward;
    current.visitCount += 1;

    const updateStart = performance.now();
    Node.updateParents(current, reward);
    const updateTime = performance.now() - updateStart;

    const createStart = performance.now();
    current.createChild();
    const createTime = performance.now() - createStart;

    if (perf) {
      perf.traverseTime = traverseTime;
      perf.rolloutTime = rolloutTime;
      perf.updateTime = updateTime;
      perf.createTime = createTime;
    }
  }

  getPolicy(): number[] {
    if (this.terminal) {
      throw new Error("Terminal node");
    }
    if (!this.child || this.child.size === 0) {
      throw new Error("No children");
    }

    const policy = Array(this.environment.getState().length).fill(0);
    for (const child of this.child.values()) {
      if (child.action === null) {
        continue;
      }
      policy[this.environment.translate(child.action)] = child.visitCount;
    }

    const sum = policy.reduce((acc, value) => acc + value, 0);
    return policy.map((value) => value / sum);
  }

  getMostVisited(): Node<TMove, TEnv> {
    if (this.terminal) {
      throw new Error("Terminal node");
    }
    if (!this.child || this.child.size === 0) {
      throw new Error("No children");
    }

    const children = [...this.child.values()];
    const maxVisit = Math.max(...children.map((child) => child.visitCount));
    const mostVisited = children.filter((child) => child.visitCount === maxVisit);
    return pickRandom(mostVisited);
  }

  detachParent(): void {
    (this as { parent: Node<TMove, TEnv> | null }).parent = null;
  }

  static traverseToLeaf<TMove, TEnv extends MctsEnvironment<TMove>>(
    node: Node<TMove, TEnv>,
  ): Node<TMove, TEnv> {
    let current = node;
    while (current.child && current.child.size > 0) {
      const ucbScores = new Map<TMove, number>();
      for (const [action, child] of current.child) {
        ucbScores.set(action, child.ucb());
      }

      const maxUcb = Math.max(...ucbScores.values());
      const bestActions = [...ucbScores.entries()]
        .filter(([, score]) => score === maxUcb)
        .map(([action]) => action);

      const action = pickRandom(bestActions);
      current = current.child.get(action)!;
    }

    return current;
  }

  static updateParents<TMove, TEnv extends MctsEnvironment<TMove>>(
    node: Node<TMove, TEnv>,
    reward: number,
  ): void {
    let flip = -1;
    let current: Node<TMove, TEnv> | null = node;

    while (current?.parent) {
      current = current.parent;
      current.visitCount += 1;
      current.totalReward += reward * flip;
      flip *= -1;
    }
  }
}

export class NeuralNode<
  TMove,
  TEnv extends MctsEnvironment<TMove>,
> extends Node<TMove, TEnv> {
  override async rollout(): Promise<number> {
    if (this.environment.isTerminal()) {
      return -(this.environment.getWinner() * this.environment.currentTurn());
    }

    const tensor = buildTensor(this);
    const { policyLogits, value } = await predict(tensor);
    const mask = this.environment.getMask();
    this.neuralPolicy = maskedSoftmax(policyLogits, mask);

    return -value;
  }

  ucb(): number {
    if (this.visitCount === 0) {
      return Number.POSITIVE_INFINITY;
    }

    const parentNode = this.parent;
    if (!parentNode?.neuralPolicy) {
      throw new Error("NeuralNode parent is missing neural policy");
    }

    if (this.action === null) {
      throw new Error("NeuralNode child is missing action");
    }

    const valueScore = this.totalReward / this.visitCount;
    const actionIndex = this.environment.translate(this.action);
    const explorationScore =
      parentNode.neuralPolicy[actionIndex] *
      Math.sqrt(Math.log(parentNode.visitCount) / this.visitCount);

    return valueScore + explorationScore;
  }
}

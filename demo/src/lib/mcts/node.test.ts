import { describe, expect, it, vi } from "vitest";

import { createRootNode, mcts, NeuralNode, Node } from "./mcts";
import { predict, setModel } from "@/lib/model/inference";
import { maskedSoftmax } from "@/lib/model/masked-softmax";
import { UTTTGame, type Move } from "@/lib/uttt";

function createChild(
  parent: Node<Move, UTTTGame>,
  move: Move,
  visitCount: number,
  totalReward = 0,
): Node<Move, UTTTGame> {
  const environment = parent.environment.copy().makeMove(move);
  const child = new Node<Move, UTTTGame>(
    environment,
    environment.isTerminal(),
    parent,
    move,
  );
  child.visitCount = visitCount;
  child.totalReward = totalReward;
  return child;
}

describe("Node UCB", () => {
  it("returns infinity when visit count is zero", () => {
    const game = new UTTTGame();
    const root = createRootNode<Move, UTTTGame>(game);
    root.visitCount = 5;

    const child = createChild(root, [0, 0], 0);
    expect(child.ucb()).toBe(Number.POSITIVE_INFINITY);
  });

  it("uses average reward plus exploration for standard MCTS", () => {
    const game = new UTTTGame();
    const root = createRootNode<Move, UTTTGame>(game);
    root.visitCount = 100;

    const child = createChild(root, [0, 0], 10, 6);
    const expected = 6 / 10 + Math.sqrt(Math.log(100) / 10);

    expect(child.ucb()).toBeCloseTo(expected);
  });
});

describe("NeuralNode UCB", () => {
  it("uses parent neural policy for exploration", () => {
    const game = new UTTTGame();
    const root = createRootNode<Move, UTTTGame>(game, true);
    root.visitCount = 10;

    const neuralPolicy = new Float32Array(81);
    neuralPolicy[5] = 0.5;
    root.neuralPolicy = neuralPolicy;

    const childEnv = game.copy().makeMove([0, 5]);
    const child = new NeuralNode<Move, UTTTGame>(
      childEnv,
      false,
      root,
      [0, 5],
    );
    child.visitCount = 2;
    child.totalReward = 1;

    const expected = 1 / 2 + 0.5 * Math.sqrt(Math.log(10) / 2);

    expect(child.ucb()).toBeCloseTo(expected);
  });
});

describe("Node policy and move selection", () => {
  it("normalizes visit counts into a policy", () => {
    const game = new UTTTGame();
    const root = createRootNode<Move, UTTTGame>(game);
    root.child = new Map([
      [[0, 0], createChild(root, [0, 0], 3)],
      [[0, 1], createChild(root, [0, 1], 1)],
    ]);

    const policy = root.getPolicy();
    expect(policy[0]).toBeCloseTo(0.75);
    expect(policy[1]).toBeCloseTo(0.25);
  });

  it("selects the most visited child", () => {
    const game = new UTTTGame();
    const root = createRootNode<Move, UTTTGame>(game);
    const mostVisited = createChild(root, [1, 1], 7);
    root.child = new Map([
      [[0, 0], createChild(root, [0, 0], 2)],
      [[1, 1], mostVisited],
      [[2, 2], createChild(root, [2, 2], 4)],
    ]);

    expect(root.getMostVisited()).toBe(mostVisited);
  });
});

describe("mcts", () => {
  it("returns a legal move and normalized policy", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    const game = new UTTTGame();
    const root = createRootNode<Move, UTTTGame>(game);
    const { move, policy } = await mcts(root, { iters: 25 });

    expect(game.validMoves().length).toBeGreaterThan(0);
    expect(move).toBeDefined();
    expect(game.copy().makeMove(move).getState()).not.toEqual(game.getState());
    expect(policy).toHaveLength(81);
    expect(policy.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);

    vi.restoreAllMocks();
  });

  it("uses neural rollout when a model is loaded", async () => {
    const logits = new Float32Array(81).fill(-1);
    logits[4] = 2;
    setModel({
      predict: () => ({
        policyLogits: logits,
        value: 0.25,
      }),
    });

    const game = new UTTTGame();
    const root = createRootNode<Move, UTTTGame>(game, true);
    await root.explore();

    expect(root.neuralPolicy).not.toBeNull();
    expect(root.neuralPolicy?.[4]).toBeGreaterThan(0);
    expect(root.neuralPolicy).toEqual(
      maskedSoftmax(logits, new Float32Array(game.getMask())),
    );

    setModel(null);
  });
});

describe("model inference stub", () => {
  it("throws when model is not loaded", async () => {
    setModel(null);
    await expect(predict(new Float32Array(162))).rejects.toThrow(
      "Model not loaded",
    );
  });
});

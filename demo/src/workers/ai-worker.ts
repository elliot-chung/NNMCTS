import { createRootNode, mcts } from "@/lib/mcts";
import { loadModel, setModel } from "@/lib/model/inference";
import { modelFileExists } from "@/lib/model/loader";
import { UTTTGame, type Move, type Player } from "@/lib/uttt";

export interface SerializedGame {
  state: number[];
  turn: Player;
  previousMove: Move | null;
  metaState: number[];
}

export type WorkerRequest =
  | { type: "init"; modelUrl?: string }
  | {
      type: "think";
      requestId: string;
      game: SerializedGame;
      iterations: number;
      useNeural?: boolean;
    }
  | { type: "cancel"; requestId?: string };

export type WorkerResponse =
  | { type: "loading" }
  | { type: "ready"; useNeural: boolean; message?: string }
  | { type: "thinking"; requestId: string }
  | {
      type: "move";
      requestId: string;
      move: Move;
      policy: number[];
    }
  | { type: "error"; requestId?: string; message: string };

let activeRequestId: string | null = null;
let cancelled = false;

function deserializeGame(game: SerializedGame): UTTTGame {
  return new UTTTGame(
    game.state,
    game.turn,
    game.previousMove,
    game.metaState,
  );
}

async function initializeModel(modelUrl?: string): Promise<{
  useNeural: boolean;
  message?: string;
}> {
  if (!modelUrl) {
    return {
      useNeural: false,
      message: "No model URL configured — using pure MCTS",
    };
  }

  const exists = await modelFileExists(modelUrl);
  if (!exists) {
    return {
      useNeural: false,
      message: "ONNX model not found — using pure MCTS",
    };
  }

  try {
    const model = await loadModel(modelUrl);
    setModel(model);
    return { useNeural: true };
  } catch (error) {
    setModel(null);
    const message =
      error instanceof Error ? error.message : "Failed to load ONNX model";
    return {
      useNeural: false,
      message: `${message} — using pure MCTS`,
    };
  }
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  try {
    switch (message.type) {
      case "init": {
        self.postMessage({ type: "loading" } satisfies WorkerResponse);
        setModel(null);
        const initResult = await initializeModel(message.modelUrl);
        self.postMessage({
          type: "ready",
          useNeural: initResult.useNeural,
          message: initResult.message,
        } satisfies WorkerResponse);
        break;
      }

      case "cancel": {
        if (
          message.requestId === undefined ||
          message.requestId === activeRequestId
        ) {
          cancelled = true;
        }
        break;
      }

      case "think": {
        activeRequestId = message.requestId;
        cancelled = false;

        self.postMessage({
          type: "thinking",
          requestId: message.requestId,
        } satisfies WorkerResponse);

        const game = deserializeGame(message.game);
        const useNeural = message.useNeural ?? false;
        const root = createRootNode<Move, UTTTGame>(game, useNeural);

        if (cancelled || activeRequestId !== message.requestId) {
          return;
        }

        const { move, policy } = await mcts(root, { iters: message.iterations });

        if (cancelled || activeRequestId !== message.requestId) {
          return;
        }

        self.postMessage({
          type: "move",
          requestId: message.requestId,
          move,
          policy,
        } satisfies WorkerResponse);
        break;
      }

      default:
        self.postMessage({
          type: "error",
          message: "Unknown worker message type",
        } satisfies WorkerResponse);
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      requestId: message.type === "think" ? message.requestId : undefined,
      message: error instanceof Error ? error.message : String(error),
    } satisfies WorkerResponse);
  }
};

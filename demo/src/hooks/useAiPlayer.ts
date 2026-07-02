"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { MctsDifficulty } from "@/hooks/useGame";
import { UTTTGame, type Move } from "@/lib/uttt";
import type {
  SerializedGame,
  WorkerRequest,
  WorkerResponse,
} from "@/workers/ai-worker";

export interface AiMoveResult {
  move: Move;
  policy: number[];
}

export interface UseAiPlayerOptions {
  modelUrl?: string;
}

function serializeGame(game: UTTTGame): SerializedGame {
  return {
    state: [...game.state],
    turn: game.turn,
    previousMove: game.previousMove,
    metaState: [...game.metaState],
  };
}

export function useAiPlayer(options: UseAiPlayerOptions = {}) {
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  const pendingRef = useRef<{
    requestId: string;
    resolve: (result: AiMoveResult) => void;
    reject: (error: Error) => void;
  } | null>(null);

  const [isReady, setIsReady] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [useNeural, setUseNeural] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL("../workers/ai-worker.ts", import.meta.url));
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;

      switch (message.type) {
        case "loading":
          setIsReady(false);
          setUseNeural(false);
          setStatusMessage(null);
          setError(null);
          break;

        case "ready":
          setIsReady(true);
          setUseNeural(message.useNeural);
          setStatusMessage(message.message ?? null);
          break;

        case "thinking":
          if (pendingRef.current?.requestId === message.requestId) {
            setIsThinking(true);
          }
          break;

        case "move": {
          const pending = pendingRef.current;
          if (!pending || pending.requestId !== message.requestId) {
            return;
          }

          pendingRef.current = null;
          setIsThinking(false);
          pending.resolve({
            move: message.move,
            policy: message.policy,
          });
          break;
        }

        case "error": {
          const pending = pendingRef.current;
          if (
            message.requestId !== undefined &&
            pending?.requestId !== message.requestId
          ) {
            return;
          }

          pendingRef.current = null;
          setIsThinking(false);
          setError(message.message);

          if (pending) {
            pending.reject(new Error(message.message));
          }
          break;
        }
      }
    };

    worker.onerror = () => {
      setError("AI worker failed");
      setIsThinking(false);
      pendingRef.current?.reject(new Error("AI worker failed"));
      pendingRef.current = null;
    };

    worker.postMessage({
      type: "init",
      modelUrl: options.modelUrl,
    } satisfies WorkerRequest);

    return () => {
      worker.terminate();
      workerRef.current = null;
      pendingRef.current = null;
    };
  }, [options.modelUrl]);

  const cancel = useCallback(() => {
    const worker = workerRef.current;
    const pending = pendingRef.current;

    if (worker && pending) {
      worker.postMessage({
        type: "cancel",
        requestId: pending.requestId,
      } satisfies WorkerRequest);
    }

    pendingRef.current = null;
    setIsThinking(false);
  }, []);

  const think = useCallback(
    (game: UTTTGame, iterations: MctsDifficulty): Promise<AiMoveResult> => {
      const worker = workerRef.current;
      if (!worker || !isReady) {
        return Promise.reject(new Error("AI worker is not ready"));
      }

      cancel();

      const requestId = String(++requestIdRef.current);

      return new Promise<AiMoveResult>((resolve, reject) => {
        pendingRef.current = { requestId, resolve, reject };
        setError(null);
        setIsThinking(true);

        worker.postMessage({
          type: "think",
          requestId,
          game: serializeGame(game),
          iterations,
          useNeural,
        } satisfies WorkerRequest);
      });
    },
    [cancel, isReady, useNeural],
  );

  return {
    think,
    cancel,
    isReady,
    isThinking,
    useNeural,
    statusMessage,
    error,
  };
}

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useAiPlayer } from "@/hooks/useAiPlayer";
import { getUtttModelEntry } from "@/lib/model/loader";
import { UTTTGame, type Move, type Player } from "@/lib/uttt";

export const MCTS_TIME_LIMITS = [0.01, 0.1, 1, 3, 5] as const;
export type MctsTimeLimit = (typeof MCTS_TIME_LIMITS)[number];

export type GamePhase = "playing" | "ai_thinking" | "finished";

export type ModelLoadState =
  | { status: "loading" }
  | { status: "ready"; useNeural: true }
  | { status: "ready"; useNeural: false; message: string }
  | { status: "error"; message: string };

export interface GameConfig {
  humanSide: Player;
  searchTimeLimit: MctsTimeLimit;
}

function moveToIndex(move: Move): number {
  return UTTTGame.translate(move);
}

function buildLegalMoveSet(game: UTTTGame): Set<number> {
  return new Set(game.validMoves().map(moveToIndex));
}

function getForcedBoard(game: UTTTGame): number | null {
  if (game.previousMove === null) {
    return null;
  }
  const boardId = game.previousMove[1];
  return game.metaState[boardId] === 0 ? boardId : null;
}

export function useGame(initialConfig?: Partial<GameConfig>) {
  const [humanSide, setHumanSide] = useState<Player>(
    initialConfig?.humanSide ?? 1,
  );
  const [searchTimeLimit, setSearchTimeLimit] = useState<MctsTimeLimit>(
    initialConfig?.searchTimeLimit ?? 1,
  );
  const [game, setGame] = useState(() => new UTTTGame());
  const [phase, setPhase] = useState<GamePhase>("playing");
  const [modelUrl, setModelUrl] = useState<string | undefined>();
  const [manifestError, setManifestError] = useState<string | null>(null);

  const aiRequestRef = useRef(0);
  const pendingAiRef = useRef<{
    game: UTTTGame;
    searchTimeLimit: MctsTimeLimit;
  } | null>(null);
  const runAiTurnRef = useRef<
    (currentGame: UTTTGame, currentSearchTimeLimit: MctsTimeLimit) => void
  >(() => {});

  const ai = useAiPlayer({ modelUrl });

  useEffect(() => {
    let cancelled = false;

    async function loadManifest() {
      try {
        const { fetchManifest } = await import("@/lib/model/loader");
        const manifest = await fetchManifest();
        if (cancelled) {
          return;
        }
        const entry = getUtttModelEntry(manifest);
        setModelUrl(entry?.onnxPath);
        setManifestError(null);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setManifestError(
          error instanceof Error ? error.message : "Failed to load manifest",
        );
        setModelUrl(undefined);
      }
    }

    void loadManifest();

    return () => {
      cancelled = true;
    };
  }, []);

  const modelLoad = useMemo((): ModelLoadState => {
    if (manifestError) {
      return { status: "error", message: manifestError };
    }
    if (!ai.isReady) {
      return { status: "loading" };
    }
    if (ai.useNeural) {
      return { status: "ready", useNeural: true };
    }
    return {
      status: "ready",
      useNeural: false,
      message: ai.statusMessage ?? "Neural model unavailable — using pure MCTS",
    };
  }, [ai.isReady, ai.statusMessage, ai.useNeural, manifestError]);

  const aiSide = (humanSide * -1) as Player;
  const winner = game.getWinner();
  const forcedBoard = getForcedBoard(game);
  const legalMoves = useMemo(() => buildLegalMoveSet(game), [game]);
  const isHumanTurn = game.currentTurn() === humanSide && phase === "playing";

  const runAiTurn = useCallback(
    async (currentGame: UTTTGame, currentSearchTimeLimit: MctsTimeLimit) => {
      const requestId = ++aiRequestRef.current;
      setPhase("ai_thinking");

      const continueAiChain = (nextGame: UTTTGame) => {
        if (nextGame.isTerminal()) {
          setPhase("finished");
        } else if (nextGame.currentTurn() === humanSide) {
          setPhase("playing");
        } else {
          runAiTurnRef.current(nextGame, currentSearchTimeLimit);
        }
      };

      try {
        const { move } = await ai.think(currentGame, currentSearchTimeLimit);
        if (requestId !== aiRequestRef.current) {
          return;
        }

        const nextGame = currentGame.makeMove(move);
        setGame(nextGame);
        continueAiChain(nextGame);
      } catch (error) {
        if (requestId !== aiRequestRef.current) {
          return;
        }

        if (!currentGame.isTerminal()) {
          const nextGame = currentGame.makeRandomMove();
          setGame(nextGame);
          continueAiChain(nextGame);
        } else {
          setPhase("finished");
        }

        console.warn(
          "AI think failed, used random fallback:",
          error instanceof Error ? error.message : error,
        );
      }
    },
    [ai, humanSide],
  );

  useEffect(() => {
    runAiTurnRef.current = runAiTurn;
  }, [runAiTurn]);

  const startPendingAiTurn = useCallback(() => {
    const pending = pendingAiRef.current;
    if (!pending || !ai.isReady) {
      return;
    }
    pendingAiRef.current = null;
    void runAiTurn(pending.game, pending.searchTimeLimit);
  }, [ai.isReady, runAiTurn]);

  const scheduleAiMove = useCallback(
    (currentGame: UTTTGame, currentSearchTimeLimit: MctsTimeLimit) => {
      pendingAiRef.current = {
        game: currentGame,
        searchTimeLimit: currentSearchTimeLimit,
      };
      setPhase("ai_thinking");
      startPendingAiTurn();
    },
    [startPendingAiTurn],
  );

  useEffect(() => {
    startPendingAiTurn();
  }, [ai.isReady, startPendingAiTurn]);

  const newGame = useCallback(
    (config?: Partial<GameConfig>) => {
      ai.cancel();
      aiRequestRef.current += 1;
      pendingAiRef.current = null;

      const nextHumanSide = config?.humanSide ?? humanSide;
      const nextSearchTimeLimit = config?.searchTimeLimit ?? searchTimeLimit;

      if (config?.humanSide !== undefined) {
        setHumanSide(config.humanSide);
      }
      if (config?.searchTimeLimit !== undefined) {
        setSearchTimeLimit(config.searchTimeLimit);
      }

      const freshGame = new UTTTGame();
      setGame(freshGame);

      if (nextHumanSide === -1) {
        scheduleAiMove(freshGame, nextSearchTimeLimit);
      } else {
        setPhase("playing");
      }
    },
    [ai, humanSide, scheduleAiMove, searchTimeLimit],
  );

  const makeHumanMove = useCallback(
    (boardId: number, cellId: number) => {
      if (phase !== "playing" || game.currentTurn() !== humanSide) {
        return;
      }

      const move: Move = [boardId, cellId];
      if (!legalMoves.has(moveToIndex(move))) {
        return;
      }

      ai.cancel();
      aiRequestRef.current += 1;
      pendingAiRef.current = null;

      const nextGame = game.makeMove(move);
      setGame(nextGame);

      if (nextGame.isTerminal()) {
        setPhase("finished");
        return;
      }

      scheduleAiMove(nextGame, searchTimeLimit);
    },
    [ai, game, humanSide, legalMoves, phase, scheduleAiMove, searchTimeLimit],
  );

  return {
    game,
    phase,
    humanSide,
    aiSide,
    searchTimeLimit,
    forcedBoard,
    legalMoves,
    isHumanTurn,
    winner,
    modelLoad,
    aiError: ai.error,
    makeHumanMove,
    newGame,
    setHumanSide,
    setSearchTimeLimit,
  };
}

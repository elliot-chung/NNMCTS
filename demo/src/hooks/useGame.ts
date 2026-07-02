"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useAiPlayer } from "@/hooks/useAiPlayer";
import { getUtttModelEntry } from "@/lib/model/loader";
import { UTTTGame, type Move, type Player } from "@/lib/uttt";

export const MCTS_ITERATIONS = [10, 25, 50, 100] as const;
export type MctsDifficulty = (typeof MCTS_ITERATIONS)[number];

export type GamePhase = "playing" | "ai_thinking" | "finished";

export type ModelLoadState =
  | { status: "loading" }
  | { status: "ready"; useNeural: true }
  | { status: "ready"; useNeural: false; message: string }
  | { status: "error"; message: string };

export interface GameConfig {
  humanSide: Player;
  difficulty: MctsDifficulty;
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
  const [difficulty, setDifficulty] = useState<MctsDifficulty>(
    initialConfig?.difficulty ?? 25,
  );
  const [game, setGame] = useState(() => new UTTTGame());
  const [phase, setPhase] = useState<GamePhase>("playing");
  const [modelUrl, setModelUrl] = useState<string | undefined>();
  const [manifestError, setManifestError] = useState<string | null>(null);

  const aiRequestRef = useRef(0);
  const pendingAiRef = useRef<{
    game: UTTTGame;
    difficulty: MctsDifficulty;
  } | null>(null);
  const runAiTurnRef = useRef<
    (currentGame: UTTTGame, currentDifficulty: MctsDifficulty) => void
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
      message:
        ai.statusMessage ?? "Neural model unavailable — using pure MCTS",
    };
  }, [ai.isReady, ai.statusMessage, ai.useNeural, manifestError]);

  const aiSide = (humanSide * -1) as Player;
  const winner = game.getWinner();
  const forcedBoard = getForcedBoard(game);
  const legalMoves = useMemo(() => buildLegalMoveSet(game), [game]);
  const isHumanTurn = game.currentTurn() === humanSide && phase === "playing";

  const runAiTurn = useCallback(
    async (currentGame: UTTTGame, currentDifficulty: MctsDifficulty) => {
      const requestId = ++aiRequestRef.current;
      setPhase("ai_thinking");

      const continueAiChain = (nextGame: UTTTGame) => {
        if (nextGame.isTerminal()) {
          setPhase("finished");
        } else if (nextGame.currentTurn() === humanSide) {
          setPhase("playing");
        } else {
          runAiTurnRef.current(nextGame, currentDifficulty);
        }
      };

      try {
        const { move } = await ai.think(currentGame, currentDifficulty);
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
    void runAiTurn(pending.game, pending.difficulty);
  }, [ai.isReady, runAiTurn]);

  const scheduleAiMove = useCallback(
    (currentGame: UTTTGame, currentDifficulty: MctsDifficulty) => {
      pendingAiRef.current = { game: currentGame, difficulty: currentDifficulty };
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
      const nextDifficulty = config?.difficulty ?? difficulty;

      if (config?.humanSide !== undefined) {
        setHumanSide(config.humanSide);
      }
      if (config?.difficulty !== undefined) {
        setDifficulty(config.difficulty);
      }

      const freshGame = new UTTTGame();
      setGame(freshGame);

      if (nextHumanSide === -1) {
        scheduleAiMove(freshGame, nextDifficulty);
      } else {
        setPhase("playing");
      }
    },
    [ai, difficulty, humanSide, scheduleAiMove],
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

      scheduleAiMove(nextGame, difficulty);
    },
    [ai, difficulty, game, humanSide, legalMoves, phase, scheduleAiMove],
  );

  return {
    game,
    phase,
    humanSide,
    aiSide,
    difficulty,
    forcedBoard,
    legalMoves,
    isHumanTurn,
    winner,
    modelLoad,
    aiError: ai.error,
    makeHumanMove,
    newGame,
    setHumanSide,
    setDifficulty,
  };
}

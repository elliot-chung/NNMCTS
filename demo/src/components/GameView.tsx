"use client";

import { Board } from "@/components/game/Board";
import { GameStatus } from "@/components/game/GameStatus";
import { DifficultySlider } from "@/components/controls/DifficultySlider";
import { ModeSelector } from "@/components/controls/ModeSelector";
import { NewGameButton } from "@/components/controls/NewGameButton";
import { SideSelector } from "@/components/controls/SideSelector";
import { Header } from "@/components/layout/Header";
import { ModelStatus } from "@/components/layout/ModelStatus";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useGame } from "@/hooks/useGame";

export function GameView() {
  const {
    game,
    phase,
    humanSide,
    searchTimeLimit,
    aiMode,
    forcedBoard,
    legalMoves,
    isHumanTurn,
    winner,
    modelLoad,
    makeHumanMove,
    newGame,
    setHumanSide,
    setSearchTimeLimit,
    setAiMode,
  } = useGame();

  const boardDisabled = !isHumanTurn || phase === "ai_thinking";
  const settingsDisabled = phase === "ai_thinking";
  const policyOnlyAvailable =
    modelLoad.status === "ready" && modelLoad.useNeural;

  return (
    <div className="flex min-h-full flex-col bg-background">
      <Header />

      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-6 px-4 py-6 sm:px-6 lg:flex-row lg:items-start">
        <section className="flex flex-1 flex-col items-center gap-4">
          <Board
            state={game.getState()}
            metaState={game.metaState}
            legalMoves={legalMoves}
            forcedBoard={forcedBoard}
            disabled={boardDisabled}
            onCellClick={makeHumanMove}
          />
          <GameStatus
            currentTurn={game.currentTurn()}
            humanSide={humanSide}
            forcedBoard={forcedBoard}
            winner={winner}
            phase={phase}
          />
        </section>

        <aside className="flex w-full flex-col gap-4 lg:w-72 lg:shrink-0">
          <Card size="sm">
            <CardHeader>
              <CardTitle>Game Controls</CardTitle>
              <CardDescription>Configure side, mode, and search time</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <SideSelector
                value={humanSide}
                onChange={setHumanSide}
                disabled={settingsDisabled}
              />
              <ModeSelector
                value={aiMode}
                onChange={setAiMode}
                disabled={settingsDisabled}
                policyOnlyDisabled={!policyOnlyAvailable}
              />
              {aiMode === "mcts" && (
                <DifficultySlider
                  value={searchTimeLimit}
                  onChange={setSearchTimeLimit}
                  disabled={settingsDisabled}
                />
              )}
              <NewGameButton
                onNewGame={() =>
                  newGame({ humanSide, searchTimeLimit, aiMode })
                }
                disabled={settingsDisabled}
              />
            </CardContent>
          </Card>

          <ModelStatus modelLoad={modelLoad} aiMode={aiMode} />
        </aside>
      </main>
    </div>
  );
}

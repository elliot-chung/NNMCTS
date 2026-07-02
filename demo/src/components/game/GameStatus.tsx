"use client";

import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import type { GamePhase } from "@/hooks/useGame";
import type { Player } from "@/lib/uttt";

interface GameStatusProps {
  currentTurn: Player;
  humanSide: Player;
  forcedBoard: number | null;
  winner: number;
  phase: GamePhase;
}

function playerLabel(side: Player): string {
  return side === 1 ? "X" : "O";
}

function boardLabel(boardId: number): string {
  const row = Math.floor(boardId / 3) + 1;
  const col = (boardId % 3) + 1;
  return `row ${row}, col ${col}`;
}

export function GameStatus({
  currentTurn,
  humanSide,
  forcedBoard,
  winner,
  phase,
}: GameStatusProps) {
  const aiSide = (humanSide * -1) as Player;

  if (phase === "ai_thinking") {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Spinner className="size-4" />
        <span>AI ({playerLabel(aiSide)}) is thinking…</span>
      </div>
    );
  }

  if (winner !== 0) {
    const humanWon = winner === humanSide;
    return (
      <Badge variant={humanWon ? "default" : "secondary"}>
        {humanWon ? "You win!" : "AI wins!"}
      </Badge>
    );
  }

  if (phase === "finished") {
    return <Badge variant="outline">Draw</Badge>;
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-muted-foreground">Turn:</span>
        <Badge variant="outline">{playerLabel(currentTurn)}</Badge>
        <span className="text-sm text-muted-foreground">
          ({currentTurn === humanSide ? "You" : "AI"})
        </span>
      </div>
      <p className="text-sm text-center text-muted-foreground">
        {forcedBoard === null
          ? "Play in any open mini-board"
          : `Forced mini-board: ${boardLabel(forcedBoard)}`}
      </p>
    </div>
  );
}

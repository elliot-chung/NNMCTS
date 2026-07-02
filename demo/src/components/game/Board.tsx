"use client";

import type { CellValue } from "@/lib/uttt";

import { MiniBoard } from "./MiniBoard";

interface BoardProps {
  state: readonly number[];
  metaState: readonly number[];
  legalMoves: Set<number>;
  forcedBoard: number | null;
  disabled: boolean;
  onCellClick: (boardId: number, cellId: number) => void;
}

export function Board({
  state,
  metaState,
  legalMoves,
  forcedBoard,
  disabled,
  onCellClick,
}: BoardProps) {
  return (
    <div className="grid w-full max-w-md grid-cols-3 overflow-hidden rounded-lg border-4 border-foreground sm:max-w-lg">
      {Array.from({ length: 9 }, (_, boardId) => (
        <MiniBoard
          key={boardId}
          boardId={boardId}
          cells={state.slice(boardId * 9, boardId * 9 + 9) as CellValue[]}
          legalMoves={legalMoves}
          isFinished={metaState[boardId] !== 0}
          isForced={forcedBoard === boardId}
          disabled={disabled}
          onCellClick={onCellClick}
        />
      ))}
    </div>
  );
}

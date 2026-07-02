"use client";

import { cn } from "@/lib/utils";
import type { CellValue } from "@/lib/uttt";

import { Cell } from "./Cell";

interface MiniBoardProps {
  boardId: number;
  cells: readonly CellValue[];
  legalMoves: Set<number>;
  isFinished: boolean;
  isForced: boolean;
  disabled: boolean;
  onCellClick: (boardId: number, cellId: number) => void;
}

export function MiniBoard({
  boardId,
  cells,
  legalMoves,
  isFinished,
  isForced,
  disabled,
  onCellClick,
}: MiniBoardProps) {
  return (
    <div
      className={cn(
        "grid grid-cols-3 border-2 border-foreground/80",
        isFinished && "bg-muted/70",
        isForced && "ring-2 ring-primary ring-inset",
      )}
    >
      {cells.map((value, cellId) => {
        const flatIndex = boardId * 9 + cellId;
        const isLegal = legalMoves.has(flatIndex);

        return (
          <Cell
            key={cellId}
            value={value}
            isLegal={isLegal}
            disabled={disabled || isFinished}
            onClick={() => onCellClick(boardId, cellId)}
          />
        );
      })}
    </div>
  );
}

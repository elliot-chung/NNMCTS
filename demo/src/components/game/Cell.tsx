"use client";

import { cn } from "@/lib/utils";
import type { CellValue } from "@/lib/uttt";

interface CellProps {
  value: CellValue;
  isLegal: boolean;
  disabled: boolean;
  onClick: () => void;
}

function cellLabel(value: CellValue): string {
  if (value === 1) {
    return "X";
  }
  if (value === -1) {
    return "O";
  }
  return "";
}

export function Cell({ value, isLegal, disabled, onClick }: CellProps) {
  return (
    <button
      type="button"
      aria-label={value === 0 ? "Empty cell" : `Cell ${cellLabel(value)}`}
      disabled={disabled || !isLegal}
      onClick={onClick}
      className={cn(
        "flex aspect-square items-center justify-center border border-border/60 text-lg font-semibold transition-colors sm:text-xl",
        value === 1 && "text-primary",
        value === -1 && "text-muted-foreground",
        isLegal &&
          !disabled &&
          "cursor-pointer bg-primary/10 hover:bg-primary/20",
        (!isLegal || disabled) && "cursor-not-allowed opacity-60",
      )}
    >
      {cellLabel(value)}
    </button>
  );
}

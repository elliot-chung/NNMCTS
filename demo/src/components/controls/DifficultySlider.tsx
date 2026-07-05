"use client";

import { Slider } from "@/components/ui/slider";
import { MCTS_TIME_LIMITS, type MctsTimeLimit } from "@/hooks/useGame";

interface DifficultySliderProps {
  value: MctsTimeLimit;
  onChange: (value: MctsTimeLimit) => void;
  disabled?: boolean;
}

function timeLimitIndex(value: MctsTimeLimit): number {
  return MCTS_TIME_LIMITS.indexOf(value);
}

function formatTimeLimit(seconds: MctsTimeLimit): string {
  return `${seconds}s`;
}

export function DifficultySlider({
  value,
  onChange,
  disabled,
}: DifficultySliderProps) {
  const index = timeLimitIndex(value);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <label className="text-sm font-medium">MCTS search time</label>
        <span className="text-sm text-muted-foreground">
          {formatTimeLimit(value)}
        </span>
      </div>
      <Slider
        min={0}
        max={MCTS_TIME_LIMITS.length - 1}
        step={1}
        value={[index]}
        disabled={disabled}
        onValueChange={(next) => {
          const values = Array.isArray(next) ? next : [next];
          const nextIndex = values[0] ?? 0;
          onChange(MCTS_TIME_LIMITS[nextIndex] ?? MCTS_TIME_LIMITS[0]);
        }}
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        {MCTS_TIME_LIMITS.map((timeLimit) => (
          <span key={timeLimit}>{formatTimeLimit(timeLimit)}</span>
        ))}
      </div>
    </div>
  );
}

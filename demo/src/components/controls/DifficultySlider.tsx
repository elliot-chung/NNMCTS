"use client";

import { Slider } from "@/components/ui/slider";
import { MCTS_ITERATIONS, type MctsDifficulty } from "@/hooks/useGame";

interface DifficultySliderProps {
  value: MctsDifficulty;
  onChange: (value: MctsDifficulty) => void;
  disabled?: boolean;
}

function difficultyIndex(value: MctsDifficulty): number {
  return MCTS_ITERATIONS.indexOf(value);
}

export function DifficultySlider({
  value,
  onChange,
  disabled,
}: DifficultySliderProps) {
  const index = difficultyIndex(value);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <label className="text-sm font-medium">MCTS iterations</label>
        <span className="text-sm text-muted-foreground">{value}</span>
      </div>
      <Slider
        min={0}
        max={MCTS_ITERATIONS.length - 1}
        step={1}
        value={[index]}
        disabled={disabled}
        onValueChange={(next) => {
          const values = Array.isArray(next) ? next : [next];
          const nextIndex = values[0] ?? 0;
          onChange(MCTS_ITERATIONS[nextIndex] ?? MCTS_ITERATIONS[0]);
        }}
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        {MCTS_ITERATIONS.map((iterations) => (
          <span key={iterations}>{iterations}</span>
        ))}
      </div>
    </div>
  );
}

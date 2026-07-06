"use client";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { AiMode } from "@/hooks/useGame";

interface ModeSelectorProps {
  value: AiMode;
  onChange: (mode: AiMode) => void;
  disabled?: boolean;
  policyOnlyDisabled?: boolean;
}

export function ModeSelector({
  value,
  onChange,
  disabled,
  policyOnlyDisabled,
}: ModeSelectorProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm font-medium">AI mode</label>
      <Select
        value={value}
        onValueChange={(next) => onChange(next as AiMode)}
        disabled={disabled}
      >
        <SelectTrigger className="w-full">
          <SelectValue placeholder="Choose mode" />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value="mcts">Neural MCTS</SelectItem>
            <SelectItem value="policy" disabled={policyOnlyDisabled}>
              Policy network only
            </SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
      {policyOnlyDisabled && (
        <p className="text-xs text-muted-foreground">
          Policy-only mode requires the ONNX model.
        </p>
      )}
    </div>
  );
}

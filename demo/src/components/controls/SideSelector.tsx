"use client";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Player } from "@/lib/uttt";

interface SideSelectorProps {
  value: Player;
  onChange: (side: Player) => void;
  disabled?: boolean;
}

export function SideSelector({ value, onChange, disabled }: SideSelectorProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm font-medium">Your side</label>
      <Select
        value={String(value)}
        onValueChange={(next) => onChange(Number(next) as Player)}
        disabled={disabled}
      >
        <SelectTrigger className="w-full">
          <SelectValue placeholder="Choose side" />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value="1">X (first)</SelectItem>
            <SelectItem value="-1">O (second)</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
}

"use client";

import { Button } from "@/components/ui/button";

interface NewGameButtonProps {
  onNewGame: () => void;
  disabled?: boolean;
}

export function NewGameButton({ onNewGame, disabled }: NewGameButtonProps) {
  return (
    <Button type="button" onClick={onNewGame} disabled={disabled}>
      New Game
    </Button>
  );
}

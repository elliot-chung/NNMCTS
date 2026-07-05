export function Header() {
  return (
    <header className="border-b bg-card">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-1 px-4 py-4 sm:px-6">
        <h1 className="font-heading text-xl font-semibold tracking-tight sm:text-2xl">
          Ultimate Tic-Tac-Toe vs AI
        </h1>
        <p className="text-sm text-muted-foreground">
          Play against a neural-network MCTS opponent powered by an ONNX model.
        </p>
      </div>
    </header>
  );
}

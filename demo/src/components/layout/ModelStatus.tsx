import {

  Card,

  CardContent,

  CardDescription,

  CardHeader,

  CardTitle,

} from "@/components/ui/card";

import { Spinner } from "@/components/ui/spinner";

import type { AiMode, ModelLoadState } from "@/hooks/useGame";

import { Badge } from "@/components/ui/badge";



interface ModelStatusProps {
  modelLoad: ModelLoadState;
  aiMode: AiMode;
}

export function ModelStatus({ modelLoad, aiMode }: ModelStatusProps) {

  return (

    <Card size="sm">

      <CardHeader>

        <CardTitle>Model</CardTitle>

        <CardDescription>Neural policy/value network</CardDescription>

      </CardHeader>

      <CardContent>

        {modelLoad.status === "loading" && (

          <div className="flex items-center gap-2 text-sm text-muted-foreground">

            <Spinner className="size-4" />

            <span>Loading model…</span>

          </div>

        )}



        {modelLoad.status === "error" && (

          <div className="flex flex-col gap-2 text-sm">

            <Badge variant="destructive">Manifest error</Badge>

            <p className="text-muted-foreground">{modelLoad.message}</p>

          </div>

        )}



        {modelLoad.status === "ready" && modelLoad.useNeural && aiMode === "policy" && (
          <div className="flex flex-col gap-2 text-sm">
            <Badge>Policy network active</Badge>
            <p className="text-muted-foreground">
              Moves come directly from the policy head — no MCTS search.
            </p>
          </div>
        )}

        {modelLoad.status === "ready" && modelLoad.useNeural && aiMode === "mcts" && (
          <div className="flex flex-col gap-2 text-sm">
            <Badge>Neural MCTS active</Badge>
            <p className="text-muted-foreground">
              Policy network guides MCTS rollouts.
            </p>
          </div>
        )}



        {modelLoad.status === "ready" && !modelLoad.useNeural && (

          <div className="flex flex-col gap-2 text-sm">

            <Badge variant="secondary">Pure MCTS fallback</Badge>

            <p className="text-muted-foreground">{modelLoad.message}</p>

          </div>

        )}

      </CardContent>

    </Card>

  );

}



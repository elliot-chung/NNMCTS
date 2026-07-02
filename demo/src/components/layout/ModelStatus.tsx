import {

  Card,

  CardContent,

  CardDescription,

  CardHeader,

  CardTitle,

} from "@/components/ui/card";

import { Spinner } from "@/components/ui/spinner";

import type { ModelLoadState } from "@/hooks/useGame";

import { Badge } from "@/components/ui/badge";



interface ModelStatusProps {

  modelLoad: ModelLoadState;

}



export function ModelStatus({ modelLoad }: ModelStatusProps) {

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



        {modelLoad.status === "ready" && modelLoad.useNeural && (

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



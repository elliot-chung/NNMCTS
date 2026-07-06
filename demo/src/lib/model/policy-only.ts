import type { MctsEnvironment } from "@/lib/mcts/types";

import { buildTensor } from "./build-tensor";
import { predict } from "./inference";
import { maskedSoftmax } from "./masked-softmax";

export async function selectPolicyMove<TMove>(
  environment: MctsEnvironment<TMove>,
): Promise<{ move: TMove; policy: number[] }> {
  const tensor = buildTensor({ environment });
  const { policyLogits } = await predict(tensor);
  const mask = environment.getMask();
  const policy = maskedSoftmax(policyLogits, mask);

  const validMoves = environment.validMoves();
  if (validMoves.length === 0) {
    throw new Error("No legal moves");
  }

  let bestMove = validMoves[0];
  let bestProb = policy[environment.translate(bestMove)];

  for (let i = 1; i < validMoves.length; i++) {
    const move = validMoves[i];
    const prob = policy[environment.translate(move)];
    if (prob > bestProb) {
      bestProb = prob;
      bestMove = move;
    }
  }

  return { move: bestMove, policy: [...policy] };
}

import type { MctsEnvironment } from "@/lib/mcts/types";

export const TENSOR_CHANNELS = 2;
export const TENSOR_SIZE = 81;

export function buildTensor<TMove>(
  node: { environment: MctsEnvironment<TMove> },
): Float32Array {
  const [state, mask] = node.environment.getCanonicalState();
  const tensor = new Float32Array(TENSOR_CHANNELS * TENSOR_SIZE);

  for (let i = 0; i < TENSOR_SIZE; i++) {
    tensor[i] = state[i];
    tensor[TENSOR_SIZE + i] = mask[i];
  }

  return tensor;
}

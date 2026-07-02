import * as ort from "onnxruntime-web";

import { TENSOR_CHANNELS, TENSOR_SIZE } from "./build-tensor";
import { createInferenceSession } from "./loader";

export interface InferenceResult {
  policyLogits: Float32Array;
  value: number;
}

export interface ModelInference {
  predict(input: Float32Array): InferenceResult | Promise<InferenceResult>;
}

class OnnxModelInference implements ModelInference {
  constructor(
    private readonly session: ort.InferenceSession,
    private readonly inputName = "input",
  ) {}

  async predict(input: Float32Array): Promise<InferenceResult> {
    const tensor = new ort.Tensor("float32", input, [
      1,
      TENSOR_CHANNELS,
      TENSOR_SIZE,
    ]);
    const outputs = await this.session.run({ [this.inputName]: tensor });

    const policyOutput = outputs.policy_logits;
    const valueOutput = outputs.value;
    if (!policyOutput || !valueOutput) {
      throw new Error("ONNX model returned unexpected outputs");
    }

    const policyLogits = policyOutput.data as Float32Array;
    const valueData = valueOutput.data as Float32Array | number[];
    const value = valueData[0] ?? 0;

    return {
      policyLogits:
        policyLogits instanceof Float32Array
          ? policyLogits
          : new Float32Array(policyLogits),
      value: Number(value),
    };
  }
}

let loadedModel: ModelInference | null = null;

export function setModel(model: ModelInference | null): void {
  loadedModel = model;
}

export function isModelLoaded(): boolean {
  return loadedModel !== null;
}

export async function loadModel(modelUrl: string): Promise<ModelInference> {
  const session = await createInferenceSession(modelUrl);
  return new OnnxModelInference(session);
}

export async function predict(input: Float32Array): Promise<InferenceResult> {
  if (!loadedModel) {
    throw new Error("Model not loaded");
  }
  return await loadedModel.predict(input);
}

export interface InferenceFixture {
  name: string;
  state: number[];
  mask: number[];
  expected_logits: number[];
  expected_value: number;
}

export async function validateAgainstFixture(
  fixture: InferenceFixture,
  tolerance = 1e-4,
): Promise<void> {
  const tensor = new Float32Array(TENSOR_CHANNELS * TENSOR_SIZE);
  for (let i = 0; i < TENSOR_SIZE; i++) {
    tensor[i] = fixture.state[i] ?? 0;
    tensor[TENSOR_SIZE + i] = fixture.mask[i] ?? 0;
  }

  const { policyLogits, value } = await predict(tensor);

  for (let i = 0; i < fixture.expected_logits.length; i++) {
    const expected = fixture.expected_logits[i] ?? 0;
    const actual = policyLogits[i] ?? 0;
    if (Math.abs(actual - expected) > tolerance) {
      throw new Error(
        `Fixture ${fixture.name}: logit[${i}] expected ${expected}, got ${actual}`,
      );
    }
  }

  if (Math.abs(value - fixture.expected_value) > tolerance) {
    throw new Error(
      `Fixture ${fixture.name}: value expected ${fixture.expected_value}, got ${value}`,
    );
  }
}

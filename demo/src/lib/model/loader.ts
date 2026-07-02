import * as ort from "onnxruntime-web";

export interface ModelManifestEntry {
  id: string;
  gameType: string;
  onnxPath: string;
  inputShape: [number, number, number];
  policySize: number;
  defaultMctsIters: number;
}

export interface ModelManifest {
  models: ModelManifestEntry[];
}

const DEFAULT_WASM_PATH =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";

let wasmConfigured = false;

export function configureOrtWasm(wasmPaths = DEFAULT_WASM_PATH): void {
  if (wasmConfigured) {
    return;
  }
  ort.env.wasm.wasmPaths = wasmPaths;
  ort.env.wasm.numThreads = 1;
  wasmConfigured = true;
}

export async function fetchManifest(
  url = "/models/manifest.json",
): Promise<ModelManifest> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load model manifest (${response.status})`);
  }
  return response.json() as Promise<ModelManifest>;
}

export function getUtttModelEntry(
  manifest: ModelManifest,
): ModelManifestEntry | undefined {
  return manifest.models.find((entry) => entry.gameType === "UTTT");
}

export async function createInferenceSession(
  modelUrl: string,
): Promise<ort.InferenceSession> {
  configureOrtWasm();
  return ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
  });
}

export async function modelFileExists(modelUrl: string): Promise<boolean> {
  try {
    const response = await fetch(modelUrl, { method: "HEAD" });
    return response.ok;
  } catch {
    return false;
  }
}

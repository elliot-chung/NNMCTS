declare module "onnxruntime-web" {
  export class Tensor {
    readonly data: Float32Array | number[];
    readonly dims: readonly number[];
    readonly type: string;
    constructor(
      type: string,
      data: Float32Array | number[],
      dims: readonly number[],
    );
  }

  export interface InferenceSession {
    run(
      feeds: Record<string, Tensor>,
    ): Promise<Record<string, Tensor>>;
  }

  export namespace env {
    namespace wasm {
      let wasmPaths: string;
      let numThreads: number;
    }
  }

  export const InferenceSession: {
    create: (
      path: string,
      options?: { executionProviders?: string[] },
    ) => Promise<InferenceSession>;
  };
}

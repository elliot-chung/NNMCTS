import * as codebuild from "aws-cdk-lib/aws-codebuild";
import { TrainingProfile } from "./cloud-training-config";

export interface PipelineBuildSpecOptions {
  device: "cpu" | "cuda";
  gameType?: string;
  rounds?: string;
  gamesPerRound?: string;
  epochs?: string;
  batchSize?: string;
  mctsIters?: string;
  player1Type?: string;
  player2Type?: string;
  maxRuntimeSeconds?: string;
  outputDir?: string;
}

export function trainingProfileToBuildSpecOptions(
  profile: TrainingProfile,
  device: "cpu" | "cuda",
  maxRuntimeSeconds: number,
): PipelineBuildSpecOptions {
  return {
    device,
    gameType: profile.gameType,
    rounds: String(profile.rounds),
    gamesPerRound: String(profile.gamesPerRound),
    epochs: String(profile.epochs),
    batchSize: String(profile.batchSize),
    mctsIters: String(profile.mctsIters),
    player1Type: profile.player1Type,
    player2Type: profile.player2Type,
    maxRuntimeSeconds: String(maxRuntimeSeconds),
  };
}

function torchInstallCommand(device: "cpu" | "cuda"): string {
  if (device === "cuda") {
    return "pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124";
  }
  return "pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu";
}

export function createPipelineBuildSpec(options: PipelineBuildSpecOptions) {
  const device = options.device;
  const installCommands = [
    device === "cuda"
      ? 'echo "Installing PyTorch CUDA and dependencies..."'
      : 'echo "Installing PyTorch CPU and dependencies..."',
    torchInstallCommand(device),
    "pip install --quiet -r requirements.txt",
  ];

  const preBuildCommands = [
    'echo "Run ID ${CODEBUILD_BUILD_ID}"',
    'echo "Artifacts bucket ${ARTIFACTS_BUCKET}"',
    'export PYTHONPATH="${CODEBUILD_SRC_DIR}:${PYTHONPATH:-}"',
    'mkdir -p "${OUTPUT_DIR}/datasets" "${OUTPUT_DIR}/checkpoints"',
    [
      'if [ -n "${INITIAL_CHECKPOINT_S3_URI:-}" ]; then',
      '  aws s3 cp "${INITIAL_CHECKPOINT_S3_URI}" initial_checkpoint.pt',
      '  export INITIAL_CHECKPOINT_FLAG="--initial-checkpoint initial_checkpoint.pt"',
      "else",
      '  export INITIAL_CHECKPOINT_FLAG=""',
      "fi",
    ].join("\n"),
  ];

  if (device === "cuda") {
    preBuildCommands.splice(
      3,
      0,
      "nvidia-smi",
      'python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"',
    );
  }

  return codebuild.BuildSpec.fromObject({
    version: "0.2",
    env: {
      variables: {
        GAME_TYPE: options.gameType ?? "TTT",
        ROUNDS: options.rounds ?? "3",
        GAMES_PER_ROUND: options.gamesPerRound ?? "50",
        EPOCHS: options.epochs ?? "10",
        BATCH_SIZE: options.batchSize ?? "64",
        MCTS_ITERS: options.mctsIters ?? "50",
        PLAYER1_TYPE: options.player1Type ?? "mcts",
        PLAYER2_TYPE: options.player2Type ?? "mcts",
        MAX_RUNTIME_SECONDS: options.maxRuntimeSeconds ?? "3600",
        OUTPUT_DIR: options.outputDir ?? "output",
        DEVICE: device === "cuda" ? "cuda" : "cpu",
      },
    },
    phases: {
      install: {
        "runtime-versions": {
          python: "3.11",
        },
        commands: installCommands,
      },
      pre_build: {
        commands: preBuildCommands,
      },
      build: {
        commands: [
          'echo "Starting NNMCTS pipeline on ${DEVICE} (max ${MAX_RUNTIME_SECONDS}s)..."',
          'PYTHONPATH="${CODEBUILD_SRC_DIR}:${PYTHONPATH:-}" timeout "${MAX_RUNTIME_SECONDS}" python run_pipeline.py --game-type "${GAME_TYPE}" --rounds "${ROUNDS}" --games-per-round "${GAMES_PER_ROUND}" --output-dir "${OUTPUT_DIR}" --device "${DEVICE}" --player1-type "${PLAYER1_TYPE}" --player2-type "${PLAYER2_TYPE}" --player1-iters "${MCTS_ITERS}" --player2-iters "${MCTS_ITERS}" --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --augment-train --deduplicate-train ${INITIAL_CHECKPOINT_FLAG}',
        ],
      },
      post_build: {
        commands: [
          'echo "Uploading artifacts to s3://${ARTIFACTS_BUCKET}/runs/${CODEBUILD_BUILD_ID}/"',
          'aws s3 sync "${OUTPUT_DIR}/" "s3://${ARTIFACTS_BUCKET}/runs/${CODEBUILD_BUILD_ID}/output/" --exclude "*.pkl"',
          'aws s3 sync "${OUTPUT_DIR}/checkpoints/" "s3://${ARTIFACTS_BUCKET}/runs/${CODEBUILD_BUILD_ID}/checkpoints/"',
          [
            "python -c \"import json, os, pathlib; output_dir = os.environ['OUTPUT_DIR']; checkpoints = sorted(pathlib.Path(output_dir, 'checkpoints').glob('*.pt')); manifest = {'build_id': os.environ['CODEBUILD_BUILD_ID'], 'game_type': os.environ['GAME_TYPE'], 'device': os.environ['DEVICE'], 'rounds': int(os.environ['ROUNDS']), 'games_per_round': int(os.environ['GAMES_PER_ROUND']), 'epochs': int(os.environ['EPOCHS']), 'latest_checkpoint': checkpoints[-1].name if checkpoints else None, 'status': 'complete'}; pathlib.Path('manifest.json').write_text(json.dumps(manifest, indent=2))\"",
          ].join("\n"),
          'aws s3 cp manifest.json "s3://${ARTIFACTS_BUCKET}/runs/${CODEBUILD_BUILD_ID}/manifest.json"',
          'echo "Pipeline complete. Manifest uploaded."',
        ],
      },
    },
  });
}

import * as fs from "fs";
import * as path from "path";

export interface TrainingProfile {
  gameType: string;
  rounds: number;
  gamesPerRound: number;
  epochs: number;
  batchSize: number;
  mctsIters: number;
  player1Type: string;
  player2Type: string;
  selfPlayWorkers?: number;
  playDevice?: string;
  trainDevice?: string;
}

export interface CloudTrainingTimeouts {
  maxInstanceSeconds: number;
  maxTrainingSeconds: number;
  maxSmokeInstanceSeconds: number;
  maxSmokeTrainingSeconds: number;
}

export interface CloudTrainingConfig {
  timeouts: CloudTrainingTimeouts;
  gpuSmoke: TrainingProfile;
  gpu: TrainingProfile;
  gpuAmiIds?: Record<string, string>;
}

const DEFAULT_CONFIG: CloudTrainingConfig = {
  timeouts: {
    maxInstanceSeconds: 5400,
    maxTrainingSeconds: 3600,
    maxSmokeInstanceSeconds: 1800,
    maxSmokeTrainingSeconds: 600,
  },
  gpuSmoke: {
    gameType: "UTTT",
    rounds: 1,
    gamesPerRound: 2,
    epochs: 1,
    batchSize: 32,
    mctsIters: 10,
    player1Type: "mcts",
    player2Type: "mcts",
    selfPlayWorkers: 1,
    playDevice: "cpu",
    trainDevice: "cuda",
  },
  gpu: {
    gameType: "UTTT",
    rounds: 5,
    gamesPerRound: 100,
    epochs: 20,
    batchSize: 128,
    mctsIters: 75,
    player1Type: "nmcts",
    player2Type: "nmcts",
    selfPlayWorkers: 3,
    playDevice: "cpu",
    trainDevice: "cuda",
  },
};

function mergeProfile(base: TrainingProfile, override?: Partial<TrainingProfile>): TrainingProfile {
  return { ...base, ...override };
}

function mergeTimeouts(base: CloudTrainingTimeouts, override?: Partial<CloudTrainingTimeouts>): CloudTrainingTimeouts {
  return { ...base, ...override };
}

export function loadCloudTrainingConfig(configPath?: string): CloudTrainingConfig {
  const resolvedPath =
    configPath ?? path.resolve(process.cwd(), "../config/cloud-training.json");

  if (!fs.existsSync(resolvedPath)) {
    return DEFAULT_CONFIG;
  }

  const parsed = JSON.parse(fs.readFileSync(resolvedPath, "utf8")) as Partial<CloudTrainingConfig> & {
    smoke?: TrainingProfile;
  };
  const gpuSmoke = parsed.gpuSmoke ?? parsed.smoke;
  return {
    timeouts: mergeTimeouts(DEFAULT_CONFIG.timeouts, parsed.timeouts),
    gpuSmoke: mergeProfile(DEFAULT_CONFIG.gpuSmoke, gpuSmoke),
    gpu: mergeProfile(DEFAULT_CONFIG.gpu, parsed.gpu),
    gpuAmiIds: parsed.gpuAmiIds,
  };
}

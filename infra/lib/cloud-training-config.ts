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
}

export interface CloudTrainingTimeouts {
  maxRuntimeSeconds: number;
  codeBuildTimeoutMinutes: number;
  codeBuildQueuedTimeoutMinutes: number;
  maxInstanceSeconds: number;
  maxTrainingSeconds: number;
}

export interface CloudTrainingConfig {
  timeouts: CloudTrainingTimeouts;
  smoke: TrainingProfile;
  gpu: TrainingProfile;
}

const DEFAULT_CONFIG: CloudTrainingConfig = {
  timeouts: {
    maxRuntimeSeconds: 3600,
    codeBuildTimeoutMinutes: 60,
    codeBuildQueuedTimeoutMinutes: 30,
    maxInstanceSeconds: 5400,
    maxTrainingSeconds: 3600,
  },
  smoke: {
    gameType: "TTT",
    rounds: 3,
    gamesPerRound: 50,
    epochs: 10,
    batchSize: 64,
    mctsIters: 50,
    player1Type: "mcts",
    player2Type: "mcts",
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

  const parsed = JSON.parse(fs.readFileSync(resolvedPath, "utf8")) as Partial<CloudTrainingConfig>;
  return {
    timeouts: mergeTimeouts(DEFAULT_CONFIG.timeouts, parsed.timeouts),
    smoke: mergeProfile(DEFAULT_CONFIG.smoke, parsed.smoke),
    gpu: mergeProfile(DEFAULT_CONFIG.gpu, parsed.gpu),
  };
}

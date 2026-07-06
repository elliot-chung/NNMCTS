# Neural Monte Carlo Tree Search

An MCTS implementation combined with neural policy/value networks, trained via supervised learning on self-play data.

## Overview

AlphaGo used neural networks with MCTS; this project follows the same approach. It includes:
- MCTS from scratch
- PyTorch models for policy and value
- Data generation via parallel self-play
- Supervised training with augmentation and deduplication
- Optional AWS pipeline for GPU UTTT training

## Technical highlights

### Monte Carlo Tree Search

Four stages: selection (UCB1), expansion, rollout (policy/value from network), backpropagation. `NeuralNode` guides search with the trained network.

### Neural architecture

`TTTNet` and `UTTTNet` use CNNs with separate policy and value heads, canonical state handling, and valid-move masks (UTTT).

### Data pipeline

Multiprocess self-play, symmetry augmentation, deduplication with averaged labels, and train/val splits.


## Requirements

- Python 3.8+
- PyTorch, NumPy, tqdm (see [requirements.txt](requirements.txt))

For cloud training:
- AWS CLI v2
- Node.js 18+ and npm (for CDK)
- PowerShell (Windows) or adapt the scripts for bash

## Local usage

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

See [Usage guide](#usage-guide) below for the three root scripts.

## Usage guide

Three CLI entry points live in the repo root. All support `TTT` (tic-tac-toe) and `UTTT` (ultimate tic-tac-toe). Player engines are `random`, `mcts` (pure MCTS), or `nmcts` (MCTS guided by a neural checkpoint).

Run any script with `-h` / `--help` for the full argument list and defaults.

### `play_matches.py`

Pit two players against each other and print win/draw statistics. Optionally record games to a dataset file (`.pkl`) for later training.

**Required flags**

| Flag | Description |
|------|-------------|
| `--game-type` | `TTT` or `UTTT` |
| `--num-games` | Number of games to play |
| `--player1-type` | Player 1 engine: `random`, `mcts`, or `nmcts` |
| `--player2-type` | Player 2 engine |

**Common optional flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--player1-iters` / `--player2-iters` | 100 | MCTS simulations per move |
| `--player1-model` / `--player2-model` | — | Checkpoint path (required for `nmcts`) |
| `--device` | auto (`cuda` if available) | Default device when `--play-device` is omitted |
| `--play-device` | — | Device for self-play, including NMCTS inference (overrides `--device`) |
| `--workers` | 1 | Parallel self-play worker processes |
| `--record-output` | — | Save recorded positions to this `.pkl` path |
| `--show-mcts-timing` | off | Print MCTS phase breakdown per move |
| `--non-interactive-logging` | off | Throttle tqdm progress updates and emit plain log lines (for non-TTY shells and log files) |

**Examples**

```bash
# MCTS vs MCTS on tic-tac-toe
python play_matches.py \
  --game-type TTT \
  --num-games 100 \
  --player1-type mcts \
  --player2-type mcts \
  --player1-iters 50 \
  --player2-iters 50

# Evaluate a trained model against pure MCTS and save the games
python play_matches.py \
  --game-type UTTT \
  --num-games 50 \
  --player1-type nmcts \
  --player2-type mcts \
  --player1-model output/checkpoints/round_005.pt \
  --player1-iters 100 \
  --player2-iters 100 \
  --record-output output/eval_games.pkl \
  --device cuda \
  --workers 4
```

### `train_model.py`

Train (or fine-tune) a policy/value network from a recorded dataset produced by `play_matches.py` or `run_pipeline.py`.

**Required flags**

| Flag | Description |
|------|-------------|
| `--game-type` | `TTT` or `UTTT` (must match the dataset) |
| `--dataset-path` | Path to a `.pkl` records file |
| `--output-model` | Where to write the checkpoint (`.pt`) |

**Common optional flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint-path` | — | Resume or fine-tune from an existing checkpoint |
| `--device` | auto (`cuda` if available) | PyTorch device |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 64 | Minibatch size |
| `--learning-rate` | 1e-3 | Adam learning rate |
| `--weight-decay` | 0.0 | Adam weight decay |
| `--value-loss-weight` | 0.1 | Weight on value (MSE) loss |
| `--policy-loss-weight` | 0.9 | Weight on policy (soft cross-entropy) loss |
| `--val-split` | 0.2 | Fraction held out for validation |
| `--seed` | 0 | Random seed for split and augmentation |
| `--augment-train` / `--augment-val` | off | Symmetry augmentation on train/val |
| `--deduplicate-train` / `--deduplicate-val` | off | Deduplicate positions with averaged labels |
| `--amp` | off | Mixed-precision training on CUDA |
| `--non-interactive-logging` | off | Throttle tqdm progress updates and emit plain log lines (for non-TTY shells and log files) |

**Example**

```bash
python train_model.py \
  --game-type TTT \
  --dataset-path output/datasets/round_002.pkl \
  --output-model output/checkpoints/manual_train.pt \
  --epochs 50 \
  --batch-size 64 \
  --augment-train \
  --deduplicate-train \
  --device cpu
```

### `run_pipeline.py`

Alternate self-play and supervised training for several rounds. Each round generates a dataset, trains a checkpoint, and (for `nmcts` players without a fixed model) uses the latest checkpoint in the next round.

**Output layout**

```
<output-dir>/
  datasets/round_001.pkl, round_002.pkl, ...
  checkpoints/round_001.pt, round_002.pt, ...
```

**Required flags**

| Flag | Description |
|------|-------------|
| `--game-type` | `TTT` or `UTTT` |
| `--rounds` | Number of self-play → train cycles |
| `--games-per-round` | Self-play games generated each round |
| `--output-dir` | Root directory for datasets and checkpoints |
| `--player1-type` | Player 1 engine |
| `--player2-type` | Player 2 engine |

**Common optional flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--player1-iters` / `--player2-iters` | 100 | MCTS simulations per move |
| `--player1-model` / `--player2-model` | — | Fixed checkpoint for an `nmcts` player; if omitted, the latest trained checkpoint is used |
| `--initial-checkpoint` | — | Starting weights for training and for `nmcts` before round 1 |
| `--start-round` | — | First pipeline round number (default: one past the round in `--initial-checkpoint`, or `1`) |
| `--device` | auto (`cuda` if available) | Default device when `--play-device` or `--train-device` are omitted |
| `--play-device` | — | Device for self-play and NMCTS inference; defaults to `cpu` when `--train-device` is `cuda`, otherwise `--device` |
| `--train-device` | — | Device for supervised training; defaults to `--device` |
| `--epochs` | 100 | Training epochs per round |
| `--batch-size` | 64 | Minibatch size |
| `--learning-rate` | 1e-3 | Adam learning rate |
| `--weight-decay` | 0.0 | Adam weight decay |
| `--value-loss-weight` | 0.1 | Value loss weight |
| `--policy-loss-weight` | 0.9 | Policy loss weight |
| `--val-split` | 0.2 | Validation fraction |
| `--seed` | 0 | Base seed (incremented per round) |
| `--augment-train` / `--augment-val` | off | Symmetry augmentation |
| `--deduplicate-train` / `--deduplicate-val` | off | Deduplicate positions |
| `--accumulate-records` | off | Train each round on all games seen so far (ignored when eval gating is enabled) |
| `--num-eval-games` | `0` | Head-to-head games after training; `0` disables eval gating (legacy behavior) |
| `--winrate-threshold` | `0.55` | Candidate must **strictly exceed** this win rate vs the champion to be promoted |
| `--self-play-workers` | 1 | Parallel self-play workers |
| `--show-mcts-timing` | off | MCTS timing diagnostics |
| `--amp` | off | Mixed-precision training on CUDA |
| `--non-interactive-logging` | off | Throttle tqdm progress updates and emit plain log lines (for non-TTY shells and log files) |

When no checkpoint is available for an `nmcts` player (no `--initial-checkpoint` and no `--playerN-model`), round 1 falls back to `random` for that slot; later rounds use the champion or latest trained model.

#### Evaluation gating (`--num-eval-games > 0`)

When eval is enabled, the pipeline tracks a **champion** checkpoint used for self-play and as the training parent. After each round:

1. Self-play runs with the current champion.
2. Self-play records since the last promotion are merged into a champion-streak dataset (`datasets/round_NNN_champion.pkl`).
3. A **candidate** is trained from the champion and saved to `checkpoints/round_NNN.pt` (always written, even on rejection).
4. The candidate plays the champion in `--num-eval-games` head-to-head games with seat swapping (half as player 1, half as player 2).
5. If the candidate win rate is **strictly greater than** `--winrate-threshold`, it is promoted to champion and the streak dataset resets. Otherwise the champion is kept and the streak dataset grows with the next round's self-play.

**Notes:**

- Draws count as non-wins for the candidate win rate (only outright wins count).
- The first round with no prior champion (`--initial-checkpoint` omitted) skips eval and auto-promotes the candidate.
- `--accumulate-records` is ignored with a warning; champion-streak accumulation replaces it.
- Eval requires at least one `nmcts` self-play player without a fixed `--playerN-model`.

**Examples**

```bash
# Short local smoke test: MCTS self-play + training on CPU
python run_pipeline.py \
  --game-type TTT \
  --rounds 2 \
  --games-per-round 10 \
  --output-dir output \
  --player1-type mcts \
  --player2-type mcts \
  --player1-iters 25 \
  --player2-iters 25 \
  --epochs 3 \
  --device cpu

# Neural self-play with augmentation and deduplication (cloud defaults)
python run_pipeline.py \
  --game-type UTTT \
  --rounds 5 \
  --games-per-round 100 \
  --output-dir output \
  --player1-type nmcts \
  --player2-type nmcts \
  --player1-iters 100 \
  --player2-iters 100 \
  --epochs 20 \
  --batch-size 64 \
  --augment-train \
  --deduplicate-train \
  --self-play-workers 4 \
  --play-device cpu \
  --train-device cuda \
  --device cuda \
  --amp

# Resume locally from an existing checkpoint (round numbering continues from the filename)
python run_pipeline.py \
  --game-type UTTT \
  --rounds 5 \
  --games-per-round 100 \
  --output-dir output \
  --player1-type nmcts \
  --player2-type nmcts \
  --initial-checkpoint output/checkpoints/round_020.pt \
  --play-device cpu \
  --train-device cuda

# Neural self-play with evaluation gating (promote only if candidate beats champion)
python run_pipeline.py \
  --game-type TTT \
  --rounds 3 \
  --games-per-round 10 \
  --num-eval-games 20 \
  --winrate-threshold 0.55 \
  --output-dir output/eval-test \
  --player1-type nmcts \
  --player2-type nmcts \
  --player1-iters 25 \
  --player2-iters 25 \
  --epochs 3 \
  --device cpu
```

## Project structure

| Path | Purpose |
|------|---------|
| `nnmcts/` | Core library (games, MCTS, models, players) |
| `run_pipeline.py` | Self-play + training loop |
| `infra/` | AWS CDK stack (S3, GPU EC2 launch template) |
| `cloud/` | GPU instance bootstrap script |
| `scripts/` | Deploy, train, and teardown helpers (see [scripts/SCRIPTS.MD](scripts/SCRIPTS.MD)) |
| `cloud/install-gpu-deps.sh` | Shared GPU dependency installer (used by AMI bake and instance bootstrap) |
| `config/local.example.json` | Example local AWS settings (copy to `local.json`) |
| `config/cloud-training.json` | Cloud training hyperparameters and timeouts |

## AWS infrastructure setup

The cloud pipeline deploys:

- **S3 bucket** for source uploads and training artifacts
- **EC2 launch template** (`g4dn.xlarge`) for GPU UTTT training
- **VPC + IAM** for GPU instances (auto-shutdown after training)

Each pipeline run launches one GPU instance that runs a minimal UTTT smoke test, then continues with full training on the same instance if the smoke test succeeds.

### 1. Prerequisites

1. Configure AWS credentials (`aws configure` or `aws login`).
2. Ensure your account can launch GPU instances if you plan to use GPU training. Request a quota increase for **Running On-Demand G and VT instances** in your target region if needed.
3. Copy the example config and edit it:

```powershell
Copy-Item config\local.example.json config\local.json
```

`config/local.json` is gitignored. Set your preferred profile and region:

```json
{
  "awsProfile": "default",
  "region": "us-west-1",
  "stackName": "NnmctsPipelineStack"
}
```

Alternatively, set `AWS_PROFILE` and `AWS_REGION` in your environment instead of using `local.json`.

### Cloud training configuration (`config/cloud-training.json`)

This file is the default source for training hyperparameters, GPU AMI IDs, and timeouts used by the cloud scripts and CDK stack. It has these top-level sections:

- **`gpuAmiIds`** — optional per-region EC2 AMI IDs for GPU training instances (falls back to the base DLAMI when unset)
- **`gpuSmoke`** — defaults for the minimal GPU smoke test (`run_cloud_pipeline.ps1 -SmokeOnly` or the first step of a full run)
- **`gpu`** — defaults for full GPU EC2 training (`run_gpu_training.ps1` or the second step of a full run)
- **`timeouts`** — wall-clock limits for GPU instances

#### Training hyperparameters (`gpuSmoke` and `gpu`)

Both sections use the same fields. Values are passed to `run_pipeline.py` on each cloud run.

| Field | Description |
|-------|-------------|
| `gameType` | Game to train on: `TTT` (tic-tac-toe) or `UTTT` (ultimate tic-tac-toe). Cloud defaults use `UTTT`. |
| `rounds` | Number of self-play → train cycles. Each round generates games, then trains on the accumulated data. |
| `gamesPerRound` | Self-play games generated per round. |
| `epochs` | Training epochs per round on the current dataset. |
| `batchSize` | Minibatch size during supervised training. |
| `mctsIters` | MCTS simulations per move for both players (`--player1-iters` and `--player2-iters`). |
| `player1Type` | Player 1 engine: `random`, `mcts` (pure MCTS), or `nmcts` (MCTS guided by the neural net). |
| `player2Type` | Player 2 engine; same choices as `player1Type`. |
| `selfPlayWorkers` | Parallel self-play worker processes (GPU runs only). |
| `playDevice` | Device for self-play and NMCTS inference on the instance (`cpu` or `cuda`). Cloud defaults use `cpu`. |
| `trainDevice` | Device for supervised training on the instance (`cpu` or `cuda`). Cloud defaults use `cuda`. |
| `numEvalGames` | Optional. Head-to-head eval games after each training round (`0` = disabled, legacy behavior). Passed as `--num-eval-games`. |
| `winrateThreshold` | Optional. Win-rate threshold for promotion when eval is enabled (default `0.55`). Passed as `--winrate-threshold`. |

Smoke defaults use a tiny UTTT workload with `mcts` players. Full GPU defaults target a larger `UTTT` run with `nmcts` players. The default device split keeps self-play on CPU and training on GPU to avoid CUDA contention during data generation.

Cloud builds always enable `--augment-train`, `--deduplicate-train`, and `--non-interactive-logging` (so CloudWatch and `cloud-init` logs stay readable). Other `run_pipeline.py` flags (learning rate, loss weights, val split, etc.) are not exposed in this config and keep their Python defaults.

#### Timeouts (`timeouts`)

| Field | Used by | Description |
|-------|---------|-------------|
| `maxTrainingSeconds` | Full GPU EC2 | Max seconds spent in `run_pipeline.py` on the instance. |
| `maxInstanceSeconds` | Full GPU EC2 | Total instance wall-clock budget (bootstrap, install, training, upload). The instance shuts down when this is reached or when the script exits. |
| `maxSmokeTrainingSeconds` | GPU smoke EC2 | Max seconds for the smoke test training phase. |
| `maxSmokeInstanceSeconds` | GPU smoke EC2 | Total wall-clock budget for the smoke test instance. |

**No redeploy needed:** changing hyperparameters under `gpuSmoke` or `gpu` (including `playDevice` / `trainDevice`), or per-run timeout overrides (see below). Values are sent as EC2 tags on launch.

#### Per-run overrides

Script flags override the config file for a single run without editing JSON:

| Flag | Applies to | Config field |
|------|--------------|--------------|
| `-GameType` | Smoke, GPU | `gameType` |
| `-Rounds` | Smoke, GPU | `rounds` |
| `-GamesPerRound` | Smoke, GPU | `gamesPerRound` |
| `-Epochs` | Smoke, GPU | `epochs` |
| `-BatchSize` | Smoke, GPU | `batchSize` |
| `-MctsIters` | Smoke, GPU | `mctsIters` |
| `-Player1Type` | Smoke, GPU | `player1Type` |
| `-Player2Type` | Smoke, GPU | `player2Type` |
| `-SelfPlayWorkers` | Smoke, GPU | `selfPlayWorkers` |
| `-InitialCheckpointPath` | Full GPU only | Bundles a local `.pt` checkpoint into the source zip and resumes with `--initial-checkpoint` (skips smoke) |
| `-StartRound` | Full GPU only | First pipeline round number (default: inferred from checkpoint filename) |
| `-MaxTrainingSeconds` | Full GPU only | `timeouts.maxTrainingSeconds` |
| `-MaxInstanceSeconds` | Full GPU only | `timeouts.maxInstanceSeconds` |
| `-ConfigPath` | Smoke, GPU | Path to an alternate JSON config file |
| `-SmokeOnly` | Pipeline | Run only the GPU smoke test |
| `-SkipSmoke` | Pipeline | Skip smoke and launch full training directly |

`playDevice` and `trainDevice` are set in `config/cloud-training.json` only; they are not exposed as script CLI flags.

Examples:

```powershell
# Full pipeline: one instance runs smoke, then training
.\scripts\run_cloud_pipeline.ps1

# Smoke test only
.\scripts\run_cloud_pipeline.ps1 -SmokeOnly

# Full training only (skip smoke)
.\scripts\run_cloud_pipeline.ps1 -SkipSmoke -Rounds 10 -MaxTrainingSeconds 7200
```

### 2. Deploy the stack

Defaults in `config/cloud-training.json` are used at runtime via EC2 instance tags. See [Cloud training configuration](#cloud-training-configuration-configcloud-trainingjson) for field descriptions.

From the repo root:

```powershell
.\scripts\run_cloud_pipeline.ps1 -DeployOnly
```

This bootstraps CDK (if needed), installs npm dependencies under `infra/`, and deploys `NnmctsPipelineStack`.

Manual equivalent:

```powershell
cd infra
npm install
$env:CDK_DEFAULT_ACCOUNT = (aws sts get-caller-identity --query Account --output text)
$env:CDK_DEFAULT_REGION = "us-west-1"
npx cdk bootstrap "aws://$env:CDK_DEFAULT_ACCOUNT/$env:CDK_DEFAULT_REGION"
npx cdk deploy NnmctsPipelineStack --require-approval never
```

`CDK_DEFAULT_ACCOUNT` is required; it is read from your AWS identity, not hardcoded in the repo.

### 3. Run the cloud pipeline

Packages source (GPU bundles exclude `demo/` and `artifacts/`), uploads to S3, runs a minimal GPU smoke test on UTTT, then launches full GPU training if the smoke test succeeds:

```powershell
.\scripts\run_cloud_pipeline.ps1
```

Artifacts appear under `s3://<artifacts-bucket>/runs/<run-id>/checkpoints/`.

Per-run overrides are described in [Cloud training configuration](#cloud-training-configuration-configcloud-trainingjson). Quick examples:

```powershell
# Smoke test only
.\scripts\run_cloud_pipeline.ps1 -SmokeOnly

# Full training only (skip smoke)
.\scripts\run_cloud_pipeline.ps1 -SkipSmoke -Rounds 10 -MaxTrainingSeconds 7200
```

### 4. Run GPU training directly

Launches a single on-demand `g4dn.xlarge` without the pipeline smoke step. Use `-TrainingProfile gpuSmoke` for a smoke-only run:

```powershell
.\scripts\run_gpu_training.ps1
.\scripts\run_gpu_training.ps1 -TrainingProfile gpuSmoke -Wait

# Check status and recent logs
.\scripts\check_gpu_training.ps1

# Poll until the manifest appears or the instance stops
.\scripts\check_gpu_training.ps1 -Follow
```

Training limits and GPU defaults come from `config/cloud-training.json`. Per-run overrides:

```powershell
.\scripts\run_gpu_training.ps1 -Rounds 10 -MaxTrainingSeconds 7200
```

Resume from a local checkpoint (bundles the `.pt` into the uploaded source zip; smoke is skipped):

```powershell
.\scripts\run_gpu_training.ps1 `
  -InitialCheckpointPath artifacts\gpu-20260701-192839\round_020.pt `
  -Rounds 10 `
  -Wait
```

Round numbering continues from the checkpoint name (e.g. `round_020.pt` → next checkpoint is `round_021.pt`) unless you pass `-StartRound`. Run metadata is saved to `artifacts/latest-gpu-run.json`. See [scripts/SCRIPTS.MD](scripts/SCRIPTS.MD) for download, ONNX export, and monitoring workflows.

The instance always shuts down when the script exits. Live logs stream to CloudWatch log group `/nnmcts/gpu-training` (one stream per instance ID); `.\scripts\check_gpu_training.ps1` fetches recent events from there. A full log archive is also uploaded to `s3://<bucket>/runs/<run-id>/gpu-train.log` when the run completes.

### 5. Custom GPU AMI (optional, faster cold start)

GPU instances bootstrap faster when dependencies are pre-baked into a custom AMI. The shared installer is `cloud/install-gpu-deps.sh`; runtime training still uses `cloud/gpu-train.sh`, which defers CloudWatch agent setup to a background task and skips redundant installs when `/opt/nnmcts/.gpu-deps-ready` exists.

Build a custom AMI once (or after PyTorch / pip dependency changes):

```powershell
.\scripts\build_gpu_ami.ps1
```

Then update `gpuAmiIds` in `config/cloud-training.json` with the printed AMI ID and redeploy:

```powershell
.\scripts\run_cloud_pipeline.ps1 -DeployOnly
```

Subsequent training launches (`run_gpu_training.ps1`, `run_cloud_pipeline.ps1 -RunOnly`) do not require redeploy unless the AMI ID or CDK stack changes.

**When to rebuild:** changes to `cloud/install-gpu-deps.sh`, PyTorch/CUDA requirements, or numpy/tqdm versions. Hyperparameter and timeout edits in `cloud-training.json` do not require an AMI rebuild.

### 6. Tear down

Stop instances, destroy the CloudFormation stack, and optionally delete the artifacts bucket:

```powershell
# Interactive confirmation
.\scripts\teardown_cloud_pipeline.ps1

# Non-interactive, delete all artifacts
.\scripts\teardown_cloud_pipeline.ps1 -Force

# Keep S3 training outputs
.\scripts\teardown_cloud_pipeline.ps1 -KeepArtifacts
```

This does not remove the CDK bootstrap stack (`CDKToolkit`).

### Region notes

- GPU training uses EC2 `g4dn.xlarge` instances.
- The GPU AMI is configured per region in `config/cloud-training.json` under `gpuAmiIds`. The CDK stack reads this at deploy time. The default is the AWS Deep Learning AMI PyTorch base image for `us-west-1`.
- To speed up cold start, bake a custom AMI once with `.\scripts\build_gpu_ami.ps1`, paste the new AMI ID into `gpuAmiIds`, and redeploy: `.\scripts\run_cloud_pipeline.ps1 -DeployOnly`. Rebuild when PyTorch or `cloud/install-gpu-deps.sh` change. Later training runs can use `-RunOnly` without redeploying.

## Publishing / hygiene

Before committing or making the repo public:

```powershell
.\scripts\check_public_safety.ps1
```

Do not commit `infra/cdk.out/`, `infra/cdk.context.json`, `infra/node_modules/`, `config/local.json`, or model checkpoints (`.pt`, `.pkl`).


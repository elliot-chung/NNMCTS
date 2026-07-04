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
| `--device` | `cpu` | PyTorch device (`cpu`, `cuda`, etc.) |
| `--workers` | 1 | Parallel self-play worker processes |
| `--record-output` | — | Save recorded positions to this `.pkl` path |
| `--show-mcts-timing` | off | Print MCTS phase breakdown per move |
| `--batched-inference` | off | Shared GPU inference server for `nmcts` on CUDA |
| `--inference-batch-size` | 32 | Batch size for batched inference |
| `--inference-max-wait-ms` | 5.0 | Max wait before flushing an inference batch |

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
| `--device` | `cpu` | PyTorch device |
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
| `--accumulate-records` | off | Train each round on all games seen so far |
| `--self-play-workers` | 1 | Parallel self-play workers |
| `--batched-inference` | off | Shared GPU inference for `nmcts` |
| `--show-mcts-timing` | off | MCTS timing diagnostics |
| `--amp` | off | Mixed-precision training on CUDA |

If `--player1-type` or `--player2-type` is `nmcts` but no checkpoint exists yet (no `--initial-checkpoint` and no `--playerN-model`), round 1 falls back to `random` for that slot; later rounds use the newly trained model.

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
  --device cuda \
  --amp
```

## Project structure

| Path | Purpose |
|------|---------|
| `nnmcts/` | Core library (games, MCTS, models, players) |
| `run_pipeline.py` | Self-play + training loop |
| `infra/` | AWS CDK stack (S3, GPU EC2 launch template) |
| `cloud/` | GPU instance bootstrap script |
| `scripts/` | Deploy, train, and teardown helpers |
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

This file is the default source for training hyperparameters and timeouts used by the cloud scripts and CDK stack. It has three top-level sections:

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

Smoke defaults use a tiny UTTT workload with `mcts` players. Full GPU defaults target a larger `UTTT` run with `nmcts` players.

Cloud builds always enable `--augment-train` and `--deduplicate-train`. Other `run_pipeline.py` flags (learning rate, loss weights, val split, etc.) are not exposed in this config and keep their Python defaults.

#### Timeouts (`timeouts`)

| Field | Used by | Description |
|-------|---------|-------------|
| `maxTrainingSeconds` | Full GPU EC2 | Max seconds spent in `run_pipeline.py` on the instance. |
| `maxInstanceSeconds` | Full GPU EC2 | Total instance wall-clock budget (bootstrap, install, training, upload). The instance shuts down when this is reached or when the script exits. |
| `maxSmokeTrainingSeconds` | GPU smoke EC2 | Max seconds for the smoke test training phase. |
| `maxSmokeInstanceSeconds` | GPU smoke EC2 | Total wall-clock budget for the smoke test instance. |

**No redeploy needed:** changing hyperparameters under `gpuSmoke` or `gpu`, or per-run timeout overrides (see below). Values are sent as EC2 tags on launch.

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
| `-MaxTrainingSeconds` | Full GPU only | `timeouts.maxTrainingSeconds` |
| `-MaxInstanceSeconds` | Full GPU only | `timeouts.maxInstanceSeconds` |
| `-ConfigPath` | Smoke, GPU | Path to an alternate JSON config file |
| `-SmokeOnly` | Pipeline | Run only the GPU smoke test |
| `-SkipSmoke` | Pipeline | Skip smoke and launch full training directly |

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

Packages source, uploads to S3, runs a minimal GPU smoke test on UTTT, then launches full GPU training if the smoke test succeeds:

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

The instance always shuts down when the script exits. Live logs stream to CloudWatch log group `/nnmcts/gpu-training` (one stream per instance ID); `.\scripts\check_gpu_training.ps1` fetches recent events from there. A full log archive is also uploaded to `s3://<bucket>/runs/<run-id>/gpu-train.log` when the run completes.

### 5. Tear down

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
- The GPU AMI in the CDK stack is pinned for `us-west-1`. For other regions, update the AMI lookup in `infra/lib/nnmcts-pipeline-stack.ts`.

## Publishing / hygiene

Before committing or making the repo public:

```powershell
python scripts\strip_notebook_metadata.py MCTS.ipynb
.\scripts\check_public_safety.ps1
```

Do not commit `infra/cdk.out/`, `infra/cdk.context.json`, `infra/node_modules/`, `config/local.json`, or model checkpoints (`.pt`, `.pkl`).


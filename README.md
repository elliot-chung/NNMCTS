# Neural Monte Carlo Tree Search

An MCTS implementation combined with neural policy/value networks, trained via supervised learning on self-play data.

## Overview

AlphaGo used neural networks with MCTS; this project follows the same approach. It includes:
- MCTS from scratch
- PyTorch models for policy and value
- Data generation via parallel self-play
- Supervised training with augmentation and deduplication
- Optional AWS pipeline for CPU smoke tests and GPU training

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

Run a short local pipeline (Tic-Tac-Toe, MCTS vs MCTS):

```bash
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
```

Other entry points:
- `play_matches.py` — pit two players against each other
- `train_model.py` — train from a recorded dataset
- `run_pipeline.py` — alternate self-play and training rounds

## Project structure

| Path | Purpose |
|------|---------|
| `nnmcts/` | Core library (games, MCTS, models, players) |
| `run_pipeline.py` | Self-play + training loop |
| `infra/` | AWS CDK stack (S3, CodeBuild, GPU EC2 launch template) |
| `cloud/` | Buildspecs and GPU instance bootstrap script |
| `scripts/` | Deploy, train, and teardown helpers |
| `config/local.example.json` | Example local AWS settings (copy to `local.json`) |
| `config/cloud-training.json` | Cloud training hyperparameters and timeouts |

## AWS infrastructure setup

The cloud pipeline deploys:

- **S3 bucket** for source uploads and training artifacts
- **CodeBuild** project for CPU smoke training (max 1 hour per run)
- **EC2 launch template** (`g4dn.xlarge`) for GPU training in regions without CodeBuild GPU support
- **VPC + IAM** for GPU instances (auto-shutdown after training)

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

- **`smoke`** — defaults for CPU smoke runs (`run_cloud_pipeline.ps1` without `-Gpu`)
- **`gpu`** — defaults for GPU EC2 runs (`run_cloud_pipeline.ps1 -Gpu` or `run_gpu_training.ps1`)
- **`timeouts`** — wall-clock limits for CodeBuild and GPU instances

#### Training hyperparameters (`smoke` and `gpu`)

Both sections use the same fields. Values are passed to `run_pipeline.py` on each cloud run.

| Field | Description |
|-------|-------------|
| `gameType` | Game to train on: `TTT` (tic-tac-toe) or `UTTT` (ultimate tic-tac-toe). |
| `rounds` | Number of self-play → train cycles. Each round generates games, then trains on the accumulated data. |
| `gamesPerRound` | Self-play games generated per round. |
| `epochs` | Training epochs per round on the current dataset. |
| `batchSize` | Minibatch size during supervised training. |
| `mctsIters` | MCTS simulations per move for both players (`--player1-iters` and `--player2-iters`). |
| `player1Type` | Player 1 engine: `random`, `mcts` (pure MCTS), or `nmcts` (MCTS guided by the neural net). |
| `player2Type` | Player 2 engine; same choices as `player1Type`. |

Smoke defaults use smaller workloads on CPU (`mcts` vs `mcts`). GPU defaults target a larger `UTTT` run with `nmcts` players.

Cloud builds always enable `--augment-train` and `--deduplicate-train`. Other `run_pipeline.py` flags (learning rate, loss weights, val split, etc.) are not exposed in this config and keep their Python defaults.

#### Timeouts (`timeouts`)

| Field | Used by | Description |
|-------|---------|-------------|
| `maxRuntimeSeconds` | CPU smoke (CodeBuild) | Linux `timeout` around `run_pipeline.py` inside the build. |
| `codeBuildTimeoutMinutes` | CodeBuild project | Hard AWS cap on total build duration. On deploy, this is raised automatically to at least `maxRuntimeSeconds / 60`. |
| `codeBuildQueuedTimeoutMinutes` | CodeBuild project | Max time a build may wait in the queue before AWS cancels it. |
| `maxTrainingSeconds` | GPU EC2 | Max seconds spent in `run_pipeline.py` on the instance. |
| `maxInstanceSeconds` | GPU EC2 | Total instance wall-clock budget (bootstrap, install, training, upload). The instance shuts down when this is reached or when the script exits. |

**Redeploy required:** changing `codeBuildTimeoutMinutes`, `codeBuildQueuedTimeoutMinutes`, or `maxRuntimeSeconds` (when it affects the CodeBuild project cap). Run `.\scripts\run_cloud_pipeline.ps1 -DeployOnly` after edits.

**No redeploy needed:** changing hyperparameters under `smoke` or `gpu`, or per-run timeout overrides (see below). GPU values are sent as EC2 tags on launch; CPU values are sent as CodeBuild environment overrides.

#### Per-run overrides

Script flags override the config file for a single run without editing JSON:

| Flag | Applies to | Config field |
|------|--------------|--------------|
| `-GameType` | CPU, GPU | `gameType` |
| `-Rounds` | CPU, GPU | `rounds` |
| `-GamesPerRound` | CPU, GPU | `gamesPerRound` |
| `-Epochs` | CPU, GPU | `epochs` |
| `-BatchSize` | CPU, GPU | `batchSize` |
| `-MctsIters` | CPU, GPU | `mctsIters` |
| `-Player1Type` | CPU, GPU | `player1Type` |
| `-Player2Type` | CPU, GPU | `player2Type` |
| `-MaxRuntimeSeconds` | CPU only | `timeouts.maxRuntimeSeconds` |
| `-MaxTrainingSeconds` | GPU only | `timeouts.maxTrainingSeconds` |
| `-MaxInstanceSeconds` | GPU only | `timeouts.maxInstanceSeconds` |
| `-ConfigPath` | CPU, GPU | Path to an alternate JSON config file |

Examples:

```powershell
# CPU: heavier smoke run, longer training timeout
.\scripts\run_cloud_pipeline.ps1 -Rounds 5 -GamesPerRound 100 -MaxRuntimeSeconds 7200

# GPU: more rounds, longer training window
.\scripts\run_cloud_pipeline.ps1 -Gpu -Rounds 10 -MaxTrainingSeconds 7200
```

If `-MaxRuntimeSeconds` exceeds the deployed CodeBuild project timeout, the run script prints a warning; redeploy after raising `timeouts.codeBuildTimeoutMinutes` in the config.

### 2. Deploy the stack

Defaults in `config/cloud-training.json` are baked into the CodeBuild project at deploy time. See [Cloud training configuration](#cloud-training-configuration-configcloud-trainingjson) for field descriptions.

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

### 3. Run CPU smoke training

Packages source, uploads to S3, and runs CodeBuild:

```powershell
.\scripts\run_cloud_pipeline.ps1
```

Artifacts appear under `s3://<artifacts-bucket>/runs/<build-id>/checkpoints/`.

Per-run overrides are described in [Cloud training configuration](#cloud-training-configuration-configcloud-trainingjson). Quick example:

```powershell
.\scripts\run_cloud_pipeline.ps1 -Rounds 5 -Epochs 15 -MaxRuntimeSeconds 7200
```

### 4. Run GPU training

Launches an on-demand `g4dn.xlarge`, trains with CUDA, uploads checkpoints, then shuts down. The launch script returns immediately; use the check script to monitor progress:

```powershell
.\scripts\run_cloud_pipeline.ps1 -Gpu
# or
.\scripts\run_gpu_training.ps1

# Check status and recent logs
.\scripts\check_gpu_training.ps1

# Poll until the manifest appears or the instance stops
.\scripts\check_gpu_training.ps1 -Follow
```

Training limits and GPU defaults come from `config/cloud-training.json`. Per-run overrides:

```powershell
.\scripts\run_cloud_pipeline.ps1 -Gpu -Rounds 10 -MaxTrainingSeconds 7200
```

The instance always shuts down when the script exits. Instance logs: `/var/log/nnmcts-gpu-train.log` (fetched via SSM by the check script).

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

- **CodeBuild GPU** is not available in all regions (e.g. `us-west-1` N. California). GPU training uses EC2 instead.
- The GPU AMI in the CDK stack is pinned for `us-west-1`. For other regions, update the AMI lookup in `infra/lib/nnmcts-pipeline-stack.ts`.

## Publishing / hygiene

Before committing or making the repo public:

```powershell
python scripts\strip_notebook_metadata.py MCTS.ipynb
.\scripts\check_public_safety.ps1
```

Do not commit `infra/cdk.out/`, `infra/cdk.context.json`, `infra/node_modules/`, `config/local.json`, or model checkpoints (`.pt`, `.pkl`).


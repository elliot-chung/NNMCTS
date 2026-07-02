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

- GPU training uses EC2 `g4dn.xlarge` instances.
- The GPU AMI in the CDK stack is pinned for `us-west-1`. For other regions, update the AMI lookup in `infra/lib/nnmcts-pipeline-stack.ts`.

## Publishing / hygiene

Before committing or making the repo public:

```powershell
python scripts\strip_notebook_metadata.py MCTS.ipynb
.\scripts\check_public_safety.ps1
```

Do not commit `infra/cdk.out/`, `infra/cdk.context.json`, `infra/node_modules/`, `config/local.json`, or model checkpoints (`.pt`, `.pkl`).


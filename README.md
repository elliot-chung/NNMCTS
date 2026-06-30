# Neural Monte Carlo Tree Search

An MCTS implementation combined with neural policy/value networks, trained via supervised learning on self-play data.

## Overview

AlphaGo used neural networks with MCTS; this project follows the same approach. It includes:
- MCTS from scratch
- PyTorch models for policy and value
- Data generation via parallel self-play
- Supervised training with augmentation and deduplication
- Optional AWS pipeline for CPU smoke tests and GPU training

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

### 2. Deploy the stack

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

### 4. Run GPU training

Launches an on-demand `g4dn.xlarge`, trains with CUDA, uploads checkpoints, then shuts down:

```powershell
.\scripts\run_cloud_pipeline.ps1 -Gpu
# or
.\scripts\run_gpu_training.ps1
```

GPU runs are capped at 1 hour. Logs on the instance: `/var/log/nnmcts-gpu-train.log` (reachable via SSM Session Manager).

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

## Technical highlights

### Monte Carlo Tree Search

Four stages: selection (UCB1), expansion, rollout (policy/value from network), backpropagation. `NeuralNode` guides search with the trained network.

### Neural architecture

`TTTNet` and `UTTTNet` use CNNs with separate policy and value heads, canonical state handling, and valid-move masks (UTTT).

### Data pipeline

Multiprocess self-play, symmetry augmentation, deduplication with averaged labels, and train/val splits.

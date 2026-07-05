# Scripts

Helper scripts for AWS GPU training, checkpoint export, and demo setup. Run all commands from the **repository root** unless noted otherwise.

For local training without AWS, use the top-level entry points (`run_pipeline.py`, `train_model.py`, `play_matches.py`) described in the main [README](../README.md).

## Overview

| Category | Scripts | Purpose |
|----------|---------|---------|
| AWS cloud pipeline | `run_cloud_pipeline.ps1`, `run_gpu_training.ps1`, `resume_gpu_training.ps1`, `check_gpu_training.ps1`, `teardown_cloud_pipeline.ps1` | Deploy infrastructure, launch GPU training on EC2, resume from a local checkpoint, monitor runs, tear down |
| Source packaging | `package_source.ps1`, `package_source.py` | Zip the repo (excluding checkpoints, `node_modules`, etc.) for S3 upload |
| Model → demo | `download_checkpoint.ps1`, `export_onnx.py`, `validate_onnx.py`, `generate_uttt_fixtures.py` | Pull a trained checkpoint, export ONNX for the browser demo, validate parity, regenerate test fixtures |
| Development | `benchmark_selfplay.py` | Measure self-play throughput and MCTS timing |
| Publishing | `check_public_safety.ps1` | Scan the repo for account IDs, emails, or local paths before making it public |

Internal helpers (`Resolve-AwsConfig.ps1`, `Resolve-CloudTrainingConfig.ps1`) are dot-sourced by the PowerShell scripts above; you do not run them directly.

---

## Prerequisites

### All Python scripts

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Additional packages for ONNX export and validation:

```powershell
pip install onnx onnxscript onnxruntime
```

### AWS scripts (PowerShell)

- **AWS CLI v2** with credentials configured (`aws configure` or `aws login`)
- **Node.js 18+** and **npm** (CDK deploy/destroy under `infra/`)
- **PowerShell** (these scripts are written for Windows; adapt paths for bash if needed)
- GPU quota in your target region if launching training (`g4dn.xlarge`)

Copy and edit local AWS settings:

```powershell
Copy-Item config\local.example.json config\local.json
```

`config/local.json` is gitignored. Alternatively set `AWS_PROFILE` and `AWS_REGION` in your environment. Resolution order: script `-Profile`/`-Region` flags → environment variables → `config/local.json` → defaults (`default`, `us-west-1`, `NnmctsPipelineStack`).

Training hyperparameters and timeouts live in [`config/cloud-training.json`](../config/cloud-training.json). See the main README for field descriptions and per-run override flags. Cloud GPU runs default to **CPU self-play** (`playDevice`), including NMCTS inference, and **GPU training** (`trainDevice`).

### Demo app (after exporting ONNX)

```powershell
cd demo
npm install
npm run dev
```

---

## Recommended workflows

### 1. Full AWS GPU training (first time)

1. Configure AWS (`config/local.json` or env vars).
2. Deploy the CDK stack and run smoke test + full training on one instance:

```powershell
.\scripts\run_cloud_pipeline.ps1
```

3. Monitor until the manifest appears:

```powershell
.\scripts\check_gpu_training.ps1 -Follow
```

4. Checkpoints land in S3 under `s3://<artifacts-bucket>/runs/<run-id>/checkpoints/`. Run metadata is saved locally to `artifacts/latest-gpu-run.json`.

### 2. Resume GPU training from a local checkpoint

Bundle a local `.pt` checkpoint into the source zip, launch EC2, and continue the pipeline from that weights file. The checkpoint is copied into the instance at `bundled-checkpoint/<filename>` and installed before training starts. Round numbering continues from the checkpoint name (e.g. `round_020.pt` → next checkpoint is `round_021.pt`) unless you pass `-StartRound`.

Smoke tests are skipped automatically when a checkpoint is provided.

```powershell
# Convenience wrapper (requires -InitialCheckpointPath)
.\scripts\resume_gpu_training.ps1 `
  -InitialCheckpointPath artifacts\gpu-20260701-192839\round_020.pt `
  -Wait

# Equivalent via run_gpu_training.ps1
.\scripts\run_gpu_training.ps1 `
  -InitialCheckpointPath artifacts\gpu-20260701-192839\round_020.pt `
  -Rounds 10 `
  -Wait

# Via the full pipeline wrapper (smoke is skipped when -InitialCheckpointPath is set)
.\scripts\run_cloud_pipeline.ps1 -RunOnly `
  -InitialCheckpointPath artifacts\gpu-20260701-192839\round_020.pt `
  -Wait
```

Typical flow: download a checkpoint locally with `download_checkpoint.ps1`, then launch a resume run pointing at that file.

### 3. Checkpoint → browser demo

After training (or using a published checkpoint):

```powershell
# 1. Download manifest + checkpoint (uses artifacts/latest-gpu-run.json when present)
.\scripts\download_checkpoint.ps1

# 2. Export PyTorch checkpoint to ONNX (adjust path to match your run)
python scripts\export_onnx.py --checkpoint artifacts\<run-id>\round_020.pt

# 3. Verify PyTorch vs ONNX Runtime parity
python scripts\validate_onnx.py --checkpoint artifacts\<run-id>\round_020.pt

# 4. Regenerate golden fixtures for demo tests (optional but recommended after model changes)
python scripts\generate_uttt_fixtures.py --checkpoint artifacts\<run-id>\round_020.pt

# 5. Run the demo
cd demo
npm run dev
```

Replace `<run-id>` with your actual run ID (e.g. `gpu-20260702-140754`). If `download_checkpoint.ps1` succeeded, the checkpoint path is `artifacts\<run-id>\<CheckpointName>` (default `round_020.pt`).

### 4. Tear down AWS resources

```powershell
# Interactive confirmation
.\scripts\teardown_cloud_pipeline.ps1

# Non-interactive, delete S3 artifacts too
.\scripts\teardown_cloud_pipeline.ps1 -Force

# Keep training outputs in S3
.\scripts\teardown_cloud_pipeline.ps1 -KeepArtifacts
```

This does not remove the CDK bootstrap stack (`CDKToolkit`).

### 5. Before publishing the repo

```powershell
.\scripts\check_public_safety.ps1
```

Fix any reported leaks (account IDs, email addresses, `C:\Users\...` paths) before committing or making the repo public.

---

## Script reference

### `run_cloud_pipeline.ps1`

Orchestrates CDK deploy and GPU training. By default: bootstrap CDK (if needed), deploy `NnmctsPipelineStack`, package source, upload to S3, launch one EC2 instance that runs a smoke test then continues with full UTTT training if smoke succeeds.

**When to use:** One-command cloud setup and training; first-time infrastructure deploy.

**Prerequisites:** AWS CLI, npm, `config/local.json` (or env vars), GPU quota.

| Parameter | Description |
|-----------|-------------|
| `-Profile`, `-Region` | AWS profile and region (override config file) |
| `-DeployOnly` | Deploy CDK stack only; do not launch training |
| `-RunOnly` | Skip deploy; assume stack exists and launch training |
| `-SmokeOnly` | Launch smoke test only (`gpuSmoke` profile) |
| `-SkipSmoke` | Skip smoke; launch full GPU training directly |
| `-Wait` | Block until manifest appears in S3 (default for full pipeline) |
| `-ConfigPath` | Alternate path to `cloud-training.json` |
| `-GameType`, `-Rounds`, `-GamesPerRound`, `-Epochs`, `-BatchSize`, `-MctsIters`, `-Player1Type`, `-Player2Type`, `-SelfPlayWorkers` | Per-run training overrides |
| `-InitialCheckpointPath` | Local `.pt` checkpoint to bundle with source and use as `--initial-checkpoint` on the instance (skips smoke) |
| `-StartRound` | First pipeline round number on the instance (default: one past the round encoded in the checkpoint filename) |
| `-MaxTrainingSeconds`, `-MaxInstanceSeconds` | Timeout overrides (full GPU only) |

**Examples:**

```powershell
# Deploy infrastructure only
.\scripts\run_cloud_pipeline.ps1 -DeployOnly

# Full pipeline: smoke + training (waits by default)
.\scripts\run_cloud_pipeline.ps1

# Smoke test only
.\scripts\run_cloud_pipeline.ps1 -SmokeOnly

# Full training only, custom round count
.\scripts\run_cloud_pipeline.ps1 -SkipSmoke -Rounds 10 -MaxTrainingSeconds 7200

# Resume from a local checkpoint (smoke skipped)
.\scripts\run_cloud_pipeline.ps1 -RunOnly `
  -InitialCheckpointPath artifacts\gpu-20260701-192839\round_020.pt `
  -Rounds 10 `
  -Wait
```

Device split (`playDevice` / `trainDevice`) is configured in `config/cloud-training.json`, not via this script's CLI flags.

---

### `run_gpu_training.ps1`

Launches a single on-demand `g4dn.xlarge` GPU instance without going through the full pipeline wrapper. Packages source, uploads to S3, tags the instance with training config from `cloud-training.json`, and writes `artifacts/latest-gpu-run.json`.

**When to use:** Direct GPU runs after the stack is already deployed; smoke-only or training-only launches.

| Parameter | Description |
|-----------|-------------|
| `-TrainingProfile` | `gpuSmoke` or `gpu` (default `gpu`) |
| `-SmokeThenTrain` | Run smoke then full training on the same instance (used internally by `run_cloud_pipeline.ps1`) |
| `-Wait` | Block until manifest appears |
| `-InitialCheckpointPath` | Local `.pt` file to bundle into the source zip and use as the starting checkpoint (skips smoke) |
| `-StartRound` | First pipeline round number (default: inferred from checkpoint filename) |
| Other flags | Same AWS and training overrides as `run_cloud_pipeline.ps1` |

**Examples:**

```powershell
# Full GPU training (stack must exist)
.\scripts\run_gpu_training.ps1

# Smoke test only, wait for completion
.\scripts\run_gpu_training.ps1 -TrainingProfile gpuSmoke -Wait

# Custom training run
.\scripts\run_gpu_training.ps1 -Rounds 10 -MaxTrainingSeconds 7200

# Continue training from a local checkpoint
.\scripts\run_gpu_training.ps1 `
  -InitialCheckpointPath artifacts\my-run\round_020.pt `
  -Rounds 10 `
  -Wait
```

Resume runs use run ID prefix `gpu-resume-`. When `-InitialCheckpointPath` is set, the checkpoint is embedded at `bundled-checkpoint/<filename>` inside the uploaded source zip.

---

### `resume_gpu_training.ps1`

Convenience wrapper around `run_gpu_training.ps1` for continuing training from a local checkpoint. Equivalent to `run_gpu_training.ps1 -TrainingProfile gpu -InitialCheckpointPath ...` with smoke skipped.

**When to use:** After downloading a checkpoint locally, or when you already have a `.pt` file on disk and want a one-flag launch.

| Parameter | Description |
|-----------|-------------|
| `-InitialCheckpointPath` | **Required.** Path to a local `.pt` checkpoint |
| `-StartRound` | Optional first round number (default: inferred from filename, e.g. `round_020.pt` → round 21) |
| `-Rounds`, `-GamesPerRound`, `-Epochs`, etc. | Same training overrides as `run_gpu_training.ps1` |
| `-Wait` | Block until manifest appears |

**Example:**

```powershell
.\scripts\resume_gpu_training.ps1 `
  -InitialCheckpointPath artifacts\gpu-20260701-192839\round_020.pt `
  -Rounds 10 `
  -Wait
```

---

### `check_gpu_training.ps1`

Shows status for the latest GPU run: instance state, S3 manifest, checkpoint listing, and recent CloudWatch log events.

Live logs stream to CloudWatch log group `/nnmcts/gpu-training` (one stream per instance ID). A full log archive is uploaded to `s3://<bucket>/runs/<run-id>/gpu-train.log` when the run completes.

**When to use:** After launching training without `-Wait`, or to inspect a run in progress.

| Parameter | Description |
|-----------|-------------|
| `-MetadataPath` | Path to run metadata JSON (default `artifacts/latest-gpu-run.json`) |
| `-Follow` | Poll every 30s until manifest appears or instance stops |
| `-LogLines` | Number of CloudWatch events to fetch (default 50) |
| `-RunId`, `-InstanceId` | Override IDs from metadata file |

**Examples:**

```powershell
.\scripts\check_gpu_training.ps1

.\scripts\check_gpu_training.ps1 -Follow

.\scripts\check_gpu_training.ps1 -MetadataPath artifacts\latest-gpu-run.json -LogLines 100
```

---

### `teardown_cloud_pipeline.ps1`

Terminates tagged GPU training instances, destroys the CloudFormation stack via CDK, and optionally deletes the versioned S3 artifacts bucket.

| Parameter | Description |
|-----------|-------------|
| `-Force` | Skip confirmation prompt |
| `-KeepArtifacts` | Destroy stack and instances but retain the S3 bucket |
| `-Profile`, `-Region`, `-StackName` | AWS config overrides |

**Examples:** See [Recommended workflows §4](#4-tear-down-aws-resources).

---

### `package_source.ps1` / `package_source.py`

Creates a zip of the repository for upload to S3, excluding `.git`, virtualenvs, `node_modules`, `cdk.out`, checkpoints (`.pt`, `.pkl`), and similar paths. Optionally bundles one checkpoint file at `bundled-checkpoint/<filename>` inside the archive.

**When to use:** Rarely invoked directly; `run_gpu_training.ps1` calls `package_source.ps1` automatically (with `-CheckpointPath` when resuming). Useful for debugging packaging or manual uploads.

| Parameter | Description |
|-----------|-------------|
| `-RepoRoot` | Repository root to zip (default: parent of `scripts/`) |
| `-OutputPath` | Output zip path (default: `$env:TEMP\nnmcts-source.zip`) |
| `-CheckpointPath` | Optional local `.pt` file to include at `bundled-checkpoint/<filename>` |

```powershell
# Default output: $env:TEMP\nnmcts-source.zip
.\scripts\package_source.ps1

# Bundle a checkpoint with the source
.\scripts\package_source.ps1 -CheckpointPath artifacts\my-run\round_020.pt

# Custom paths
python scripts\package_source.py --repo-root . --output dist\nnmcts-source.zip --checkpoint artifacts\my-run\round_020.pt
```

---

### `download_checkpoint.ps1`

Downloads the run manifest and a checkpoint from S3 to `artifacts/<run-id>/`.

**When to use:** Pull training artifacts locally for ONNX export or inspection.

**Prerequisites:** AWS CLI credentials with read access to the artifacts bucket.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Bucket` | From `artifacts/latest-gpu-run.json` or hardcoded fallback | S3 artifacts bucket |
| `-RunId` | From metadata or fallback | Run ID under `runs/<run-id>/` |
| `-CheckpointName` | `round_020.pt` | Checkpoint filename |
| `-RunManifest` | `artifacts/latest-gpu-run.json` | Local file with bucket/runId/profile/region |
| `-Profile`, `-Region` | Resolved via `Resolve-AwsConfig` | AWS credentials |

**Examples:**

```powershell
# Uses artifacts/latest-gpu-run.json when present
.\scripts\download_checkpoint.ps1

# Explicit run
.\scripts\download_checkpoint.ps1 -RunId gpu-20260702-140754 -CheckpointName round_020.pt

# Specific checkpoint from a long run
.\scripts\download_checkpoint.ps1 -CheckpointName round_010.pt
```

Outputs:

- `artifacts/<run-id>/manifest.json`
- `artifacts/<run-id>/<CheckpointName>` (e.g. `round_020.pt`)

On failure, the script prints manual `aws s3 cp` commands and the follow-up Python export commands.

---

### `export_onnx.py`

Exports a UTTT PyTorch checkpoint (`.pt`) to ONNX for browser inference via `onnxruntime-web`.

**When to use:** After downloading or training a checkpoint, before running the demo.

**Prerequisites:** Activated venv, `pip install onnx onnxscript onnxruntime`, checkpoint file on disk.

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `artifacts/gpu-20260701-192839/checkpoints/round_020.pt` | Path to `.pt` file |
| `--output` | `demo/public/models/uttt-v1.onnx` | Output ONNX path |
| `--opset` | `17` | ONNX opset version |

**Examples:**

```powershell
python scripts\export_onnx.py --checkpoint artifacts\gpu-20260702-140754\round_020.pt

python scripts\export_onnx.py --checkpoint artifacts\my-run\round_020.pt --output demo\public\models\uttt-v1.onnx
```

---

### `validate_onnx.py`

Runs a parity check: compares PyTorch export wrapper output against ONNX Runtime on random UTTT positions.

**When to use:** After `export_onnx.py` to confirm the browser model matches training weights.

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | See `export_onnx.py` | PyTorch checkpoint |
| `--onnx` | `demo/public/models/uttt-v1.onnx` | Exported ONNX file |
| `--samples` | `32` | Number of random positions |
| `--max-abs-diff` | `1e-4` | Fail threshold |
| `--seed` | `42` | Random seed |

**Examples:**

```powershell
python scripts\validate_onnx.py --checkpoint artifacts\gpu-20260702-140754\round_020.pt

python scripts\validate_onnx.py --samples 64 --max-abs-diff 1e-5
```

Exit code `0` = PASS, `1` = FAIL.

---

### `generate_uttt_fixtures.py`

Generates golden inference fixtures (policy logits + value) for TypeScript tests in the demo app.

**When to use:** After exporting a new model, or when demo inference tests need updated expected values.

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | See `export_onnx.py` | PyTorch checkpoint |
| `--output-dir` | `demo/src/lib/__fixtures__/inference` | Fixture output directory |
| `--random-count` | `8` | Additional random positions |
| `--seed` | `42` | Random seed |

**Example:**

```powershell
python scripts\generate_uttt_fixtures.py --checkpoint artifacts\gpu-20260702-140754\round_020.pt
```

Writes `index.json` and one JSON file per scenario under the output directory.

---

### `benchmark_selfplay.py`

Benchmarks self-play wall time and MCTS phase breakdown using `play_matches.run_matches`.

**When to use:** Profiling throughput, comparing player types, or tuning worker settings.

**Prerequisites:** Activated venv with PyTorch; optional checkpoint for `nmcts` players.

| Argument | Default | Description |
|----------|---------|-------------|
| `--game-type` | `UTTT` | `TTT` or `UTTT` |
| `--num-games` | `5` | Games to play |
| `--device` | Auto (CUDA if available) | PyTorch device |
| `--workers` | `1` | Parallel workers |
| `--player1-type`, `--player2-type` | `nmcts` | `random`, `mcts`, or `nmcts` |
| `--player1-iters`, `--player2-iters` | `50` | MCTS simulations per move |
| `--player1-model`, `--player2-model` | None | Checkpoint paths for NMCTS |
| `--output` | None | Optional JSON report path |

**Examples:**

```powershell
# Quick UTTT benchmark with default NMCTS settings
python scripts\benchmark_selfplay.py

# TTT, pure MCTS, save report
python scripts\benchmark_selfplay.py --game-type TTT --player1-type mcts --player2-type mcts --num-games 20 --output benchmark.json

# NMCTS with a local checkpoint
python scripts\benchmark_selfplay.py --player1-model artifacts\my-run\round_020.pt --player2-model artifacts\my-run\round_020.pt --workers 4
```

Prints a JSON summary to stdout.

---

### `check_public_safety.ps1`

Recursively scans the repo (skipping `.git`, `node_modules`, virtualenvs, etc.) for patterns that should not be committed publicly: 12-digit AWS account IDs, email addresses, and Windows user paths (`C:\Users\...`).

**When to use:** Before open-sourcing or sharing the repository.

```powershell
.\scripts\check_public_safety.ps1

# Scan a different root
.\scripts\check_public_safety.ps1 -Root C:\path\to\NNMCTS
```

Exit code `0` = clean, `1` = potential issues listed to stdout.

---

## Troubleshooting

### AWS / cloud training

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `Stack 'NnmctsPipelineStack' was not found` | Infrastructure not deployed | `.\scripts\run_cloud_pipeline.ps1 -DeployOnly` |
| `CDK deploy failed` | npm/CDK or IAM issues | Check `infra/` logs; ensure `CDK_DEFAULT_ACCOUNT` resolves via `aws sts get-caller-identity` |
| Instance ends without manifest | Bootstrap or training error | `.\scripts\check_gpu_training.ps1 -LogLines 200`; check `s3://<bucket>/runs/<run-id>/gpu-train.log` |
| GPU launch fails | Quota or wrong region | Request **Running On-Demand G and VT instances** quota; GPU AMI is pinned for `us-west-1` in CDK (see main README) |
| Wrong AWS account/region | Config mismatch | Verify `config/local.json`, `$env:AWS_PROFILE`, `$env:AWS_REGION` |
| `Initial checkpoint not found` on launch | Bad `-InitialCheckpointPath` | Use a local path to an existing `.pt` file; download first with `download_checkpoint.ps1` if needed |
| Resume run overwrites low round numbers | Expected without `-StartRound` | Pass `-StartRound` explicitly, or name checkpoint `round_NNN.pt` so the next round is inferred |

### Checkpoint download and ONNX export

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `Checkpoint not found` after download | Path mismatch | Download saves to `artifacts\<run-id>\round_020.pt`, not `...\checkpoints\round_020.pt`. Pass `--checkpoint` explicitly to Python scripts. |
| Download uses wrong bucket/run | Stale or missing metadata | Check `artifacts/latest-gpu-run.json`; override with `-Bucket` and `-RunId`, or delete the file to use script defaults. |
| `validate_onnx.py` FAIL | Stale ONNX or export bug | Re-run `export_onnx.py`; increase `--samples` to reproduce; check `--max-abs-diff` |
| ONNX export encoding error on Windows | PyTorch dynamo exporter | Script already sets `dynamo=False`; ensure `onnx` and `onnxscript` are installed |
| Missing ONNX packages | Extra deps not in `requirements.txt` | `pip install onnx onnxscript onnxruntime` |

### Demo

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| Demo loads but plays poorly | Old ONNX in `demo/public/models/` | Re-export and validate; hard-refresh browser |
| Demo tests fail after model update | Stale fixtures | Run `generate_uttt_fixtures.py` after export |

### Public safety scan

The scan may flag legitimate-looking 12-digit numbers or example bucket names embedded in scripts (e.g. default bucket in `download_checkpoint.ps1`). Review each hit; replace or redact account-specific values before publishing.

---

## Related documentation

- [Main README](../README.md) — project overview, local training, AWS setup, `cloud-training.json` reference
- [demo/README.md](../demo/README.md) — Next.js demo and ONNX model notes
- [config/local.example.json](../config/local.example.json) — AWS profile/region template
- [config/cloud-training.json](../config/cloud-training.json) — GPU smoke and training defaults (`playDevice`, `trainDevice`, rounds, workers, etc.)

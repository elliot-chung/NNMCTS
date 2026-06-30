#!/bin/bash
set -euo pipefail
exec > /var/log/nnmcts-gpu-train.log 2>&1

readonly MAX_INSTANCE_SECONDS=5400
readonly MAX_TRAINING_SECONDS=3600
START_TIME=$(date +%s)

shutdown_instance() {
  echo "$(date -Is) Shutting down instance."
  /sbin/shutdown -h now 2>/dev/null || shutdown -h now || true
}

on_exit() {
  local code=$?
  if [[ $code -ne 0 ]]; then
    echo "$(date -Is) Exiting with status ${code}."
  fi
  shutdown_instance
}
trap on_exit EXIT

get_tag() {
  local key="$1"
  curl -fsS "http://169.254.169.254/latest/meta-data/tags/instance/${key}"
}

seconds_remaining() {
  echo $(( MAX_INSTANCE_SECONDS - ($(date +%s) - START_TIME) ))
}

require_time_remaining() {
  if (( $(seconds_remaining) <= 0 )); then
    echo "$(date -Is) Instance wall-clock limit (${MAX_INSTANCE_SECONDS}s) reached."
    exit 124
  fi
}

write_manifest() {
  local status="$1"
  python3 -c "
import json, pathlib
checkpoints = sorted(pathlib.Path('${OUTPUT_DIR}/checkpoints').glob('*.pt'))
manifest = {
    'run_id': '${RUN_ID}',
    'game_type': 'UTTT',
    'device': 'cuda',
    'rounds': 5,
    'games_per_round': 100,
    'epochs': 20,
    'latest_checkpoint': checkpoints[-1].name if checkpoints else None,
    'status': '${status}',
}
pathlib.Path('manifest.json').write_text(json.dumps(manifest, indent=2))
"
}

upload_artifacts() {
  local status="$1"
  write_manifest "${status}"
  if [[ -d "${OUTPUT_DIR}/checkpoints" ]] && compgen -G "${OUTPUT_DIR}/checkpoints/"'*.pt' > /dev/null; then
    aws s3 sync "${OUTPUT_DIR}/checkpoints/" "s3://${BUCKET}/runs/${RUN_ID}/checkpoints/"
  fi
  aws s3 cp manifest.json "s3://${BUCKET}/runs/${RUN_ID}/manifest.json"
}

REGION=$(curl -fsS http://169.254.169.254/latest/meta-data/placement/region)
BUCKET=$(get_tag "nnmcts-bucket")
SOURCE_KEY=$(get_tag "nnmcts-source-key")
RUN_ID=$(get_tag "nnmcts-run-id")

export AWS_DEFAULT_REGION="${REGION}"
WORKDIR=/opt/nnmcts
OUTPUT_DIR="${WORKDIR}/output"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

aws s3 cp "s3://${BUCKET}/${SOURCE_KEY}" source.zip
require_time_remaining

dnf install -y unzip
unzip -qo source.zip -d repo
cd repo
require_time_remaining

python3 -m pip install --quiet --upgrade pip
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
fi
python3 -m pip install --quiet -r requirements.txt
require_time_remaining

export PYTHONPATH="${WORKDIR}/repo"
nvidia-smi
python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

mkdir -p "${OUTPUT_DIR}/datasets" "${OUTPUT_DIR}/checkpoints"

train_limit=$(seconds_remaining)
if (( train_limit > MAX_TRAINING_SECONDS )); then
  train_limit=$MAX_TRAINING_SECONDS
fi
if (( train_limit < 60 )); then
  echo "$(date -Is) Insufficient time remaining (${train_limit}s) for training."
  upload_artifacts "failed"
  exit 1
fi

echo "$(date -Is) Starting training (limit ${train_limit}s, instance limit ${MAX_INSTANCE_SECONDS}s)."
set +e
timeout "${train_limit}" python3 run_pipeline.py \
  --game-type UTTT \
  --rounds 5 \
  --games-per-round 100 \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --player1-type nmcts \
  --player2-type nmcts \
  --player1-iters 75 \
  --player2-iters 75 \
  --epochs 20 \
  --batch-size 128 \
  --augment-train \
  --deduplicate-train
train_exit=$?
set -e

if [[ $train_exit -eq 124 ]]; then
  echo "$(date -Is) Training timed out after ${train_limit}s."
  upload_artifacts "timed_out"
  exit 0
fi

if [[ $train_exit -ne 0 ]]; then
  echo "$(date -Is) Training failed with exit code ${train_exit}."
  upload_artifacts "failed"
  exit "$train_exit"
fi

upload_artifacts "complete"
exit 0

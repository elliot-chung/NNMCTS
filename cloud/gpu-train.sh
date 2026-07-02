#!/bin/bash
set -euo pipefail
exec > /var/log/nnmcts-gpu-train.log 2>&1

readonly DEFAULT_MAX_INSTANCE_SECONDS=5400
readonly DEFAULT_MAX_TRAINING_SECONDS=3600
readonly DEFAULT_GAME_TYPE=UTTT
readonly DEFAULT_ROUNDS=5
readonly DEFAULT_GAMES_PER_ROUND=100
readonly DEFAULT_EPOCHS=20
readonly DEFAULT_BATCH_SIZE=128
readonly DEFAULT_MCTS_ITERS=75
readonly DEFAULT_PLAYER1_TYPE=nmcts
readonly DEFAULT_PLAYER2_TYPE=nmcts

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

IMDS_TOKEN=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

imds_get() {
  curl -fsS "http://169.254.169.254$1" \
    -H "X-aws-ec2-metadata-token: ${IMDS_TOKEN}"
}

get_tag() {
  local key="$1"
  local default_value="${2:-}"
  local value
  value=$(imds_get "/latest/meta-data/tags/instance/${key}" 2>/dev/null || true)
  if [[ -n "${value}" ]]; then
    echo "${value}"
  else
    echo "${default_value}"
  fi
}

get_required_tag() {
  local key="$1"
  local value=""
  local attempt
  for attempt in $(seq 1 30); do
    value=$(imds_get "/latest/meta-data/tags/instance/${key}" 2>/dev/null || true)
    if [[ -n "${value}" ]]; then
      echo "${value}"
      return 0
    fi
    sleep 2
  done
  echo "$(date -Is) Required instance tag missing: ${key}"
  exit 1
}

setup_python_env() {
  if [[ -f /opt/pytorch/bin/activate ]]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/pytorch/bin/activate
    set -u
    echo "$(date -Is) Using DLAMI PyTorch environment."
    return 0
  fi

  echo "$(date -Is) DLAMI PyTorch environment not found; installing CUDA torch via pip."
  python3 -m pip install --quiet --upgrade pip
  python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
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
  python -c "
import json, pathlib
checkpoints = sorted(pathlib.Path('${OUTPUT_DIR}/checkpoints').glob('*.pt'))
manifest = {
    'run_id': '${RUN_ID}',
    'game_type': '${GAME_TYPE}',
    'device': 'cuda',
    'rounds': int('${ROUNDS}'),
    'games_per_round': int('${GAMES_PER_ROUND}'),
    'epochs': int('${EPOCHS}'),
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
  if [[ -f /var/log/nnmcts-gpu-train.log ]]; then
    aws s3 cp /var/log/nnmcts-gpu-train.log "s3://${BUCKET}/runs/${RUN_ID}/gpu-train.log" || true
  fi
}

REGION=$(imds_get "/latest/meta-data/placement/region")
BUCKET=$(get_required_tag "nnmcts-bucket")
SOURCE_KEY=$(get_required_tag "nnmcts-source-key")
RUN_ID=$(get_required_tag "nnmcts-run-id")
MAX_INSTANCE_SECONDS=$(get_tag "nnmcts-max-instance-seconds" "${DEFAULT_MAX_INSTANCE_SECONDS}")
MAX_TRAINING_SECONDS=$(get_tag "nnmcts-max-training-seconds" "${DEFAULT_MAX_TRAINING_SECONDS}")
GAME_TYPE=$(get_tag "nnmcts-game-type" "${DEFAULT_GAME_TYPE}")
ROUNDS=$(get_tag "nnmcts-rounds" "${DEFAULT_ROUNDS}")
GAMES_PER_ROUND=$(get_tag "nnmcts-games-per-round" "${DEFAULT_GAMES_PER_ROUND}")
EPOCHS=$(get_tag "nnmcts-epochs" "${DEFAULT_EPOCHS}")
BATCH_SIZE=$(get_tag "nnmcts-batch-size" "${DEFAULT_BATCH_SIZE}")
MCTS_ITERS=$(get_tag "nnmcts-mcts-iters" "${DEFAULT_MCTS_ITERS}")
PLAYER1_TYPE=$(get_tag "nnmcts-player1-type" "${DEFAULT_PLAYER1_TYPE}")
PLAYER2_TYPE=$(get_tag "nnmcts-player2-type" "${DEFAULT_PLAYER2_TYPE}")

export AWS_DEFAULT_REGION="${REGION}"
WORKDIR=/opt/nnmcts
OUTPUT_DIR="${WORKDIR}/output"

echo "$(date -Is) GPU training bootstrap: run_id=${RUN_ID} bucket=${BUCKET} source_key=${SOURCE_KEY} region=${REGION}"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

aws s3 cp "s3://${BUCKET}/${SOURCE_KEY}" source.zip
require_time_remaining

dnf install -y unzip
unzip -qo source.zip -d repo
cd repo
require_time_remaining

setup_python_env
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy tqdm
require_time_remaining

export PYTHONPATH="${WORKDIR}/repo"
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

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

echo "$(date -Is) Starting training on ${GAME_TYPE} (limit ${train_limit}s, instance limit ${MAX_INSTANCE_SECONDS}s)."
set +e
timeout "${train_limit}" python run_pipeline.py \
  --game-type "${GAME_TYPE}" \
  --rounds "${ROUNDS}" \
  --games-per-round "${GAMES_PER_ROUND}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --player1-type "${PLAYER1_TYPE}" \
  --player2-type "${PLAYER2_TYPE}" \
  --player1-iters "${MCTS_ITERS}" \
  --player2-iters "${MCTS_ITERS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
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

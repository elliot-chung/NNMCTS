#!/bin/bash
set -euo pipefail

readonly DEFAULT_MAX_INSTANCE_SECONDS=5400
readonly CLOUDWATCH_LOG_GROUP="/nnmcts/gpu-training"
readonly CLOUD_INIT_OUTPUT_LOG="/var/log/cloud-init-output.log"
readonly DEFAULT_MAX_TRAINING_SECONDS=3600
readonly DEFAULT_GAME_TYPE=UTTT
readonly DEFAULT_ROUNDS=5
readonly DEFAULT_GAMES_PER_ROUND=100
readonly DEFAULT_EPOCHS=20
readonly DEFAULT_BATCH_SIZE=128
readonly DEFAULT_MCTS_ITERS=75
readonly DEFAULT_PLAYER1_TYPE=nmcts
readonly DEFAULT_PLAYER2_TYPE=nmcts
readonly DEFAULT_SELF_PLAY_WORKERS=3
readonly DEFAULT_PLAY_DEVICE=cpu
readonly DEFAULT_TRAIN_DEVICE=cuda

START_TIME=$(date +%s)
BUCKET=""
RUN_ID=""
OUTPUT_DIR=""
MANIFEST_UPLOADED=0

shutdown_instance() {
  echo "$(date -Is) Shutting down instance."
  /sbin/shutdown -h now 2>/dev/null || shutdown -h now || true
}

upload_artifacts() {
  local status="$1"
  if [[ -z "${BUCKET}" || -z "${RUN_ID}" || -z "${OUTPUT_DIR}" ]]; then
    return 0
  fi

  write_manifest "${status}"
  if [[ -d "${OUTPUT_DIR}/checkpoints" ]] && compgen -G "${OUTPUT_DIR}/checkpoints/"'*.pt' > /dev/null; then
    aws s3 sync "${OUTPUT_DIR}/checkpoints/" "s3://${BUCKET}/runs/${RUN_ID}/checkpoints/"
  fi
  aws s3 cp manifest.json "s3://${BUCKET}/runs/${RUN_ID}/manifest.json"
  if [[ -f "${CLOUD_INIT_OUTPUT_LOG}" ]]; then
    aws s3 cp "${CLOUD_INIT_OUTPUT_LOG}" "s3://${BUCKET}/runs/${RUN_ID}/gpu-train.log" || true
  fi
  MANIFEST_UPLOADED=1
}

on_exit() {
  local code=$?
  if [[ $code -ne 0 && "${MANIFEST_UPLOADED}" -eq 0 && -n "${BUCKET}" && -n "${RUN_ID}" ]]; then
    upload_artifacts "failed" || true
  fi
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

setup_cloudwatch_agent() {
  local instance_id="$1"
  echo "$(date -Is) Configuring CloudWatch agent for ${CLOUDWATCH_LOG_GROUP}/${instance_id}."

  if [[ ! -x /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl ]]; then
    dnf install -y amazon-cloudwatch-agent
  else
    echo "$(date -Is) CloudWatch agent already installed; skipping dnf install."
  fi
  cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<EOF
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "${CLOUD_INIT_OUTPUT_LOG}",
            "log_group_name": "${CLOUDWATCH_LOG_GROUP}",
            "log_stream_name": "${instance_id}",
            "timezone": "UTC"
          }
        ]
      }
    }
  }
}
EOF

  /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -s \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
}

INSTANCE_ID=$(imds_get "/latest/meta-data/instance-id")
( setup_cloudwatch_agent "${INSTANCE_ID}" >> /var/log/nnmcts-cloudwatch.log 2>&1 & )

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
  local phase="${2:-training}"
  python -c "
import json, pathlib
checkpoints = sorted(pathlib.Path('${OUTPUT_DIR}/checkpoints').glob('*.pt'))
manifest = {
    'run_id': '${RUN_ID}',
    'game_type': '${GAME_TYPE}',
    'play_device': '${PLAY_DEVICE}',
    'train_device': '${TRAIN_DEVICE}',
    'device': '${TRAIN_DEVICE}',
    'rounds': int('${ROUNDS}'),
    'games_per_round': int('${GAMES_PER_ROUND}'),
    'epochs': int('${EPOCHS}'),
    'initial_checkpoint_name': '${INITIAL_CHECKPOINT_NAME}' or None,
    'start_round': int('${START_ROUND}') if '${START_ROUND}' else None,
    'latest_checkpoint': checkpoints[-1].name if checkpoints else None,
    'status': '${status}',
    'phase': '${phase}',
}
pathlib.Path('manifest.json').write_text(json.dumps(manifest, indent=2))
"
}

training_limit() {
  local requested="$1"
  local remaining
  remaining=$(seconds_remaining)
  if (( remaining > requested )); then
    echo "${requested}"
  else
    echo "${remaining}"
  fi
}

run_training_phase() {
  local phase="$1"
  local requested_seconds="$2"
  local game_type="$3"
  local rounds="$4"
  local games_per_round="$5"
  local epochs="$6"
  local batch_size="$7"
  local mcts_iters="$8"
  local player1_type="$9"
  local player2_type="${10}"
  local self_play_workers="${11}"
  local play_device="${12}"
  local train_device="${13}"
  local initial_checkpoint="${14:-}"
  local start_round="${15:-1}"

  local limit
  limit=$(training_limit "${requested_seconds}")
  if (( limit < 60 )); then
    echo "$(date -Is) Insufficient time remaining (${limit}s) for ${phase}."
    return 1
  fi

  local resume_msg=""
  if [[ -n "${initial_checkpoint}" ]]; then
    resume_msg=", resume ${initial_checkpoint} from round ${start_round}"
  fi

  echo "$(date -Is) Starting ${phase} on ${game_type} (limit ${limit}s, workers ${self_play_workers}, play ${play_device}, train ${train_device}${resume_msg})."
  set +e
  local -a pipeline_args=(
    --game-type "${game_type}"
    --rounds "${rounds}"
    --games-per-round "${games_per_round}"
    --output-dir "${OUTPUT_DIR}"
    --play-device "${play_device}"
    --train-device "${train_device}"
    --player1-type "${player1_type}"
    --player2-type "${player2_type}"
    --player1-iters "${mcts_iters}"
    --player2-iters "${mcts_iters}"
    --epochs "${epochs}"
    --batch-size "${batch_size}"
    --self-play-workers "${self_play_workers}"
    --amp
    --augment-train
    --deduplicate-train
  )
  if [[ -n "${initial_checkpoint}" ]]; then
    pipeline_args+=(--initial-checkpoint "${initial_checkpoint}")
    pipeline_args+=(--start-round "${start_round}")
  fi

  timeout "${limit}" python run_pipeline.py "${pipeline_args[@]}"
  local train_exit=$?
  set -e

  if [[ $train_exit -eq 124 ]]; then
    echo "$(date -Is) ${phase} timed out after ${limit}s."
    return 124
  fi
  if [[ $train_exit -ne 0 ]]; then
    echo "$(date -Is) ${phase} failed with exit code ${train_exit}."
    return "$train_exit"
  fi

  echo "$(date -Is) ${phase} completed successfully."
  return 0
}

install_bundled_checkpoint() {
  if [[ -z "${INITIAL_CHECKPOINT_NAME}" ]]; then
    return 0
  fi

  local bundled="${WORKDIR}/repo/bundled-checkpoint/${INITIAL_CHECKPOINT_NAME}"
  local dest="${OUTPUT_DIR}/checkpoints/${INITIAL_CHECKPOINT_NAME}"
  if [[ ! -f "${bundled}" ]]; then
    echo "$(date -Is) Bundled checkpoint not found: ${bundled}"
    exit 1
  fi

  echo "$(date -Is) Installing bundled checkpoint ${INITIAL_CHECKPOINT_NAME}"
  cp "${bundled}" "${dest}"
  INITIAL_CHECKPOINT="${dest}"
}

parse_start_round_from_checkpoint() {
  local name="$1"
  if [[ "${name}" =~ round_([0-9]+)\.pt ]]; then
    echo $((10#${BASH_REMATCH[1]} + 1))
  else
    echo 1
  fi
}

resolve_start_round() {
  if [[ -n "${START_ROUND_TAG}" ]]; then
    echo "${START_ROUND_TAG}"
    return
  fi
  if [[ -n "${INITIAL_CHECKPOINT_NAME}" ]]; then
    parse_start_round_from_checkpoint "${INITIAL_CHECKPOINT_NAME}"
    return
  fi
  echo 1
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
SELF_PLAY_WORKERS=$(get_tag "nnmcts-self-play-workers" "${DEFAULT_SELF_PLAY_WORKERS}")
PLAY_DEVICE=$(get_tag "nnmcts-play-device" "${DEFAULT_PLAY_DEVICE}")
TRAIN_DEVICE=$(get_tag "nnmcts-train-device" "${DEFAULT_TRAIN_DEVICE}")
INITIAL_CHECKPOINT_NAME=$(get_tag "nnmcts-initial-checkpoint-name" "")
START_ROUND_TAG=$(get_tag "nnmcts-start-round" "")
INITIAL_CHECKPOINT=""
START_ROUND=1
RUN_SMOKE=$(get_tag "nnmcts-run-smoke" "false")

export AWS_DEFAULT_REGION="${REGION}"
WORKDIR=/opt/nnmcts
OUTPUT_DIR="${WORKDIR}/output"

echo "$(date -Is) GPU training bootstrap: run_id=${RUN_ID} bucket=${BUCKET} source_key=${SOURCE_KEY} region=${REGION} run_smoke=${RUN_SMOKE} checkpoint=${INITIAL_CHECKPOINT_NAME:-none}"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

aws s3 cp "s3://${BUCKET}/${SOURCE_KEY}" source.zip
require_time_remaining

if ! command -v unzip >/dev/null 2>&1; then
  dnf install -y unzip
fi
unzip -qo source.zip -d repo
cd repo
require_time_remaining

bash cloud/install-gpu-deps.sh
if [[ -f /opt/pytorch/bin/activate ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/pytorch/bin/activate
  set -u
fi
require_time_remaining

export PYTHONPATH="${WORKDIR}/repo"
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

mkdir -p "${OUTPUT_DIR}/datasets" "${OUTPUT_DIR}/checkpoints"

if [[ "${RUN_SMOKE}" == "true" ]]; then
  SMOKE_GAME_TYPE=$(get_tag "nnmcts-smoke-game-type" "${GAME_TYPE}")
  SMOKE_ROUNDS=$(get_tag "nnmcts-smoke-rounds" "1")
  SMOKE_GAMES_PER_ROUND=$(get_tag "nnmcts-smoke-games-per-round" "2")
  SMOKE_EPOCHS=$(get_tag "nnmcts-smoke-epochs" "1")
  SMOKE_BATCH_SIZE=$(get_tag "nnmcts-smoke-batch-size" "32")
  SMOKE_MCTS_ITERS=$(get_tag "nnmcts-smoke-mcts-iters" "10")
  SMOKE_PLAYER1_TYPE=$(get_tag "nnmcts-smoke-player1-type" "mcts")
  SMOKE_PLAYER2_TYPE=$(get_tag "nnmcts-smoke-player2-type" "mcts")
  SMOKE_SELF_PLAY_WORKERS=$(get_tag "nnmcts-smoke-self-play-workers" "1")
  SMOKE_PLAY_DEVICE=$(get_tag "nnmcts-smoke-play-device" "${PLAY_DEVICE}")
  SMOKE_TRAIN_DEVICE=$(get_tag "nnmcts-smoke-train-device" "${TRAIN_DEVICE}")
  SMOKE_MAX_TRAINING_SECONDS=$(get_tag "nnmcts-smoke-max-training-seconds" "600")

  if ! run_training_phase \
    "smoke test" \
    "${SMOKE_MAX_TRAINING_SECONDS}" \
    "${SMOKE_GAME_TYPE}" \
    "${SMOKE_ROUNDS}" \
    "${SMOKE_GAMES_PER_ROUND}" \
    "${SMOKE_EPOCHS}" \
    "${SMOKE_BATCH_SIZE}" \
    "${SMOKE_MCTS_ITERS}" \
    "${SMOKE_PLAYER1_TYPE}" \
    "${SMOKE_PLAYER2_TYPE}" \
    "${SMOKE_SELF_PLAY_WORKERS}" \
    "${SMOKE_PLAY_DEVICE}" \
    "${SMOKE_TRAIN_DEVICE}"; then
    upload_artifacts "failed"
    exit 1
  fi

  rm -rf "${OUTPUT_DIR}/datasets" "${OUTPUT_DIR}/checkpoints"
  mkdir -p "${OUTPUT_DIR}/datasets" "${OUTPUT_DIR}/checkpoints"
  require_time_remaining
fi

install_bundled_checkpoint
START_ROUND=$(resolve_start_round)

if ! run_training_phase \
  "training" \
  "${MAX_TRAINING_SECONDS}" \
  "${GAME_TYPE}" \
  "${ROUNDS}" \
  "${GAMES_PER_ROUND}" \
  "${EPOCHS}" \
  "${BATCH_SIZE}" \
  "${MCTS_ITERS}" \
  "${PLAYER1_TYPE}" \
  "${PLAYER2_TYPE}" \
  "${SELF_PLAY_WORKERS}" \
  "${PLAY_DEVICE}" \
  "${TRAIN_DEVICE}" \
  "${INITIAL_CHECKPOINT}" \
  "${START_ROUND}"; then
  train_exit=$?
  if [[ $train_exit -eq 124 ]]; then
    upload_artifacts "timed_out"
    exit 0
  fi
  upload_artifacts "failed"
  exit "$train_exit"
fi

upload_artifacts "complete"
exit 0

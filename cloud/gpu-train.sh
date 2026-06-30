#!/bin/bash
set -euo pipefail
exec > /var/log/nnmcts-gpu-train.log 2>&1

get_tag() {
  local key="$1"
  curl -fsS "http://169.254.169.254/latest/meta-data/tags/instance/${key}"
}

REGION=$(curl -fsS http://169.254.169.254/latest/meta-data/placement/region)
BUCKET=$(get_tag "nnmcts-bucket")
SOURCE_KEY=$(get_tag "nnmcts-source-key")
RUN_ID=$(get_tag "nnmcts-run-id")

export AWS_DEFAULT_REGION="${REGION}"
WORKDIR=/opt/nnmcts
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

aws s3 cp "s3://${BUCKET}/${SOURCE_KEY}" source.zip
dnf install -y unzip
unzip -qo source.zip -d repo
cd repo

python3 -m pip install --quiet --upgrade pip
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
fi
python3 -m pip install --quiet -r requirements.txt

export PYTHONPATH="${WORKDIR}/repo"
nvidia-smi
python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

OUTPUT_DIR="${WORKDIR}/output"
mkdir -p "${OUTPUT_DIR}/datasets" "${OUTPUT_DIR}/checkpoints"

timeout 3600 python3 run_pipeline.py \
  --game-type TTT \
  --rounds 5 \
  --games-per-round 100 \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --player1-type mcts \
  --player2-type mcts \
  --player1-iters 75 \
  --player2-iters 75 \
  --epochs 20 \
  --batch-size 128 \
  --augment-train \
  --deduplicate-train

python3 -c "import json, pathlib; checkpoints = sorted(pathlib.Path('${OUTPUT_DIR}/checkpoints').glob('*.pt')); manifest = {'run_id': '${RUN_ID}', 'game_type': 'TTT', 'device': 'cuda', 'rounds': 5, 'games_per_round': 100, 'epochs': 20, 'latest_checkpoint': checkpoints[-1].name if checkpoints else None, 'status': 'complete'}; pathlib.Path('manifest.json').write_text(json.dumps(manifest, indent=2))"

aws s3 sync "${OUTPUT_DIR}/checkpoints/" "s3://${BUCKET}/runs/${RUN_ID}/checkpoints/"
aws s3 cp manifest.json "s3://${BUCKET}/runs/${RUN_ID}/manifest.json"

shutdown -h now

# #!/usr/bin/env bash
# set -euo pipefail

export ROOT=../../EMR-LLM-CN
export TOKENIZERS_PARALLELISM=false
export SRC_PTH="$ROOT/src"
export CONF_DIR="$SRC_PTH/conf"
export CONF_NAME="train.yaml"

export PYTHONPATH="$ROOT/fairseq"
export CUDA_VISIBLE_DEVICES=0,1,2,3
NGPUS=4

fairseq-hydra-train \
  --config-dir "$CONF_DIR" \
  --config-name "$CONF_NAME" \
  common.user_dir="$SRC_PTH" \
  distributed_training.distributed_world_size=$NGPUS

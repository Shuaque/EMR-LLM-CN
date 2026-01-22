#!/usr/bin/env bash
set -euo pipefail

export ROOT=../../EMR-LLM-CN
SRC_PTH="$ROOT/src_avsr"
CONF_DIR="$SRC_PTH/conf"
CONF_NAME="train_avsr"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/fairseq"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
NGPUS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-hydra-train \
  --config-dir "$CONF_DIR" \
  --config-name "$CONF_NAME" \
  common.user_dir="$SRC_PTH" \
  distributed_training.distributed_world_size="$NGPUS" \
  distributed_training.nprocs_per_node="$NGPUS"
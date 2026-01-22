#!/bin/bash

export ROOT="/workspace/shuaque/EMR-LLM-CN"

export SRC_EMR="$ROOT/src"
export SRC_AVSR="$ROOT/src_avsr"

export PYTHONPATH="$ROOT:$ROOT/fairseq:$SRC_EMR:$SRC_AVSR:$PYTHONPATH"

EMR_CHECKPOINT="$ROOT/pretrained/emr_checkpoint_best.pt"
ONTOLOGY="$ROOT/data/ontology.json"

AVSR_CHECKPOINT="$ROOT/pretrained/avsr_checkpoint_best.pt"
AVSR_DATA_DIR="/workspace/shuaque/CMDD-MIE-EMR-AV" 

echo "======================================================="
echo "Starting REAL Inference Pipeline"
echo "-------------------------------------------------------"
echo "EMR Model:  $EMR_CHECKPOINT"
echo "AVSR Model: $AVSR_CHECKPOINT"
echo "======================================================="

CUDA_VISIBLE_DEVICES=0 python -B $ROOT/inference.py \
    --device "cuda:0" \
    --emr-checkpoint "$EMR_CHECKPOINT" \
    --emr-user-dir "$SRC_EMR" \
    --ontology-path "$ONTOLOGY" \
    --avsr-checkpoint "$AVSR_CHECKPOINT" \
    --avsr-user-dir "$SRC_AVSR" \
    --avsr-data-dir "$AVSR_DATA_DIR"
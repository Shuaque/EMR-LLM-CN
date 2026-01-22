#!/bin/bash

export ROOT="/workspace/shuaque/EMR-LLM-CN"
export SRC_EMR="$ROOT/src"
export SRC_AVSR="$ROOT/src_avsr"

# Ensure Python can find Fairseq and both source directories
export PYTHONPATH="$ROOT:$ROOT/fairseq:$SRC_EMR:$SRC_AVSR:$PYTHONPATH"

# --- EMR Model (Entity Extraction) ---
EMR_CHECKPOINT="$ROOT/pretrained/emr_checkpoint_best.pt"
ONTOLOGY="$ROOT/data/ontology.json"

# --- AVSR Model (Audio/Video Transcription) ---
AVSR_CHECKPOINT="$ROOT/pretrained/avsr_checkpoint_best.pt"
AVSR_DATA_DIR="/workspace/shuaque/CMDD-MIE-EMR-AV" 

# --- AVSR Auxiliary Files (Optional but Recommended) ---
# If your model uses CTC or specific pre-trained encoders, define them here.
# If not needed, you can leave them empty or comment them out.
CTC_VOCAB="$ROOT/data/global_ctc_vocab_3bi.pt"
AVHUBERT_PATH="$ROOT/pretrained/Avhubert/base_vox_iter5.pt"
WHISPER_PATH="$ROOT/pretrained/Whisper/whisper-large"

# ==============================================================================
#  EXECUTION
# ==============================================================================

echo "======================================================="
echo "Starting REAL Inference Pipeline"
echo "-------------------------------------------------------"
echo "EMR Model:  $EMR_CHECKPOINT"
echo "AVSR Model: $AVSR_CHECKPOINT"
echo "AVSR Data:  $AVSR_DATA_DIR"
echo "======================================================="

# Note: Ensure your python script is named 'inference.py' or update the path below.
CUDA_VISIBLE_DEVICES=0 python -B $ROOT/inference.py \
    --device "cuda:0" \
    --emr-checkpoint "$EMR_CHECKPOINT" \
    --emr-user-dir "$SRC_EMR" \
    --ontology-path "$ONTOLOGY" \
    --avsr-checkpoint "$AVSR_CHECKPOINT" \
    --avsr-user-dir "$SRC_AVSR" \
    --avsr-data-dir "$AVSR_DATA_DIR" \
    --avsr-ctc-vocab "$CTC_VOCAB" \
    --avsr-w2v-path "$AVHUBERT_PATH" \
    --avsr-whisper-path "$WHISPER_PATH"
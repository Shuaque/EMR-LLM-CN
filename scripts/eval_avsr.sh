#! /bin/bash

export ROOT=/../../EMR-LLM-CN
export SRC_PTH=$ROOT/src_avsr
export PYTHONPATH=$ROOT:$ROOT/fairseq:$SRC_PTH

LLM_PATH=$ROOT/pretrained/LLM/Qwen2.5-3B-Instruct
Whisper_PATH=$ROOT/pretrained/Whisper/whisper-large
Avhubert_PATH=$ROOT/pretrained/Avhubert/base_vox_iter5.pt
CTC_VOCAB=$ROOT/data/global_ctc_vocab_3bi.pt

DATA=/../../EMR-LLM-CN/data

# Fine-tuned model path (Checkpoint)
MODEL_PATH=$ROOT/pretrained/avsr_checkpoint_best.pt
OUT_PATH=$ROOT/results/

# CUDA_VISIBLE_DEVICES=0 python -B $SRC_PTH/eval.py \
#   --config-dir ${SRC_PTH}/conf \
#   --config-name eval \
#   common.user_dir=${SRC_PTH} \
#   common_eval.path=${MODEL_PATH} \
#   common_eval.results_path=${OUT_PATH} \
#   dataset.gen_subset=test \
#   model.w2v_path=${Avhubert_PATH} \
#   model.whisper_path=${Whisper_PATH} \
#   model.ctc_vocab_path=${CTC_VOCAB} \
#   override.llm_path=${LLM_PATH} \
#   override.data=${DATA} \
#   override.label_dir=${DATA} \
#   override.modalities="['audio','video']" \
#   override.noise_wav=${ROOT}/data/babble_noise.wav \
#   override.noise_prob=0.0 \
#   override.noise_snr=0

INPUT_VIDEO="$ROOT/data/examples/video96/mie_dia_11_win_7.mp4"
INPUT_AUDIO="$ROOT/data/examples/audio16k/mie_dia_11_win_7.wav"

CUDA_VISIBLE_DEVICES=0 python -B $SRC_PTH/eval_single.py \
  --config-dir ${SRC_PTH}/conf \
  --config-name eval \
  common.user_dir=${SRC_PTH} \
  common_eval.path=${MODEL_PATH} \
  common_eval.results_path=${OUT_PATH} \
  dataset.gen_subset=test \
  model.w2v_path=${Avhubert_PATH} \
  model.whisper_path=${Whisper_PATH} \
  model.ctc_vocab_path=${CTC_VOCAB} \
  override.llm_path=${LLM_PATH} \
  override.data=${DATA} \
  override.label_dir=${DATA} \
  override.noise_wav=${ROOT}/data/babble_noise.wav \
  override.noise_prob=0.0 \
  override.noise_snr=0 \
  +override.input_video="$INPUT_VIDEO" \
  +override.input_audio="$INPUT_AUDIO"
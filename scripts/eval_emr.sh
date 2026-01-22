export ROOT=/../../EMR-LLM-CN
export TOKENIZERS_PARALLELISM=false
export SRC_PTH="$ROOT/src"
export PYTHONPATH=$ROOT:$ROOT/fairseq:$SRC_PTH

CHECKPOINT=$ROOT/pretrained/emr_checkpoint_best.pt

CUDA_VISIBLE_DEVICES=0 python3 $SRC_PTH/eval.py \
    --common-user-dir $SRC_PTH \
    --checkpoint-path $CHECKPOINT \
    --split test \
    --device cuda:0 \
    --output-dir $ROOT \
    --ratios 1.0

# CUDA_VISIBLE_DEVICES=0 python $ROOT/inference.py \
#     --common-user-dir /../../EMR-LLM-CN/src \
#     --checkpoint-path $CHECKPOINT \
#     --ontology-path /../../EMR-LLM-CN/data/ontology.json \
#     --device cuda:0
#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7081

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# dev or test
split=$1
split=test

data=../../dataset/pmr/pmr_${split}.tsv
path=./checkpoints_huge_syn_caption_bbox_info/20_2e-5/checkpoint.best_pmr_score_0.8880.pt
result_path=./checkpoints_huge_syn_caption_bbox_info/20_2e-5/
selected_cols=0,2,3,4,5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=pmr \
    --batch-size=8 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
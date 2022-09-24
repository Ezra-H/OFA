#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7051

log_dir=./logs_huge_syn_caption_bbox_info
save_dir=./checkpoints_huge_syn_caption_bbox_info
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

data_dir=../../dataset/pmr_data
data=${data_dir}/snli_ve_train.tsv,${data_dir}/pmr_dev.tsv  # no use in pmr
restore_file=../../checkpoints/ofa_huge.pt
selected_cols=0,2,3,4,5

task=pmr
arch=ofa_huge
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.0
lr=2e-5
max_epoch=20
warmup_ratio=0.06
batch_size=4
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=20
num_bins=1000
patch_image_size=480
prompt_type="prev_output"

export TORCH_DISTRIBUTED_DEBUG=INFO

for max_epoch in {20,}; do
  echo "max_epoch "${max_epoch}
  for lr in {2e-5,}; do
    echo "lr "${lr}

    log_file=${log_dir}/${max_epoch}"_"${lr}".log"
    save_path=${save_dir}/${max_epoch}"_"${lr}
    mkdir -p $save_path

#        --no-reshard-after-forward \
#        --checkpoint-activations \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
        --nproc_per_node=8 --master_port=${MASTER_PORT} ../../train.py \
        $data \
        --ddp-backend fully_sharded \
        --checkpoint-activations \
        --use-syn-caption \
        --gradient-checkpoint-interval 1 \
        --add-bbox-info \
        --selected-cols=${selected_cols} \
        --bpe-dir=${bpe_dir} \
        --user-dir=${user_dir} \
        --restore-file=${restore_file} \
        --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir=${save_path} \
        --task=${task} \
        --arch=${arch} \
        --criterion=${criterion} \
        --label-smoothing=${label_smoothing} \
        --batch-size=${batch_size} \
        --update-freq=${update_freq} \
        --encoder-normalize-before \
        --decoder-normalize-before \
        --share-decoder-input-output-embed \
        --share-all-embeddings \
        --layernorm-embedding \
        --patch-layernorm-embedding \
        --code-layernorm-embedding \
        --resnet-drop-path-rate=${resnet_drop_path_rate} \
        --encoder-drop-path-rate=${encoder_drop_path_rate} \
        --decoder-drop-path-rate=${decoder_drop_path_rate} \
        --dropout=${dropout} \
        --attention-dropout=${attention_dropout} \
        --weight-decay=0.01 \
        --optimizer=cpu_adam \
        --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
        --lr-scheduler=polynomial_decay --lr=${lr} \
        --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
        --log-format=simple --log-interval=10 \
        --fixed-validation-seed=7 \
        --keep-best-checkpoints=1 \
        --save-interval=1 --validate-interval=1 \
        --save-interval-updates=500 --validate-interval-updates=500 \
        --best-checkpoint-metric=pmr_score --maximize-best-checkpoint-metric \
        --max-src-length=${max_src_length} \
        --max-tgt-length=${max_tgt_length} \
        --add-type-embedding \
        --scale-attn \
        --scale-fc \
        --scale-heads \
        --disable-entangle \
        --num-bins=${num_bins} \
        --patch-image-size=${patch_image_size} \
        --prompt-type=${prompt_type} \
        --add-caption \
        --fp16 \
        --fp16-scale-window=512 \
        --num-workers=2 > ${log_file}
  done
done

# 2>&1         --find-unused-parameters \

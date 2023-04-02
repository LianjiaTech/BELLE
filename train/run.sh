#! /bin/bash

REPO_PATH=.

# modelname='decapoda-research/llama-7b-hf'
modelname='bigscience/bloomz-1b7'
# modelname="bigscience/bloomz-7b1-mt"

# dataset_name=Belle_open_source_1M
dataset_name=Belle_open_source_0.5M
dataset_dir=belle_open_source_data
result_dir="train_logs/${modelname}/${dataset_name}_params_lr"

model_config_dir=${modelname}


PREC=16
VAL_CKPT=0.5
MAXLEN=256
max_ans_length=256
BATCH_SIZE=16
accumulate_grad_batches=4
LR=5e-6
MAX_EPOCH=3
GPUS=8
NODES=1
TOKENIZER_TYPE='bpe'
SPEEDUP='deepspeed_stage_2'
max_keep_ckpt=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --devices ${GPUS} \
    --num_nodes ${NODES} \
    --default_root_dir "${result_dir}" \
    --dataset ${dataset_name} \
    --data_dir ${dataset_dir} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --train_file ${dataset_name}.train.json \
    --dev_file ${dataset_name}.dev.json \
    --test_file ${dataset_name}.dev.json \
    --max_keep_ckpt ${max_keep_ckpt} \
    --model_name_or_path ${model_config_dir} \
    --max_length ${MAXLEN} \
    --max_ans_length ${max_ans_length} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --max_epochs ${MAX_EPOCH} \
    --workers 0 \
    --weight_decay 0.0 \
    --warmup_rate 0.03 \
    --betas 0.9 0.95 \
    --seed 42 \
    --lr_scheduler 'cosine' \
    --accumulate_grad_batches ${accumulate_grad_batches} \
    --val_check_interval ${VAL_CKPT} \
    --precision ${PREC} \
    --speedup ${SPEEDUP} \
    --target_loss_only \
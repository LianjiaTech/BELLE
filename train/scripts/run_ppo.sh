#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

BELLE_PATH=".../BELLE"
export PYTHONPATH=$BELLE_PATH/train

export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=allow
# export WANDB_DISABLED=true

model_name_or_path=...
reward_model_name_or_pat=...
output_dir="$BELLE_PATH/saved_models/$WANDB_PROJECT/$WANDB_RUN_ID"
mkdir -p ${output_dir}

train_file=$BELLE_PATH/data/xxx.jsonl
cache_dir=hf_cache_dir
mkdir -p ${cache_dir}


accelerate launch --config_file configs/accelerate_config_ppo.yaml ppo_train.py \
    --ppo_config.model_name $model_name_or_path \
    --ppo_config.reward_model $model_name_or_path \
    --ppo_config.query_dataset $train_file \
    --ppo_config.batch_size 2 \
    --ppo_config.mini_batch_size 1 \
    --ppo_config.gradient_accumulation_steps 2 \
    --ppo_config.ppo_epochs 2 \
    --ppo_config.seed 42 \
    --ppo_config.early_stopping \
    --ppo_config.learning_rate 1.4e-5 \
    --ppo_config.log_with "tensorboard" \
    --data_epochs 1 \
    --cache_dir $cache_dir \
    --output_dir $output_dir \
    --input_length 128
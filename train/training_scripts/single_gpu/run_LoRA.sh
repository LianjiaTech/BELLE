#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
#facebook/opt-1.3b
# Note that usually LoRA needs to use larger learning rate
#/nfs/v100-022/jiyunjie/anaconda3/envs/llamalora/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/v100-022/jiyunjie/anaconda3/envs/llamalora/lib/

OUTPUT_PATH=output-lora
mkdir -p $OUTPUT_PATH


deepspeed --num_gpus 1 main.py \
   --sft_only_data_path belleMath.json \
   --data_split 10,0,0 \
   --model_name_or_path facebook/opt-6.7b \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 1e-3 \
   --weight_decay 0.1 \
   --num_train_epochs 2 \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 0 \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --output_dir $OUTPUT_PATH \
   # &> $OUTPUT_PATH/training.log
#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
#facebook/opt-1.3b
# Note that usually LoRA needs to use larger learning rate

OUTPUT_PATH=output-lora
rm -rf $OUTPUT
mkdir -p $OUTPUT_PATH


deepspeed main.py \
   --sft_only_data_path belleMath.json \
   --data_split 10,0,0 \
   --model_name_or_path facebook/opt-1.3b \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 1024 \
   --learning_rate 2e-4 \
   --weight_decay 0.0001 \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --zero_stage 3 \
   --lora_dim 8 \
   --lora_module_name decoder.layers. \
   --only_optimize_lora \
   --deepspeed \
   --output_dir $OUTPUT_PATH \
   # &> $OUTPUT_PATH/training.log

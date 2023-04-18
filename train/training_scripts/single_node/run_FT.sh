#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT
#bigscience/bloomz-1b7

deepspeed main.py \
   --sft_only_data_path belleMath.json \
   --model_name_or_path decapoda-research/llama-7b-hf \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 1024 \
   --learning_rate 5e-6 \
   --weight_decay 0.0001 \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
#    &> $OUTPUT/training.log

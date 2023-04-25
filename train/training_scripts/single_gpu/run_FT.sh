#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2

rm -rf output/
rm -rf $OUTPUT
mkdir -p $OUTPUT
echo $OUTPUT
echo $ZERO_STAGE
data_output_path=$OUTPUT/data_files
#bigscience/bloomz-1b7
#facebook/opt-1.3b
#bigscience/bloomz-560m

deepspeed --num_gpus 1 main.py \
   --sft_only_data_path belleMath.json \
   --model_name_or_path bigscience/bloomz-560m \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 1024 \
   --learning_rate 5e-5 \
   --weight_decay 0.0001 \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --data_output_path $data_output_path \
#    &> $OUTPUT/training.log

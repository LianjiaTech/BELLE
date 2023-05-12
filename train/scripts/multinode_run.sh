#! /bin/bash

node_rank=$1
echo ${node_rank}

model_name_or_path=/path_to_llm/hf_llama_7b/ # or bloomz-7b1-mt

train_file=belleMath.json
validation_file=belleMath-dev1K.json
output_dir=saved_models
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024

master_addr="10.111.112.223"

# #Multi-node
torchrun --nproc_per_node 8 --nnodes 2 --master_addr ${master_addr} --master_port 14545 --node_rank ${node_rank} src/train.py \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --deepspeed configs/deepspeed_config.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 8e-6 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --fp16 True \
    --seed 1234 \
    --gradient_checkpointing True \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir}

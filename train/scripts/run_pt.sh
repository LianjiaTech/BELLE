#! /bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=allow
export ABS_PATH=...
export PYTHONPATH="$ABS_PATH/BELLE/train"
model_name_or_path=/path_to_llm/hf_llama_7b/ # or bloomz-7b1-mt

train_file=belleMath.json
validation_file=belleMath-dev1K.json
output_dir="$ABS_PATH/BELLE/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=1024

#FT
# torchrun --nproc_per_node 8 src/entry_point/pt_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --deepspeed configs/deepspeed_config.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 8e-6 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#    # --resume_from_checkpoint ...


#LoRA with 8bit
# torchrun --nproc_per_node 8 src/entry_point/pt_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --use_lora \
#     --use_int8_training \
#     --lora_config configs/lora_config_llama.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 8e-6 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#    # --resume_from_checkpoint ...

# LoRA without 8bit
torchrun --nproc_per_node 8 src/entry_point/pt_train.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora \
    --deepspeed configs/deepspeed_config_stage3.json \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 3e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
   # --resume_from_checkpoint ...

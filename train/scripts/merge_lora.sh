#! /bin/bash

model_name_or_path=/path_to_llm/hf_llama_7b/
lora_path=/path_to_lora
output_path=/path_to_saved_weights

CUDA_VISIBLE_DEVICES=0 python src/merge_llama_with_lora.py \
    --model_name_or_path ${model_name_or_path} \
    --output_path ${output_path} \
    --lora_path ${lora_path} \
    --llama
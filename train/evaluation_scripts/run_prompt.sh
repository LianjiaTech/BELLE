#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
python prompt_eval.py \
    --model_name_or_path_baseline /nfs/v100-022/pretrained_ckpt/hf_llama_7b \
    --model_name_or_path_finetune /nfs/a100-80G-20/jiyunjie/finetuned_ckpt/llama/belle_2m_tokenized_llama_epoch=2-step=182298

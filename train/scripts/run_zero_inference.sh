export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export ABS_PATH=...
export PYTHONPATH="$ABS_PATH/BELLE/train"

ckpt_path=BELLE-2/BELLE-Llama2-13B-chat-0.4M
infer_file=$ABS_PATH/BELLE/data/test_data/test_infer.jsonl

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=512

output_dir="$ABS_PATH/BELLE/infer_res"
mkdir -p ${output_dir}

torchrun --nproc_per_node 8 \
     src/entry_point/zero_inference.py \
    --ddp_timeout 36000 \
    --ckpt_path ${ckpt_path} \
    --deepspeed configs/deepspeed_config_stage3_inference.json \
    --infer_file ${infer_file} \
    --per_device_eval_batch_size 1 \
    --model_max_length ${cutoff_len} \
    --torch_dtype "float16" \
    --fp16 \
    --seed 1234 \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --report_to "tensorboard" \
    --llama \
    --use_flash_attention \
    --temperature 0.9 \
    --top_p 0.6 \
    --top_k 30 \
    --num_beams 1 \
    --do_sample \
    --max_new_tokens 128 \
    --min_new_tokens 1 \
    --repetition_penalty 1.2

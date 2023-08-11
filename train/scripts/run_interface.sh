export ABS_PATH=...
export PYTHONPATH="$ABS_PATH/BELLE/train"
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

model_name_or_path=BELLE-2/BELLE-Llama2-13B-chat-0.4M
infer_file=$ABS_PATH/BELLE/data/test_data/test_infer.jsonl

# ft
python src/entry_point/interface.py \
    --model_name_or_path $model_name_or_path \
    --llama \
    --local_rank $1 \
    # --ckpt_path ... \
    # --use_lora

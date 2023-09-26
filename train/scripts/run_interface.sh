export ABS_PATH=...
export PYTHONPATH="$ABS_PATH/BELLE/train"
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

ckpt_path=BELLE-2/BELLE-Llama2-13B-chat-0.4M

# ft
python src/entry_point/interface.py \
    --ckpt_path $ckpt_path \
    --llama \
    --local_rank $1 \
    # --use_lora \
    # --lora_path

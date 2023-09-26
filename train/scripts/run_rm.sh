export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpus=8

BELLE_PATH=".../BELLE"
export PYTHONPATH="$BELLE_PATH/train"

export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=...

model_name_or_path="..."
output_dir="$BELLE_PATH/saved_models/$WANDB_PROJECT/$WANDB_RUN_ID"
mkdir -p ${output_dir}

train_file=$BELLE_PATH/data/xxx.jsonl
validation_file=$BELLE_PATH/data/xxx.jsonl
cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=64

accelerate launch \
    --config_file configs/accelerate_config_rm.yaml \
    --num_processes $gpus \
    "src/entry_point/rm_train.py" \
    --model_name $model_name_or_path \
    --train_data $train_file \
    --eval_data $validation_file \
    --cache_dir $cache_dir \
    --report_to "tensorboard" \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 2 \
    --seq_length $cutoff_len \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --load_in_8bit False \
    --load_in_4bit False \
    --use_lora False \
    --trust_remote_code True \
    --output_dir $output_dir \
    --use_llama True

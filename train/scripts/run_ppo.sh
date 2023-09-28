export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
gpus=8

BELLE_PATH=".../BELLE"
export PYTHONPATH=$BELLE_PATH/train

export WANDB_PROJECT=...
export WANDB_RUN_ID=...
export WANDB_RESUME=allow

model_name_or_path=...
output_dir="$BELLE_PATH/saved_models/$WANDB_PROJECT/$WANDB_RUN_ID"
mkdir -p ${output_dir}

train_file=$BELLE_PATH/data/xxx.jsonl
cache_dir=hf_cache_dir
mkdir -p ${cache_dir}

accelerate launch \
    --config_file configs/accelerate_config_ppo.yaml \
    --num_processes $gpus \
    --main_process_port 29600 \
    "src/entry_point/ppo_train.py" \
    --model_name $model_name_or_path \
    --reward_model_name $model_name_or_path \
    --train_data $train_file \
    --cache_dir $cache_dir \
    --adafactor False \
    --save_freq 100 \
    --output_max_length 128 \
    --batch_size 32 \
    --mini_batch_size 2 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --ppo_epochs 2 \
    --data_epochs 1 \
    --seed 42 \
    --learning_rate 1.4e-5 \
    --early_stopping True \
    --do_sample True \
    --output_dir $output_dir \
    --log_with "tensorboard" \
    --logging_dir "$output_dir/logs" \
    --use_llama True \
    --reward_model_use_llama True \
    --use_lora False \
    --input_length 512 

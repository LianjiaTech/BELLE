export ABS_PATH=...
export PYTHONPATH="$ABS_PATH/BELLE/train"
devices="0,1,2,3,4,5,6,7"

model_name_or_path=BELLE-2/BELLE-Llama2-13B-chat-0.4M

deepspeed --include localhost:${devices} \
    src/entry_point/zero_inference_backend_without_trainer.py \
    --deepspeed configs/deepspeed_config_stage3_inference.json \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --base_port 17860

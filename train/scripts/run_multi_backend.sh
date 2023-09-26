ckpt_path='...'
lora_path='...'
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MKL_SERVICE_FORCE_INTEL='1'

# ft
# python scripts/run_multi_backend.py \
#     --command "python ../src/entry_point/interface.py --ckpt_path $ckpt_path --llama"

# lora
python scripts/run_multi_backend.py \
    --command "python ../src/entry_point/interface.py --ckpt_path $ckpt_path --lora_path $lora_path --use_lora --llama"

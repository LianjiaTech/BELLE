model_name_or_path='...'
ckpt_path='...'
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MKL_SERVICE_FORCE_INTEL='1'

# python scripts/run_multi_backend.py \
#     --command "python ../src/entry_point/interface.py --model_name_or_path $model_name_or_path --ckpt_path $ckpt_path"

python scripts/run_multi_backend.py \
    --command "python ../src/entry_point/interface.py --model_name_or_path $model_name_or_path --ckpt_path $ckpt_path --use_lora"

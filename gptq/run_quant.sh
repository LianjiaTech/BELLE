CUDA_VISIBLE_DEVICES=0 python3 bloom.py \
  --model=BelleGroup/BELLE-7B-2M \
  --dataset=wikitext2 \
  --wbits=8 \
  --groupsize=128 \
  --save=BELLE-7B-gptq/bloom7b-2m-8bit-128g.pt
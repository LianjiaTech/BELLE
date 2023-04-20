# GPTQ-for-Bloom & LLaMa
8 bits quantization of [Bloom](https://arxiv.org/pdf/2211.05100.pdf) using [GPTQ](https://arxiv.org/abs/2210.17323)

GPTQ is SOTA one-shot weight quantization method

**This code is based on [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)**

## [Huggingface models](https://huggingface.co/BelleGroup/BELLE-7B-gptq) 


| model name       |  file size | GPU memory usage |
| -------------------------------------------------- |  ------------------- | ------------------ |
|           base                 |          27G        |       ~28.2G         |
|           bloom7b-2m-8bit-128g.pt                  |          9.7G        |       ~11.4G          |
|           bloom7b-2m-4bit-128g.pt                  |          6.9G        |        ~8.4G          |
|           bloom7b-0.2m-8bit-128g.pt                  |          9.7G        |       ~11.4G          |
|           bloom7b-0.2m-4bit-128g.pt                  |          6.9G        |        ~8.4G          |


All experiments were run on a single NVIDIA A100.

## Installation
If you don't have [conda](https://docs.conda.io/en/latest/miniconda.html), install it first.
```
conda create --name gptq python=3.9 -y
conda activate gptq
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Or, if you're having trouble with conda, use pip with python3.9:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
python setup_cuda.py install

# Benchmark performance for FC2 layer of LLaMa-7B
CUDA_VISIBLE_DEVICES=0 python test_kernel.py
```
## Dependencies

* `torch`: tested on v2.0.0+cu117
* `transformers`: tested on v4.28.0.dev0
* `datasets`: tested on v2.10.1
* `safetensors`: tested on v0.3.0
* (to run 4-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html, tested on CUDA 11.7)


## Model inference with the saved model
```
# BELLE-7B-gptq: local saved model path from Huggingface
git lfs install
git clone https://huggingface.co/BelleGroup/BELLE-7B-gptq
# model inference with the saved model
CUDA_VISIBLE_DEVICES=0 python bloom_inference.py BELLE-7B-gptq --wbits 8 --groupsize 128 --load BELLE-7B-gptq/bloom7b-2m-8bit-128g.pt --text "hello"
```

## Model quantization

```
# BELLE-7B-gptq: local saved model path
# Save compressed model
CUDA_VISIBLE_DEVICES=0 python bloom.py BelleGroup/BELLE-7B-2M wikitext2 --wbits 8 --groupsize 128 --save BELLE-7B-gptq/bloom7b-2m-8bit-128g.pt

```
CUDA Kernels support 2,3,4,8 bits.

Basically, 8-bit quantization and 128 groupsize are recommended.

# Acknowledgements
This code is based on [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)

Thanks to [Bloom](https://arxiv.org/pdf/2211.05100.pdf), a powerful LLM.

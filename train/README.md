*Read this in [English](README_en.md).*

# 项目说明

本仓库用于微调Bloom和Llama两个大语言模型，并且支持LoRA训练

## 环境安装

```bash
conda env create -f environment.yml
conda activate Belle
conda install -c nvidia libcusolver-dev
```

## 数据下载

```bash
python download_data.py
```

创建data_dir文件夹，并且下载我们参考[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 生成的中文数据集[1M](https://huggingface.co/datasets/BelleGroup/train_1M_CN) + [0.5M](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)，同时随机地划分训练和测试数据

## 训练模型

训练模型的配置文件存放在run_config文件夹中

- Bloom_config.json: 配置Bloom模型的超参数
- Llama_config.json: 配置Llama模型的超参数
- deepspeed_config.json: 配置Deepspeed策略的参数
- lora_hyperparams_bloom.json: LoRA训练Bloom模型的参数
- lora_hyperparams_llama.json: LoRA训练Llama模型的参数


训练Bloom模型的启动命令：

```bash
deepspeed --num_gpus=8 finetune.py --model_config_file run_config/Bloom_config.json  --deepspeed run_config/deepspeed_config.json 
```

训练Llama模型的启动命令：

```bash
deepspeed --num_gpus=8 finetune.py --model_config_file run_config/Llama_config.json  --deepspeed run_config/deepspeed_config.json 
```

### LoRA

如果采用LoRA，需要使用torchrun命令启动分布式训练(使用deepspeed启动会出现错误)，同时需要指定use_lora参数并且给出LoRA需要的参数配置文件lora_hyperparams_file

采用LoRA训练的启动命令(Bloom模型):

```bash
torchrun --nproc_per_node=8 finetune.py --model_config_file run_config/Bloom_config.json --lora_hyperparams_file run_config/lora_hyperparams_bloom.json  --use_lora
```

采用LoRA训练的启动命令(Llama模型):

```bash
torchrun --nproc_per_node=8 finetune.py --model_config_file run_config/Llama_config.json --lora_hyperparams_file run_config/lora_hyperparams_llama.json  --use_lora
```

## 文本生成

训练的模型将会保存在trained_models/model_name目录下，其中model_name是模型名，比如Bloom，Llama。假设训练的模型是Bloom，训练数据采用的是Belle_open_source_0.5M，下面的命令将读取模型并生成测试集中每一个样本的生成结果

```bash
python generate.py --dev_file data_dir/Belle_open_source_0.5M.dev.json --model_name_or_path trained_models/bloom/
```

如果是LoRA模型，需要给出LoRA权重保存的位置，如：--lora_weights trained_models/lora-llama

## 参考

本仓库的代码基于[alpaca-lora](https://github.com/tloen/alpaca-lora)

## 常见问题
### 1. torchrun --nproc_per_node=1 finetune.py 启动报错

报错信息如下：
```bash
ValueError: DistributedDataParallel device_ids and output_device arguments only work with single-device/multiple-device GPU modules or CPU modules, but got device_ids [0], output_device 0, and module parameters {device(type='cuda', index=0), device(type='cuda', index=1), device(type='cuda', index=2)}.
```
解决办法：
如果是单张显卡，建议使用如下命令启动：
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py 
```

### 2. RuntimeError: expected scalar type Half but found Float

在跑Bloom模型时，可能会遇到这个问题。经过我们的实验，有如下结论：

- 如果显卡是A100，不会出现expected scalar type Half but found Float的问题，Bloom和Llama都可以跑起来
- 如果显卡是V100，可以跑起来Llama模型，但是Bloom模型就会出现这个错误，此时需要把代码中fp16改为False，才能跑Bloom模型

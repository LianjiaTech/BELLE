



# 项目介绍

本仓库基于[Deepspeed-Chat](https://github.com/microsoft/DeepSpeedExamples)项目，可用于微调大语言模型，包括全量参数微调(fine-tuning)和基于LoRA的参数高效微调。


## 环境安装
我们提供了一个完整可运行的Docker环境，Dockerfile写在Docker文件夹下
```bash
docker build -t  belle:v1.0 --shm-size="10g" -f Docker/Dockerfile  .
docker run -it belle:v1.0 /bin/bash
```



## Run

### Step1 数据下载

```bash
wget https://huggingface.co/datasets/BelleGroup/school_math_0.25M/resolve/main/school_math_0.25M.json
head -n 1000 school_math_0.25M.json > utils/data/dev1K.json
tail -n +1001 school_math_0.25M.json > belleMath.json
```

该数据是数学应用题求解任务，包含解题过程，包含约25万条生成的中文数学题数据。

我们选取前1000条作为验证集，其余数据作为训练集

| Instruction                                                  | Answer                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Human: 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？\n Assistant: | 一箱鸡蛋有25个，小明每天吃9个鸡蛋，用除法计算一下即可： 25 ÷ 9 = 2余7 也就是说，小明第1天到第2天共吃2个鸡蛋，第3天吃剩下的7个鸡蛋。 因此，一箱鸡蛋能够供小明吃2天。 |
| Human: 小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？\n Assistant: | 该题可以用减法来解决。小明手中有15颗糖果，每人分得3颗，所以小红和小李一共分得6颗糖果（3+3=6）。 2. 然后再用原来的糖果总数减去分给小红和小李的糖果数，即可得到小明手中剩余的糖果数。 计算过程如下： 15 - 6 = 9 所以，小明还剩下9颗糖果。 |

我们会在Instruction的开头和结尾给出Human和Assistant

其他的训练数据见：https://huggingface.co/BelleGroup

### Step2 训练



#### 单机多卡

##### Fine-Tuning

```bash
bash training_scripts/single_node/run_FT.sh
```

训练日志记录在当前目录output/training.log中.

##### LoRA-based Tuning

```bash
bash training_scripts/single_node/run_LoRA.sh
```

训练日志记录在当前目录output-lora/training.log中.

#### 单卡

##### Fine-Tuning

```bash
bash training_scripts/single_gpu/run_FT.sh
```

训练日志记录在当前目录output/training.log中.

##### LoRA-based Tuning

```bash
bash training_scripts/single_gpu/run_LoRA.sh
```

训练日志记录在当前目录output-lora/training.log中.



#### 部分参数说明

| Args               | Explanation                                                  |
| ------------------ | ------------------------------------------------------------ |
| model_name_or_path | 默认是decapoda-research/llama-7b-hf，如果想训练Bloom模型，可改为bigscience/bloomz-7b1 |
| sft_only_data_path | 训练数据                                                     |
| lora_dim           | LoRA的秩                                                     |
| lora_module_name   | 指定adapt哪些参数，对于LLaMA模型，我们的实验配置是attention和mlp的参数 |

其余参数说明以及运行所需的机器配置详见：https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/README.md

如果出现显存不足的问题，可参考[Deepspeed-Chat-training_scripts](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts) 中各个启动脚本内的参数配置



## 部分代码实现细节

本仓库实验代码仅对Deepspeed-Chat项目中training/step1_supervised_finetuning内的部分代码做了简单的修改以适配训练LLaMA模型。具体修改内容如下：

1. 需要在utils/data/raw_datasets.py中实现一个类，比如BelleOpenSoucreDataset，用于读取训练数据
2. 由于训练的目标是为了让模型学会回复人类指令，所以我们仅对answer文本计算loss。需要在utils/data/data_utils.py的create_dataset_split方法中修改tokenize部分，在instruction文本部分加上-100作为mask





## 实验结果

我们对LoRA-based tuning和fine-tuning这两种微调策略的效果进行了实验对比，实验结果如下。

FT指的是Fine-Tuning。具体实验细节可参考论文。

| LLaMA     | 训练策略 | 训练数据 | Score |
| --------- | -------- | -------- | ----- |
| LLaMA-7B  | FT       | 2M       | 0.710 |
| LLaMA-7B  | FT       | 0.6M     | 0.686 |
| LLaMA-13B | LoRA     | 2M       | 0.648 |
| LLaMA-7B  | LoRA     | 4M       | 0.624 |
| LLaMA-7B  | LoRA     | 2M       | 0.609 |
| LLaMA-7B  | LoRA     | 0.6M     | 0.589 |



## 致谢

1. [Deepspeed-Chat](https://github.com/microsoft/DeepSpeedExamples)




## 问题反馈

如有问题，请在GitHub Issue中提交。在提交问题前，请先查看 https://github.com/microsoft/DeepSpeedExamples/issues 中是否已出现过解决类似问题的方法。

## FAQ

### transformers

1. transformers的版本是4.28.1

### 机器配置

所有实验均在8卡A100 40G运行


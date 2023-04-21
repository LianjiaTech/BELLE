# 项目介绍

本仓库基于[Deepspeed-Chat](https://github.com/microsoft/DeepSpeedExamples)项目，可用于微调大语言模型，包括全量参数微调(fine-tuning)和基于LoRA的参数高效微调。


## 1. 环境安装

我们提供了一个完整可运行的Docker环境，Dockerfile写在Docker文件夹下. 

```bash
docker build -t  belle:v1.0 --shm-size="10g" -f docker/Dockerfile  .
docker run -it belle:v1.0 /bin/bash
```

考虑到build存在一定的困难，我们提供了镜像下载，你可以使用下面命令从dockerhub拉取我们的镜像。

```shell
docker pull belleagi/belle:v1.0
```

## 2. Run

### 2.1 Download data

```bash
wget https://huggingface.co/datasets/BelleGroup/school_math_0.25M/resolve/main/school_math_0.25M.json
head -n 1000 school_math_0.25M.json > utils/data/dev1K.json
tail -n +1001 school_math_0.25M.json > belleMath.json
```

该数据是数学应用题求解任务，包含解题过程，包含约25万条生成的中文数学题数据。

我们选取前1000条作为验证集，其余数据作为训练集

| Instruction                                                  | Answer                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？ | 一箱鸡蛋有25个，小明每天吃9个鸡蛋，用除法计算一下即可： 25 ÷ 9 = 2余7 也就是说，小明第1天到第2天共吃2个鸡蛋，第3天吃剩下的7个鸡蛋。 因此，一箱鸡蛋能够供小明吃2天。 |
| 小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？ | 该题可以用减法来解决。小明手中有15颗糖果，每人分得3颗，所以小红和小李一共分得6颗糖果（3+3=6）。 2. 然后再用原来的糖果总数减去分给小红和小李的糖果数，即可得到小明手中剩余的糖果数。 计算过程如下： 15 - 6 = 9 所以，小明还剩下9颗糖果。 |

我们会在Instruction的开头和结尾给出Human和Assistant作为模型的输入，形如：

| Instruction                                                  |
| ------------------------------------------------------------ |
| Human: 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？\nAssistant: |
| Human: 小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？\nAssistant: |

其他的训练数据见：https://huggingface.co/BelleGroup

### 2.2 Train

目前支持单机单卡和单机多卡的训练。不同于 [Deepspeed-Chat ](https://github.com/microsoft/DeepSpeedExamples)，我们仅针对stage1，也就是SFT阶段（具体来说是instruction-tuning）。

#### 2.2.1 单机多卡训练

##### Fine-Tuning

如果要实现单机多卡微调，仅需要运行如下命令

```bash
bash training_scripts/single_node/run_FT.sh output 2
```

- output 代表数据和模型保存的路径，如果没有则会创建。

- 2 代表zero_stage

具体参数配置如下：

```bash
deepspeed main.py \
   --sft_only_data_path belleMath.json \
   --model_name_or_path BelleGroup/BELLE-7B-2M \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 1024 \
   --learning_rate 5e-7 \
   --weight_decay 0.0001 \
   --num_train_epochs 1  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --data_output_path $data_output_path \
```

- model_name_or_path就是基础模型。我们建议基于我们开源的模型(如：[BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M)) 作为基础模型进行进一步微调，这样仅需要少量训练数据和训练轮次即可微调一个效果不错的模型。如果您需要尝试原生的Bloom模型，可改为bigscience/bloomz-7b1-mt，此时需要适当的调大learning_rate和num_train_epochs
- zero_stage。可优先设置为1或者2，如果显存不足，设置为3。关于zero-stage的详细介绍可参考： https://www.deepspeed.ai/tutorials/zero/ 

其余参数说明详见：https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/README.md

如果出现显存不足的问题，可参考[Deepspeed-Chat-training_scripts](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts) 中各个启动脚本内的参数配置



#### 2.2.2 单机单卡训练

##### Fine-Tuning

如果要实现单机单卡微调，仅需要运行如下命令

```bash
bash training_scripts/single_gpu/run_FT.sh
```

其余配置与上述内容一致。

### 2.3 Generate

现在我们加载训练好的模型，验证模型生成文本的效果。

```bash
export CUDA_VISIBLE_DEVICES=0 python prompt_eval.py \
    --model_name_or_path model_name_or_path \
    --finetuned_model_name_or_path finetuned_model_name_or_path \
    --test_file utils/data/dev1K.json
```

在run_prompt.sh脚本中

- model_name_or_path 是原生预训练模型的路径
- finetuned_model_name_or_path 是训练后保存的模型

- test_file就是验证集数据，默认路径是utils/data/dev1K.json

模型生成的数据保存在predictions.json文件中

这里给出一些预测的示例:

```bash
{"instruction": "小明有12个橙子，他想把它们分给他的4个朋友，每人分到的橙子数量相同，每人能分到几个橙子？", "generate_text": "Human: 小明有12个橙子，他想把它们分给他的4个朋友，每人分到的橙子数量相同，每人能分到几个橙子？\nAssistant: 将12个橙子平均分成4份，可以用除法来解决。首先，我们需要计算出每一份应该包含多少个橙子：\n12÷4=3\n因此，每个朋友都会得到3个橙子。答案是每人能分到3个橙子。"}
{"instruction": "以下是一道小学数学题：小明家里有 3 只宠物猫和 2 只宠物狗，小花家里有 4 只宠物猫和 1 只宠物狗，谁家里宠物更多？", "generate_text": "Human: 以下是一道小学数学题：小明家里有 3 只宠物猫和 2 只宠物狗，小花家里有 4 只宠物猫和 1 只宠物狗，谁家里宠物更多？\nAssistant: 首先，我们需要计算出两个人的宠物总数。小明家里共有 5 只宠物（3 + 2 = 5）；小花家里共有 6 只宠物（4 + 1 = 5）。因此，两个人的宠物数量相同。所以，无法确定谁家里宠物更多。"}
{"instruction": "题目：小明有5个球，他送给小红2个球，还剩多少个球？", "generate_text": "Human: 题目：小明有5个球，他送给小红2个球，还剩多少个球？\nAssistant: 小明原来有5个球，送出去了2个球后，就是5-2=3。所以小明现在手里只剩下3个球了。答案为3。"}
{"instruction": "请问2+3等于几？", "generate_text": "Human: 请问2+3等于几？\nAssistant: 将两个数字相加，得到答案为5。因此，2+3=5。"}
```

generate方法的参数配置如下：

```bash
max_new_tokens = 1024
generation_config = dict(
    temperature=0.001,
    top_k=30,
    top_p=0.85,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=max_new_tokens
)
```



## 3. Additional Notes

### 3.1 LLaMA模型的使用

#### 3.1.1 facebook官方LLaMA权重转为hf格式

首先，您需要从[facebookresearch/llama](https://github.com/facebookresearch/llama)获取LLaMA模型的访问权限。下载官方检查点并将其保存到

```bash
python training_scripts/convert_llama_weights_to_hf.py --input_dir download_official_llama_path --model_size 7B --output_dir xx/llama-7b-hf
```

运行训练脚本时将model_name_or_path改为xx/llama-7b-hf即可

#### 3.1.2 BELLE-LLaMA转为hf格式

由于LLaMA模型的使用约束，我们只能开源与原始模型的diff（如：[BELLE-LLaMA-7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc)）。当您已经从[facebookresearch/llama](https://github.com/facebookresearch/llama)获取LLaMA模型的访问权限后，可参考 https://github.com/LianjiaTech/BELLE/tree/main/models ，转换后的模型即为我们指令调优后的LLaMA模型。



## 4. 致谢

1. [Deepspeed-Chat](https://github.com/microsoft/DeepSpeedExamples)




## 5. 问题反馈

如有问题，请在GitHub Issue中提交。在提交问题前，请先查看 https://github.com/microsoft/DeepSpeedExamples/issues 中是否已出现过解决类似问题的方法。

**我们的实验均在8卡A100 40G上运行，在之前的实验过程中发现在V100上运行可能会遇到问题。因此如果是在V100上运行报错，请自行查阅相关解决方案，可主要参考 [deepspeed-chat issues]( https://github.com/microsoft/DeepSpeedExamples/issues)**。

## 6. FAQ

我们会持续更新FAQ，并对询问的问题进行分类。Others中给出的是我们在实验过程中遇到的一些报错的情况以及参考的解决方案

- [1. 单机单卡可以训练多大参数量的模型](FAQ.md#1)
- [2. 单机多卡可以训练多大参数量的模型](FAQ.md#2)
- [Others](FAQ.md#Others)





## 7. 部分代码实现细节

本仓库实验代码仅对Deepspeed-Chat项目中training/step1_supervised_finetuning内的部分代码做了简单的修改。具体修改内容如下：

1. 需要在utils/data/raw_datasets.py中实现一个类，比如BelleOpenSoucreDataset，用于读取训练数据
2. 由于训练的目标是为了让模型学会回复人类指令，所以我们仅对answer文本计算loss。需要在utils/data/data_utils.py的create_dataset_split方法中修改tokenize部分，在instruction文本部分加上-100作为mask



## 8. 后续todo

1. 目前基于LoRA训练的代码还不完善，后续会完善基于LoRA训练的代码
2. 目前仅在训练结束后保存checkpoint，后续会根据评估集上的loss来保存模型

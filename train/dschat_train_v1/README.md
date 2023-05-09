# 项目介绍

本目录代码基于[Deepspeed-Chat](https://github.com/microsoft/DeepSpeedExamples)项目，可用于微调大语言模型，包括全量参数微调(fine-tuning)和基于LoRA的参数高效微调。

## 1. 准备环境

我们提供了一个完整可运行的Docker环境，Dockerfile写在docker文件夹下.

考虑到build存在一定的困难，我们提供了镜像下载，你可以使用下面命令从dockerhub拉取我们的镜像，然后在镜像中运行代码。

```shell
docker pull belleagi/belle:v1.0
git clone https://github.com/LianjiaTech/BELLE.git
docker run -it --runtime=nvidia --shm-size="40g" -v /path/BELLE/train:/workspace/BELLE-train -v /path/huggingface_models/:/workspace/model_name_or_path belleagi/belle:v1.0 /bin/bash
```

上述命令实现了以下几点：

1. 拉取docker镜像
2. clone BELLE仓库
3. 将BELLE/train目录与Docker环境中/workspace/BELLE-train目录挂载
4. 将huggingface_models目录与Docker环境中/workspace/model_name_or_path目录挂载。其中huggingface_models代表预训练模型的保存路径，该目录下存放所有需要的预训练语言模型，如llama-7b, bloomz-7b1-mt等
5. 注意：上述挂载的目录必须是绝对路径

## 2. Run

### 2.1 Download data

```bash
wget https://huggingface.co/datasets/BelleGroup/school_math_0.25M/resolve/main/school_math_0.25M.json
```

该数据是数学应用题求解任务，包含解题过程，共计约25万条。示例：

| Instruction                                                                                     | Answer                                                                                                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？                       | 一箱鸡蛋有25个，小明每天吃9个鸡蛋，用除法计算一下即可： 25 ÷ 9 = 2余7 也就是说，小明第1天到第2天共吃2个鸡蛋，第3天吃剩下的7个鸡蛋。 因此，一箱鸡蛋能够供小明吃2天。                                                                    |
| 小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？ | 该题可以用减法来解决。小明手中有15颗糖果，每人分得3颗，所以小红和小李一共分得6颗糖果（3+3=6）。 2. 然后再用原来的糖果总数减去分给小红和小李的糖果数，即可得到小明手中剩余的糖果数。 计算过程如下： 15 - 6 = 9 所以，小明还剩下9颗糖果。 |

#### 2.1.1 Prepare data

```bash
python training_scripts/convert_to_conv_data.py --orig_data school_math_0.25M.json --write_data school_math_0.25M_conv.json --dataset_name bellemath
head -n 1000 school_math_0.25M_conv.json > belleMath-dev1K.json
tail -n +1001 school_math_0.25M_conv.json > belleMath.json
```

我们选取前1000条作为验证集，其余数据作为训练集

我们会在Instruction的开头和结尾加上Human和Assistant作为模型的输入，形如：

| Instruction                                                                                                          |
| -------------------------------------------------------------------------------------------------------------------- |
| Human: 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？\n\nAssistant:                       |
| Human: 小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？\n\nAssistant: |

**注：**目前仓库的代码要兼容单轮和多轮数据，因此对于之前的开源数据(格式为(instruction,output)或者(input,target)。目前绝大部分开源的单轮的instruction数据均为这种格式)，需要用convert_to_conv_data.py脚本转换为如下的统一格式:

```bash
{"id": "bellemath42", "conversations": [{"from": "human", "value": " 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？"}, {"from": "assistant", "value": "\n一箱鸡蛋有25个，小明每天吃9个鸡蛋，用除法计算一下即可：\n25 ÷ 9 = 2余7\n也就是说，小明第1天到第2天共吃2个鸡蛋，第3天吃剩下的7个鸡蛋。\n因此，一箱鸡蛋能够供小明吃2天。"}]}
{"id": "bellemath80", "conversations": [{"from": "human", "value": "小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？"}, {"from": "assistant", "value": " \n1. 该题可以用减法来解决。小明手中有15颗糖果，每人分得3颗，所以小红和小李一共分得6颗糖果（3+3=6）。\n2. 然后再用原来的糖果总数减去分给小红和小李的糖果数，即可得到小明手中剩余的糖果数。 \n计算过程如下：\n15 - 6 = 9\n所以，小明还剩下9颗糖果。"}]}
```

其他的训练数据见：https://huggingface.co/BelleGroup  按照上述流程转换格式即可。

#### 2.1.2 Multi-turn data

[shareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main) 是一个开源的大规模的多轮对话数据，具体效果可参考我们的工作：[Towards Better Instruction Following Language Models for Chinese: Investigating the Impact of Training Data and Evaluation](https://arxiv.org/pdf/2304.07854.pdf)

当前代码已支持训练这种多轮对话数据。数据下载：

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
mv ShareGPT_V3_unfiltered_cleaned_split.json shareGPT.json
```

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

具体启动命令和参数配置如下：

```bash
deepspeed main.py \
   --sft_only_data_path belleMath.json \
   --eval_data_file belleMath-dev1K.json \
   --model_name_or_path /workspace/model_name_or_path/hf_llama_7b \
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

- sft_only_data_path 就是训练集数据。如果想换成shareGPT，仅需替换为shareGPT.json
- eval_data_file 代表验证集数据，如果没有预先划分出训练和验证数据，可以不指定该参数，此时将会从训练数据中随机抽取1000条作为验证数据
- model_name_or_path就是基础模型。我们建议基于我们开源的模型(如：[BelleGroup/BELLE-LLaMA-EXT-7B](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-7B)) 作为基础模型进行进一步微调，这样仅需要少量训练数据和训练轮次即可微调一个效果不错的模型。
- zero_stage。可优先设置为1或者2，如果显存不足，设置为3。关于zero-stage的详细介绍可参考： https://www.deepspeed.ai/tutorials/zero/

##### LoRA

如果要实现单机多卡LoRA-based tuning，需要运行如下命令：

```bash
bash training_scripts/single_node/run_LoRA.sh output-lora 2
```

- output 代表数据和模型保存的路径，如果没有则会创建。
- 2 代表zero_stage

具体启动命令和参数配置如下：

```bash
model_name_or_path=/workspace/model_name_or_path/hf_llama_7b
lora_module_name="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"
echo ${lora_module_name}

deepspeed main.py \
   --sft_only_data_path belleMath.json \
   --eval_data_file belleMath-dev1K.json \
   --data_split 10,0,0 \
   --model_name_or_path ${model_name_or_path} \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 1024 \
   --learning_rate 3e-4 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --lora_dim 16 \
   --lora_alpha 16 \
   --lora_droppout 0.05 \
   --lora_module_name ${lora_module_name} \
   --deepspeed \
   --output_dir $OUTPUT_PATH \
```

- lora_module_name代表LoRA需要adapt的参数，我们的实验设置是attention+MLP的参数。不同的预训练模型的权重名称不一样，比如对于Bloom模型，对应的attention权重的名称是query_key_value，此时lora_module_name可以改为"query_key_value,mlp"
- lora_dim、lora_alpha、lora_droppout均为LoRA训练的超参数

#### 2.2.2 单机单卡训练

##### Fine-Tuning

如果要实现单机单卡微调，仅需要运行如下命令

```bash
bash training_scripts/single_gpu/run_FT.sh output 3
```

其余配置与上述内容一致。

##### LoRA

如果要实现单机单卡LoRA-based tuning，需要运行如下命令：

```bash
bash training_scripts/single_gpu/run_LoRA.sh output-lora 3
```

其余配置与上述内容一致。

如果出现显存不足的情况，需要调整per_device_train_batch_size、max_seq_len、zero_stage三个参数。另外可参考[Deepspeed-Chat-training_scripts](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts) 中各个启动脚本内的参数配置

其余参数说明详见：https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/README.md

**注：**

- 如果是单轮instruction数据，比如 [BELLE-2M](https://huggingface.co/datasets/BelleGroup/train_2M_CN) 等。对于Bloom模型，建议max_seq_len设置为512-1024之间。而对于LLaMA模型，max_seq_len尽可能不要低于1024。避免切割出太多不完整的句子，不利于模型学习。
- 如果是多轮对话数据，比如 [shareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main)，max_seq_len建议设置为2048
- 运行脚本中各个参数的设置需要按实际情况进行调整，下面给出一些参数值供参考：
  - 对于learning_rate，FT训练时设置为5e-6量级，LoRA训练时设置为3e-4量级
  - 对于epoch，FT训练时设置为2或3，LoRA训练时设置为4或5

### 2.3 Generate

现在我们加载训练好的模型，验证模型生成文本的效果。

```bash
CUDA_VISIBLE_DEVICES=0 python prompt_eval.py \
    --model_name_or_path model_name_or_path \
    --finetuned_model_name_or_path finetuned_model_name_or_path \
    --test_file test_file_name_or_path
```

参数说明：

- model_name_or_path 是原生预训练模型的路径
- finetuned_model_name_or_path 是训练后保存的模型
- test_file就是验证集数据，默认路径是belleMath-dev1K.json

举例：

```bash
CUDA_VISIBLE_DEVICES=0 python prompt_eval.py \
    --model_name_or_path /workspace/model_name_or_path/hf_llama_7b \
    --finetuned_model_name_or_path output-lora \
    --test_file belleMath-dev1K.json
```

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

首先，您需要从[facebookresearch/llama](https://github.com/facebookresearch/llama)获取LLaMA模型的访问权限，下载官方检查点

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

**我们的实验均在8卡A100 40G上运行，在之前的实验过程中发现在V100上运行可能会遇到问题。因此如果是在V100上运行报错，请自行查阅相关解决方案，可主要参考 [deepspeed-chat issues](https://github.com/microsoft/DeepSpeedExamples/issues)**。

## 6. FAQ

我们会持续更新FAQ，并对询问的问题进行分类。Others中给出的是我们在实验过程中遇到的一些报错的情况以及参考的解决方案

- [1. 单机单卡可以训练多大参数量的模型](FAQ.md#1)
- [2. 单机多卡可以训练多大参数量的模型](FAQ.md#2)
- [3. 单机单卡采用LoRA可以训练多大参数量的模型](FAQ.md#3)
- [4. 单机多卡采用LoRA可以训练多大参数量的模型](FAQ.md#4)
- [5. 加载Llama tokenizer时存在的问题](FAQ.md#5)
- [6. 加载2M的数据量需要多大的内存和多长时间](FAQ.md#6)
- [7. 训练模型的生成结果非常糟糕](FAQ.md#7)
- [Others](FAQ.md#Others)

## 7. 部分代码实现细节

本仓库实验代码仅对Deepspeed-Chat项目中training/step1_supervised_finetuning内的部分代码做了简单的修改。具体修改内容如下：

1. 需要在utils/data/raw_datasets.py中实现一个类，比如BelleOpenSoucreDataset，用于读取训练数据
2. 由于训练的目标是为了让模型学会回复人类指令，所以我们仅对answer文本计算loss。需要在utils/data/data_utils.py的create_dataset_split方法中修改tokenize部分，在human instruction文本部分对应的label加上-100作为mask。如果是多轮对话数据，每一轮的human instruction对应的label都会加上-100

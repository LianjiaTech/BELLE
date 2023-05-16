# BELLE训练代码

 | [English](https://github.com/LianjiaTech/BELLE/blob/main/train/docs/README_en.md) | [中文](https://github.com/LianjiaTech/BELLE/blob/main/train/README.md) 

当前仓库的代码属于BELLE项目训练代码v2版，上一版基于deepspeed-chat的代码放在dschat_train_v1目录下，未做任何改动。

考虑到如下因素和目前大家提出的issues，我们更新了仓库的训练代码

1. 没有deepspeed环境时无法使用仓库代码训练模型
2. deepspeed-chat没有集成peft包，对参数高效微调这一块的可扩展性不高

当前v2版本的代码对环境的依赖性较低，而且更加简洁。


## 1. 准备环境

### 1.1 Docker镜像

我们提供了一个完整可运行的Docker镜像，Dockerfile写在docker文件夹下.

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

### 1.2 conda

假如由于机器等原因不能使用docker，也可以通过conda创建环境，然后pip安装需要的包

```bash
pip install -r requirements.txt
```

但是通过pip安装deepspeed很有可能安装或者运行失败，[FAQ](https://github.com/LianjiaTech/BELLE/blob/main/train/docs/FAQ.md) 中给出了一些安装deepspeed的教程以及可能遇到的问题

## 2. Run

### 2.1 Download data

```bash
wget https://huggingface.co/datasets/BelleGroup/school_math_0.25M/resolve/main/school_math_0.25M.json
```

该数据是数学应用题求解任务，包含解题过程，共计约25万条。示例：

| Instruction                                                  | Answer                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？ | 一箱鸡蛋有25个，小明每天吃9个鸡蛋，用除法计算一下即可： 25 ÷ 9 = 2余7 也就是说，小明第1天到第2天共吃2个鸡蛋，第3天吃剩下的7个鸡蛋。 因此，一箱鸡蛋能够供小明吃2天。 |
| 小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？ | 该题可以用减法来解决。小明手中有15颗糖果，每人分得3颗，所以小红和小李一共分得6颗糖果（3+3=6）。 2. 然后再用原来的糖果总数减去分给小红和小李的糖果数，即可得到小明手中剩余的糖果数。 计算过程如下： 15 - 6 = 9 所以，小明还剩下9颗糖果。 |

#### 2.1.1 Convert data format

```bash
python scripts/convert_to_conv_data.py --orig_data school_math_0.25M.json --write_data school_math_0.25M_conv.json --dataset_name bellemath
head -n 1000 school_math_0.25M_conv.json > belleMath-dev1K.json
tail -n +1001 school_math_0.25M_conv.json > belleMath.json
```

我们选取前1000条作为验证集，其余数据作为训练集

**注：**目前仓库的代码要兼容单轮和多轮数据，因此对于之前的开源数据(格式为[instruction,output]，目前绝大部分开源的单轮instruction数据均为这种格式)，需要用convert_to_conv_data.py脚本转换为如下的统一格式:

```bash
{"id": "bellemath42", "conversations": [{"from": "human", "value": " 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？"}, {"from": "assistant", "value": "\n一箱鸡蛋有25个，小明每天吃9个鸡蛋，用除法计算一下即可：\n25 ÷ 9 = 2余7\n也就是说，小明第1天到第2天共吃2个鸡蛋，第3天吃剩下的7个鸡蛋。\n因此，一箱鸡蛋能够供小明吃2天。"}]}
{"id": "bellemath80", "conversations": [{"from": "human", "value": "小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？"}, {"from": "assistant", "value": " \n1. 该题可以用减法来解决。小明手中有15颗糖果，每人分得3颗，所以小红和小李一共分得6颗糖果（3+3=6）。\n2. 然后再用原来的糖果总数减去分给小红和小李的糖果数，即可得到小明手中剩余的糖果数。 \n计算过程如下：\n15 - 6 = 9\n所以，小明还剩下9颗糖果。"}]}
```

其他的训练数据见：https://huggingface.co/BelleGroup  按照上述流程转换格式即可。

对于多轮对话数据， [shareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main) 是一个开源的大规模多轮对话数据，具体效果可参考我们的工作：[Towards Better Instruction Following Language Models for Chinese: Investigating the Impact of Training Data and Evaluation](https://arxiv.org/pdf/2304.07854.pdf)

当前代码已支持训练这种多轮对话数据。数据下载：

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```



### 2.2 模型训练

训练的启动脚本写在scripts/run.sh，你需要按照实际需求修改run.sh中的参数

```bash
bash scripts/run.sh
```

- model_name_or_path 代表预训练模型（如果是LLaMA模型，需事先转为hf格式才能通过from_pretrained读取）
- train_file 代表训练数据
- validation_file 代表验证数据
- output_dir 代表训练日志和模型保存的路径
- cache_dir 代表缓存数据处理过程的路径
- cutoff_len 代表最长输入序列长度（LLaMA模型建议设置为1024以上，Bloom模型设置为512以上）

run.sh中包含了全量参数微调和LoRA两种训练方式的启动命令，这里将简单说明下启动命令中各个参数的含义

#### 2.2.1 全量参数微调

下面的命令是单机多卡进行全量参数微调，同时采用deepspeed，基础模型是LLaMA

```bash
torchrun --nproc_per_node 8 train.py \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --deepspeed configs/deepspeed_config.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 8e-6 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --fp16 True \
    --seed 1234 \
    --gradient_checkpointing True \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir}
```

**参数说明**

1. 如果想要单卡训练，仅需将nproc_per_node设置为1即可
2. 如果预训练模型不是LLaMA，则去掉--llama。如果是LLaMA模型，需要指定--llama。因为LLaMA模型需要采用LLamaTokenizer加载，如果用AutoTokenizer加载llama可能会出现无限递归的问题，这和transformers版本有关
3. 如果运行环境不支持deepspeed，去掉--deepspeed 

deepspeed 的参数配置可参考：

1. https://www.deepspeed.ai/docs/config-json/
2. https://huggingface.co/docs/accelerate/usage_guides/deepspeed
3. https://github.com/huggingface/transformers/blob/main/tests/deepspeed

**关于deepspeed**

如果显存充足，可优先考虑stage 2，对应的配置文件是configs/deepspeed_config.json。如果显存不足，可采用stage 3，该模式采用模型参数并行，可显著减小显存占用，对应的配置文件是configs/deepspeed_config_stage3.json。（需要注意的是在stage=3 模式下，默认不会保存模型的权重，要指定stage3_gather_16bit_weights_on_model_save 为True）


训练日志和模型保存在output_dir目录下，目录下的文件结构应该如下：

```Arduino
output_dir/
├── checkpoint-244/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── trainer_state.json
├── checkpoint-527/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── trainer_state.json
├── pytorch_model.bin
├── print_log.txt
└── config.json
```

trainer_state.json记录了loss、learning_rate的变化

#### 2.2.2 LoRA

```bash
torchrun --nproc_per_node 8 train.py \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora True \
    --use_int8_training \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 8e-6 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --fp16 True \
    --seed 1234 \
    --gradient_checkpointing True \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir}
```

**参数说明**

- use_lora 代表采用LoRA训练
- use_int8_training 代表采用8bit量化训练，可显著减少显存占用
- lora_config 给出了LoRA的参数配置。如果训练Bloom模型，则改为configs/lora_config_bloom.json

output_dir目录的文件结构如下：

```Arduino
output_dir/
├── checkpoint-244/
│   ├── pytorch_model.bin
│   └── trainer_state.json
├── checkpoint-527/
│   ├── pytorch_model.bin
│   └── trainer_state.json
├── adapter_model.bin
├── print_log.txt
└── adapter_config.json
```

**注：LoRA训练后保存的模型adapter_model.bin有可能是空文件，此时需要将其它checkpoint-step下保存的pytorch_model.bin复制到output_dir目录下。如：**

```bash
cd output_dir
cp checkpoint-527/pytorch_model.bin adapter_model.bin
```

**确保adapter_model.bin是有效的LoRA权重**



#### 2.2.3 合并LoRA权重

如果您想要实现LoRA权重与预训练模型的合并，可运行如下命令：

```bash
bash scripts/merge_lora.sh
```

合并后的权重保存在output_path目录下，后续可通过from_pretrained直接加载



#### 2.2.4 多机多卡训练

以两台机器为例，每台机器上有8张卡

首先需要在第一台机器(主机器)上运行

```bash
bash scripts/multinode_run.sh 0
```

然后在第二台机器上运行

```bash
bash scripts/multinode_run.sh 1
```

**参数说明**

```bash
node_rank=$1
echo ${node_rank}
master_addr="10.111.112.223"

# #Multi-node
torchrun --nproc_per_node 8 --nnodes 2 --master_addr ${master_addr} --master_port 14545 --node_rank ${node_rank} src/train.py 
```

- node_rank 代表节点的rank，第一台机器（主机器）的rank设置为0，第二台机器的rank设置为1
- nnodes 代表节点机器的数量
- master_addr 代表主机器的ip地址
- master_port 代表与主机器通信的端口号



## 3. Inference

### 3.1 Inference

如果您看到了这里，说明您已经完成了训练。现在我们加载训练好的模型，验证模型生成文本的效果。

```bash
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --model_name_or_path model_name_or_path \
    --ckpt_path ckpt_path \
    --llama \
    --use_lora
```

**参数说明：**

- model_name_or_path 是原生预训练模型的路径
- ckpt_path 是训练后保存的模型路径，也就是output_dir
- llama 代表基础模型是否是LLaMA模型
- use_lora 代表ckpt_path是否是LoRA权重

**注：LoRA训练后保存的模型adapter_model.bin有可能是空文件，此时需要将其它checkpoint-step下保存的pytorch_model.bin复制到output_dir目录下**

此外，如果您已经将LoRA权重与预训练模型进行了合并，则ckpt_path指定为合并后权重保存的路径即可，不需要再指定use_lora

### 3.2 webUI


我们也提供了一个简洁的基于gradio的交互式web界面，启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 python src/interface.py \
    --model_name_or_path model_name_or_path \
    --ckpt_path ckpt_path \
    --llama \
    --use_lora
```

服务访问地址是 hostip:17860 

![webUI](https://github.com/LianjiaTech/BELLE/blob/main/train/docs/interface.png)

## 4. Additional Notes

### 4.1 LLaMA模型的使用

#### 4.1.1 facebook官方LLaMA权重转为hf格式

首先，您需要从[facebookresearch/llama](https://github.com/facebookresearch/llama)获取LLaMA模型的访问权限，下载官方检查点

```bash
python training_scripts/convert_llama_weights_to_hf.py --input_dir download_official_llama_path --model_size 7B --output_dir xx/llama-7b-hf
```

运行训练脚本时将model_name_or_path改为xx/llama-7b-hf即可

#### 4.1.2 BELLE-LLaMA转为hf格式

由于LLaMA模型的使用约束，我们只能开源与原始模型的diff（如：[BELLE-LLaMA-7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc)）。当您已经从[facebookresearch/llama](https://github.com/facebookresearch/llama)获取LLaMA模型的访问权限后，可参考 https://github.com/LianjiaTech/BELLE/tree/main/models ，转换后的模型即为我们指令调优后的LLaMA模型。

### 4.2 合并词表

如果您想在原版LLaMA的基础上扩充中文词表，可参考scripts/merge_tokenizers.py，后续会开放训练embedding的代码。扩充词表后的效果可参考我们的工作：[Towards Better Instruction Following Language Models for Chinese: Investigating the Impact of Training Data and Evaluation](https://arxiv.org/pdf/2304.07854.pdf)



## 5. 问题反馈

如有问题，请在GitHub Issue中提交。在遇到问题前，请先在 [FAQ](https://github.com/LianjiaTech/BELLE/blob/main/train/docs/FAQ.md) 中查找相似问题的解决方案。

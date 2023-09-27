# RLHF训练流程

## 一、PPO

### 1. 奖励模型

#### 1.1 准备数据

```jsonl
{"chosen": xxx, "rejected": xxx}
```

注意：
xxx文本已经添加了正确的提示，用于区别人类和bot，如 `Human: \n{text}\n\nAssistant: \n{text}`

#### 1.2 训练

```bash
bash scripts/run_rm.sh
```

- use_llama：是否使用llama作为基础模型
- use_lora：是否使用lora微调
- load_in_8bit：是否使用8bit载入
- load_in_4bit：是否使用4bit载入

注意：

- 支持deepspeed stage 3直接运行
- 不支持deepspeed stage 3 + lora，deepspeed stage 1/2 + lora可以运行
- load_in_8bit和load_in_4bit不能和deepspeed同时使用，可以和lora同时使用。需要将 `configs/accelerate_config_rm.yaml`中"distributed_type"从"DEEPSPEED"改为"MULTI_GPU"

### TODO

- [ ] deepspeed stage 3 + lora支持

### 2. PPO

#### 2.1 准备数据

```jsonl
{"text": xxx}
```

注意：xxx文本已经添加了正确的提示，用于区别人类和bot，如 `Human: \n{text}\n\nAssistant: \n`

#### 2.2 训练

```bash
bash scripts/run_ppo.sh
```

- use_llama：使用llama作为基础模型
- use_lora：使用lora微调
- batch_size：每个进程搜集的experience大小
- mini_batch_size：在experience上训练的batch大小
- input_length：输入文本的最长长度，超过该长度会被过滤掉
- ppo_epochs：在采样的batch训练多少轮
- data_epochs：在prompt数据上训练多少轮

注意：

- 支持deepspeed zero stage 3，拆分model、ref_model和reward_model
- 支持deepspeed zero stage 3 + lora
- $batch\\_size == mini\\_batch\\_size * gradient\\_accumulation\\_steps$
- 数据集大小要大于 `num_processes * batch_size`，否则部分进程拿不到数据，出现报错，输出中 `Train dataset length`可以看到经过长度过滤的数据集大小

#### TODO

- [ ] 每次训练的batch_size是 `num_processes * batch_size`，每个进程只会从自己的 `batch`中采样，而不是从全局的 `num_processes * batch_size`中采样，这会导致每个gpu采到的 `mini_batch`不是完全随机的，`mini_batch`不包含其它进程 `batch`中的样本
- [ ] gradient checkpointing
- [ ] resume from checkpoint

## 二、DPO

### 2.1 准备数据

格式：

```jsonl
{"chosen":xxx, "reject":xxx, "prompt":xxx}
```

一条数据样例：

```jsonl
{"chosen": "水的化学式是H2O。这意味着每个水分子由两个氢原子（H）和一个氧原子（O）组成。在这个结构中，氢原子和氧原子通过共价键相连。", "rejected": "H2O.", "prompt": "Human: \n水的化学式是什么？\n\nAssistant: \n"}
```

### 2.2 训练

首先，请将“train/scripts”下“run_dpo.sh”脚本中的“...”改成所需参数值

其次：

```bash
cd train/scripts
bash run_dpo.sh
```

注：DPO部分代码基于[trl项目](https://github.com/huggingface/trl)，具体原理见[论文](https://arxiv.org/abs/2305.18290)

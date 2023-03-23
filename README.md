## <img src="assets/belle_logo.png" style="vertical-align: middle; width: 35px;"> BELLE: Bloom-Enhanced Large Language model Engine 
本项目基于 [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) ，Stanford Alpaca 的目标是构建和开源一个基于LLaMA的模型。 Stanford Alpaca 的种子任务都是英语，收集的数据也都是英文，因此训练出来的模型未对中文优化。<br/>
<br/>

本项目目标是促进中文对话大模型开源社区的发展。本项目针对中文做了优化，模型调优仅使用由ChatGPT生产的数据（不包含任何其他数据）。项目包含以下内容:
- 175个中文种子任务
- 生成数据的代码
- 0.5M生成的数据
- 基于BLOOMZ-7B1-mt优化后的模型

**欢迎大家通过issue贡献更多的prompts！**


## What's Coming Next
* March 23, 2023: 应很多朋友的建议(https://github.com/LianjiaTech/BELLE/issues/18, https://github.com/LianjiaTech/BELLE/issues/10, https://github.com/LianjiaTech/BELLE/issues/9, https://github.com/LianjiaTech/BELLE/issues/9, https://github.com/LianjiaTech/BELLE/issues/3 )，我们正在研发量化功能（LoRA下次一定），将大大降低推理的硬件需求，预计本周发布

## What's New
* March 20, 2023: [发布了2M数据训练的7B模型](https://huggingface.co/BelleGroup/BELLE-7B-2M).
* March 18, 2023: [发布了1M数据训练的7B模型](https://huggingface.co/BelleGroup/BELLE-7B-1M). [发布了0.6M数据训练的7B模型](https://huggingface.co/BelleGroup/BELLE-7B-0.6M)
* March 17, 2023: [发布了0.2M数据训练的7B模型](https://huggingface.co/BelleGroup/BELLE-7B-0.2M). [发布了0.5M中文数据集](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN)


## 概述
 [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 中提到
> 使用了Self-Instruct论文中介绍的技术生成了52K条指令数据，同时进行了一些修改，在初步的人类评估中，发现Alpaca 7B模型在Self-Instruct指令评估上的表现类似于text-davinci-003模型。

使用Alpaca模型的在线演示服务，我们发现该模型在中文上的表现还不太好。推测是因为Stanford Alpaca 的种子任务都是英语，收集的数据也都是英文，因此训练出来的模型未对中文优化。为了提升在中文上的效果，本项目基于[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)优化了中文种子任务，对生成代码进行了一些修改，同时选用BLOOMZ-7B作为基础模型训练得到了一个可以更好支持中文指令的开源模型 - BELLE。

我们开源基于Alpaca的数据收集代码，基于这段代码生成了约100万条中文数据，结合Alpaca的5万条英文数据，在BLOOMZ-7B模型训练得到的checkpoint上传在[Hugging Face](https://huggingface.co/BelleGroup)。


## 局限性和使用限制
基于当前数据和基础模型训练得到的SFT模型，在效果上仍存在以下问题：

1. 在涉及事实性的指令上可能会产生违背事实的错误回答。

2. 对于具备危害性的指令无法很好的鉴别，由此会产生危害性言论。

3. 在一些涉及推理、代码等场景下模型的能力仍有待提高。

基于以上模型局限性，我们要求开发者仅将我们开源的代码、数据、模型及后续用此项目生成的衍生物用于研究目的，不得用于商业，以及其他会对社会带来危害的用途。


## 数据发布
1. `zh_seed_tasks.jsonl`：包含175个种子任务，样例如下<br/>

```
{"id": "seed_task_20", "name": "horror_movie_opening", "instruction": "你需要为一部恐怖电影写一个创意的开场场景。", "instances": [{"input": "", "output": "太阳已经落山，留下了一个黑暗的小镇。微风吹拂空荡的街道，让每一个冒险走出门外的人感到一阵寒意。唯一的声音是被风吹动的树叶发出的轻微沙沙声。突然，一声令人毛骨悚然的尖叫声划破了寂静，随后是玻璃破碎的声音。一所房子亮起了灯光，可以看到一个人影朝镇中心奔跑。当那个人影越来越靠近时，清楚地看到那是一个年轻女子，她浑身血迹斑斑。"}], "is_classification": false}` 
```

2. `prompt_cn.txt`: 生成所使用的提示语
3. [0.5M生成的数据](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN) ： 为了方便模型训练，huggingface开源数据将原始生成文件中的"instruction"、"input"字段合并成"input"字段，"output"字段修改为"target"字段。


## 数据生成
沿用Alpaca的方式：
```
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_API_KEY
python generate_instruction.py generate_instruction_following_data
```

默认使用`Completion` API，模型`text-davinci-003`。如果想使用`Chat` API并使用`gpt-3.5-turbo`模型，可通过参数控制：

```
python generate_instruction.py generate_instruction_following_data \
    --api=chat --model_name=gpt-3.5-turbo
```

输出文件在`Belle.train.json`，可以人工筛选后再使用。

## 模型调优
我们基于BLOOMZ-7B1-mt模型和Belle.train.json训练模型，具体参数如下：<br/>


| 参数 | 值 |
| ------ | ------ |
| Batch size | 64 |
| Learning rate | 3e-6 |
| Epochs | 3 |
|Weight_decay | 0.001 |
|Warmup_rate | 0.1 |
|LR_scheduler | linear |


我们采取了不同大小规模（20万、60万、100万和200万样本）的指令学习的数据集训练模型，我们得到不同的模型版本如下所示:
| Datasize| 200,000 | 600,000 | 1,000,000 | 2,000,000 |
| ----- | ----- | ----- | ----- | ----- | 
| Finetuned Model | [BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M) | [BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M) | [BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M) | [BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M) |

随后，我们会基于我们之前的工作[Exploring ChatGPT's Ability to Rank Content: A Preliminary Study on Consistency with Human Preferences](https://arxiv.org/abs/2303.07610)中的方法对比下这几个模型的效果。

## 模型使用例子
| Human | Assitant |
| ----- | ------ |
| 写一篇英文散文诗，主题是春雨，想象自己是春雨，和英国古代诗人莎士比亚交流 | I am the spring rain,<br/>Dancing on your face and body.<br/>My dew-drenched petals dance with you,<br/>Bringing life to every plant.<br/><br/>Shakespeare, I am here to meet you.<br/>Your works have filled my heart with joy.<br/>The beauty of nature in its fullness<br/>Makes me feel like a partaker.<br/><br/>As the leaves sway and rustle by,<br/>My dew-filled hand reaches out for them.<br/>The sweet scent of flowers fill the air,<br/>And bring me peace from all around.<br/><br/>Butterflies flit about as well,<br/>Glistening in the sun's rays.<br/>Their wings are made of dew,<br/>A beautiful sight to behold.<br/><br/>You see, Shakespeare, I too am one with nature.<br/>My dew-drenched hair is a symbol of hope.<br/>For when we can connect back to our roots,<br/>We will be able to make it through this season. |
| I am the spring rain,<br/>Dancing on your face and body.<br/>My dew-drenched petals dance with you,<br/>Bringing life to every plant.<br/><br/>Shakespeare, I am here to meet you.<br/>Your works have filled my heart with joy.<br/>The beauty of nature in its fullness<br/>Makes me feel like a partaker.<br/><br/>As the leaves sway and rustle by,<br/>My dew-filled hand reaches out for them.<br/>The sweet scent of flowers fill the air,<br/>And bring me peace from all around.<br/><br/>Butterflies flit about as well,<br/>Glistening in the sun's rays.<br/>Their wings are made of dew,<br/>A beautiful sight to behold.<br/><br/>You see, Shakespeare, I too am one with nature.<br/>My dew-drenched hair is a symbol of hope.<br/>For when we can connect back to our roots,<br/>We will be able to make it through this season. <br/>把上面的散文诗翻译成中文 | 我是一场春雨，<br/>在你脸上跳跃。<br/>我浸润着露珠的翅膀和你一起舞动，<br/>为每一株植物带来生命。<br/><br/>莎士比亚，我来与你相遇。<br/>你的作品填满了我的心房。<br/>大自然的美丽让我感到与之一起分享，<br/>让我觉得自己成为了其中的一部分。<br/><br/>当树叶摇曳时，<br/>我的手掌沾满了露水。<br/>花香弥漫在空气中，<br/>带给我宁静和安宁。<br/><br/>你看，莎士比亚，我也是与大自然融为一体的。<br/>我沾湿的头发象征着希望。<br/>当我们能够回到我们的根源，<br/>我们就能共同度过这个季节。|
| Classify the sentiment of the following sentence into Positive, Neutral, or Negative: <br/>Super excited about teaching Stanford’s first course on Large Language Models! Check the syllabus out here | Positive |
| 小明的爸爸有三个孩子，老大叫王一，老二叫王二，老三叫什么？ | 小明 |
| 使用python写一个二分查找的代码 | def binary_search(arr, target):<br/>&nbsp;&nbsp;low = 0<br/>&nbsp;&nbsp;high = len(arr) - 1<br/>&nbsp;&nbsp;while (low <= high):<br/>&nbsp;&nbsp;&nbsp;&nbsp;mid = low + high // 2<br/>&nbsp;&nbsp;&nbsp;&nbsp;if arr[mid] == target:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return mid<br/>&nbsp;&nbsp;&nbsp;&nbsp;elif target < arr[mid]:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;low = mid + 1<br/>&nbsp;&nbsp;&nbsp;&nbsp;else:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;high = mid - 1<br/>&nbsp;&nbsp;return -1 |
| 今天天气怎么样，把这句话翻译成英语 | What's the weather like today? |
| 怎么让自己精力充沛，列5点建议 | 以下是一些可以让你保持精力充沛的建议： <br/>1. 睡眠充足。确保每天晚上得到充足的睡眠时间。 <br/>2. 多运动。进行适度的有氧运动，如慢跑、游泳等，可以帮助你提高能量和耐力。<br/>3. 吃健康食品。选择富含蛋白质的食物，如鸡肉、鱼肉和鸡蛋，以及新鲜的水果和蔬菜。 <br/>4. 减少咖啡因的摄入量。尽量避免在白天喝含咖啡因的饮料或吃含有咖啡因的食物。 <br/>5. 放松身心。尝试冥想、瑜伽或其他形式的放松活动来减轻压力和焦虑感。 |
| 明天就假期结束了，有点抗拒上班，应该什么办？ | 面对即将结束的假期和开始工作时的紧张情绪，您可以采取以下措施来帮助您更好地适应： <br/>1. 为自己制定一个清晰的工作计划。这可以帮助您更好地组织时间和任务，并避免在工作中感到混乱。 <br/>2. 与同事或朋友交流，分享您的感受和工作压力。这将有助于缓解一些焦虑感，同时也可以获得支持和建议。 <br/>3. 在工作日前花些时间放松身心，例如进行瑜伽、冥想等活动。这些方法可以减轻身体和精神上的疲劳感，使您更加精力充沛地迎接挑战。 |
| 父母都姓吴，取一些男宝宝和女宝宝的名字 | - 男宝：吴俊杰、吴雨辰、吴子墨、吴思源、吴泽阳、吴宇航、吴博文、吴哲毅。 <br/>- 女宝：吴美琪、吴佳怡、吴欣瑜、吴香凝、吴雪柔、吴颖诗、吴语嫣。 |
| 推荐几本金庸的武侠小说 | 以下是三本我最喜欢的金庸武侠小说的推荐： <br/>1. 《射雕英雄传》 - 一部充满智慧和机智的小说，讲述了郭靖、黄蓉夫妇的故事。 <br/>2. 《神雕侠侣》- 这是一部浪漫而感人的故事，讲述了一对情侣小龙女与杨过之间的爱情故事。 <br/>3. 《倚天屠龙记》- 这是一个宏伟而壮观的故事，描绘了明教领袖张无忌如何成为一位出色的英雄。 |


## 引用

如果使用本项目的代码、数据或模型，请引用本项目。

```
@misc{BELLE,
  author = {Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Baochang Ma, Xiangang Li},
  title = {BELLE: Bloom-Enhanced Large Language model Engine },
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LianjiaTech/BELLE}},
}
```

当然，你也需要引用原始的BLOOM论文、Stanford Alpaca和Self-Instruct论文。
<br/>
<br/>

***
***

<br/>
<br/>

## <img src="assets/belle_logo.png" style="vertical-align: middle; width: 35px;"> BELLE: Bloom-Enhanced Large Language model Engine
This project is from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) which aims to build and share instruction-following LLaMA model. <br/>
The seed tasks in Stanford Alpaca are English only, and the model performs relatively poorly in Chinese. <br/>
<br/>
The goal of this project is to promote the development of the open-source community for Chinese language large-scale conversational models. This project optimizes Chinese performance in addition to original Alpaca. The model finetuning uses only data generated via ChatGPT (without other data). This repo contains:
- The 175 chinese seed tasks used for generating the data
- The code for generating the data
- The 0.5M generated data used for fine-tuning the model
- The model finetuned from BLOOMZ-7B1-mt on data generated by this project

**More prompts are welcomed via issues!**


## What's Coming Next
* March 23, 2023: As many friends' requested (https://github.com/LianjiaTech/BELLE/issues/18, https://github.com/LianjiaTech/BELLE/issues/10, https://github.com/LianjiaTech/BELLE/issues/9, https://github.com/LianjiaTech/BELLE/issues/9, https://github.com/LianjiaTech/BELLE/issues/3 )，we are working on quantization (LoRA may be next time), will decrease hardware requirement. Coming this week (maybe)!


## What's New
* March 20, 2023: [Released 7B model trained on 2M data](https://huggingface.co/BelleGroup/BELLE-7B-2M).
* March 18, 2023: [Released 7B model trained on 1M data](https://huggingface.co/BelleGroup/BELLE-7B-1M). [Released 7B model trained on 0.6M data](https://huggingface.co/BelleGroup/BELLE-7B-0.6M)
* March 17, 2023: [Initial release of 7B model trained on 0.2M data](https://huggingface.co/BelleGroup/BELLE-7B-0.2M). [Released 0.5M dataset](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN)


## Overview
 [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) mentioned
> The current Alpaca model is fine-tuned from a 7B LLaMA model on 52K instruction-following data generated by the techniques in the Self-Instruct paper, with some modifications... . In a preliminary human evaluation, we found that the Alpaca 7B model behaves similarly to the text-davinci-003 model on the Self-Instruct instruction-following evaluation suite.

From the web demo of Alpaca, we found it's performance on Chinese is not as well. We speculate the reason to be that the seed tasks of Stanford Alpaca are all English, and the generated data are also in English, so model tuned on it is not optimized for Chinese. This project aims to boost Chinese performance with improved Chinese seed tasks based on [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), some modification to to instruction generation code, and also BLOOMZ-7B as the base model. The result is a model which better supports Chinese - **BELLE**.

The instruction generation code and finetuned model checkpoint [Hugging Face](https://huggingface.co/BelleGroup/BELLE-7B-0.2M) trained on the generated dataset (approx. 1m instruction and answer pairs, plus original ~50k Alpaca pairs) based on BLOOMZ-7B are both open sourced.

## Limitation and Usage Limits
There still exists a few issues in the model trained on current base model and data:

1. The model might generate factual errors when asked to follow instructions related to facts.

2. Occasionally generates harmful responses since the model still struggles to identify potential harmful instructions.

3. Needs improvements on reasoning and coding.

Since the model still has its limitations, we require developers only use the open-sourced code, data, model and any other artifacts generated via this project for research purposes. Commercial use and other potential harmful use cases are not allowed.



## Data Release
1. `zh_seed_tasks.jsonl` contains 175 seed tasks, for example:<br/>
`{"id": "seed_task_20", "name": "horror_movie_opening", "instruction": "你需要为一部恐怖电影写一个创意的开场场景。", "instances": [{"input": "", "output": "太阳已经落山，留下了一个黑暗的小镇。微风吹拂空荡的街道，让每一个冒险走出门外的人感到一阵寒意。唯一的声音是被风吹动的树叶发出的轻微沙沙声。突然，一声令人毛骨悚然的尖叫声划破了寂静，随后是玻璃破碎的声音。一所房子亮起了灯光，可以看到一个人影朝镇中心奔跑。当>那个人影越来越靠近时，清楚地看到那是一个年轻女子，她浑身血迹斑斑。"}], "is_classification": false}` 
2. `prompt_cn.txt` Chinese prompt for generating instructions
3. [0.5M generated data](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN)：To facilitate model training, Hugging Face open-sourced data that merged the "instruction" and "input" fields in the original generation file into a single "input" field, and renamed the "output" field as the "target" field.


## Data Generation Process
Following Alpaca:
```
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_API_KEY
python generate_instruction.py generate_instruction_following_data
```

Uses the `Completion` API and `text-davinci-003` model by default. To use `Chat` API and `gpt-3.5-turbo` model, just change the arguments:

```
python generate_instruction.py generate_instruction_following_data \
    --api=chat --model_name=gpt-3.5-turbo
```

Generated instructions are in `Belle.train.json`, you can check manually before using it.


## Fine-tuning
Finetuning is done based on `BLOOMZ-7B1-mt` and `Belle.train.json` using the following hyperparameters:<br/>

| Hyperparameter | Value |
| ------ | ------ |
| Batch size | 64 |
| Learning rate | 3e-6 |
| Epochs | 3 |
|Weight_decay | 0.001 |
|Warmup_rate | 0.1 |
|LR_scheduler | linear |


We trained models using datasets of different sizes (200,000, 600,000, 1,000,000 and 2,000,000 samples) for instruction learning, and we obtained different model versions as shown below:
| Datasize| 200,000 | 600,000 | 1,000,000 | 2,000,000 |
| ----- | ----- | ----- | ----- | ----- | 
| Finetuned Model | [BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M) | [BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M) | [BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M) | [BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M) |



## Citation

Please cite us when using our code, data or model.

```
@misc{BELLE,
  author = {Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Baochang Ma, Xiangang Li},
  title = {BELLE: Bloom-Enhanced Large Language model Engine },
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LianjiaTech/BELLE}},
}
```

Cite the original BLOOM, Stanford Alpaca and Self-Instruct papers as well!

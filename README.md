## <img src="assets/belle_logo.png" style="vertical-align: middle; width: 35px;"> BELLE: BE Large Language model Engine 
本项目目标是促进中文对话大模型开源社区的发展。现阶段本项目基于BLOOM和LLAMA针对中文做了优化，模型调优仅使用由ChatGPT生产的数据（不包含任何其他数据）。
<br/>

项目包含以下内容:
* 数据开放：参考[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 生成的中文数据集[1M](https://huggingface.co/datasets/BelleGroup/generated_train_1M_CN) + [0.5M](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN)
* 基于BLOOMZ-7B1-mt优化后的模型：[BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M)，[BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M)，[BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M)，[BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M)
* 基于LLAMA优化后的模型：[BELLE-LLAMA-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-0.6M)，[BELLE-LLAMA-7B-2M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M)（强烈建议大家重新下载最新的BELLE-LLAMA模型，调优了模型学习过程，性能有所提升）

**欢迎大家通过issue贡献更多的prompts！**
<br/>

## 局限性和使用限制
基于当前数据和基础模型训练得到的SFT模型，在效果上仍存在以下问题：

1. 在涉及事实性的指令上可能会产生违背事实的错误回答。

2. 对于具备危害性的指令无法很好的鉴别，由此会产生危害性言论。

3. 在一些涉及推理、代码等场景下模型的能力仍有待提高。

基于以上模型局限性，我们要求开发者仅将我们开源的代码、数据、模型及后续用此项目生成的衍生物用于研究目的，不得用于商业，以及其他会对社会带来危害的用途。

<br/>

## 模型发布

我们采取了不同大小规模（20万、60万、100万和200万样本）的指令学习的数据集训练模型，基于BLOOMZ-7B1-mt训练调优后的模型，现已开放:
| Datasize| 200,000 | 600,000 | 1,000,000 | 2,000,000 |
| ----- | ----- | ----- | ----- | ----- | 
| Finetuned Model | [BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M) | [BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M) | [BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M) | [BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M) |

我们也采用对应数据集基于LLAMA-7B训练调优了模型，现已开放:
| Datasize| 600,000 | 2,000,000 | 2,000,000 |
| ----- | ----- | ----- |  ----- |
| Finetuned Model | [BELLE-LLAMA-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-0.6M) | [BELLE-LLAMA-7B-2M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M) | BELLE-LLAMA-13B-2M (to be released) |


此外，方便大家使用，也对模型进行了量化[基于GPTQ量化后的模型](https://huggingface.co/BelleGroup/)，其中包含针对bloom和llama基础的模型上的4bit和8bit的量化模型。
| model name |  file size | GPU memory usage |
| ----- | ----- | ----- |
| bloom7b-2m  | 27G   | ~28.2G |
| bloom7b-2m-8bit-128g.pt | 9.7G | ~11.4G |
| bloom7b-2m-4bit-128g.pt | 6.9G | ~8.4G |
| bloom7b-0.2m-8bit-128g.pt | 9.7G | ~11.4G |
| bloom7b-0.2m-4bit-128g.pt | 6.9G | ~8.4G |
| llama7b-2m | 26G | ~15G |
| llama7b-2m-8bit-128g.pt | 6.8G | ~8.9G |
| llama7b-2m-4bit-128g.pt | 3.8G | ~5.6G |


### 模型效果比较
以Bloomz-7b1-mt为基础，我们评估了不同数量的instruction tuning数据，对模型效果的影响。总的来说，提升数据量能持续带来效果的提升，但是在不同类型的任务上表现有所不同。在Extract, Classification, Closed QA, 和Summarization任务上，增加数据能持续带来效果的提升，还未达到瓶颈。在Translation, Rewrite, 和Brainstorming任务上，几十万的数据量就能获得较好的效果。在Math, Code, 和COT任务上，模型效果较差，而且增加数据量已经无法带来效果的提升。
![Image text](assets/model_compare.jpg)
<br/>
详见论文：[Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases](https://arxiv.org/abs/2303.14742)。
<br/>

## 数据发布
1. [zh_seed_tasks.jsonl](https://github.com/LianjiaTech/BELLE/blob/main/zh_seed_tasks.json)：包含175个种子任务。
2. [0.5M生成的数据](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN) ： 为了方便模型训练，huggingface开源数据将原始生成文件中的"instruction"、"input"字段合并成"input"字段，"output"字段修改为"target"字段。
3. [1M生成的数据](https://huggingface.co/datasets/BelleGroup/generated_train_1M_CN)：生成方式与0.5M数据集相同，在后处理中去掉了一些质量不高的数据，例如自称`GPT模型`的数据、由于input不完善导致模型无法回答的数据，以及指令是中文但input或target是英文的数据。
<br/>


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

<br/>

## 引用

如果使用本项目的代码、数据或模型，请引用本项目。

```
@misc{BELLE,
  author = {Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Baochang Ma and Xiangang Li},
  title = {BELLE: BE Large Language model Engine },
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LianjiaTech/BELLE}},
}
@article{belle2023exploring,
  title={Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases},
  author={Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Lei Zhang, Baochang Ma, Xiangang Li},
  journal={arXiv preprint arXiv:2303.14742},
  year={2023}
}
```

当然，你也需要引用原始的BLOOM论文、LLaMA论文、Stanford Alpaca和Self-Instruct论文。
<br/>


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

<br/>

***
***

<br/>

## <img src="assets/belle_logo.png" style="vertical-align: middle; width: 35px;"> BELLE: BE Large Language model Engine
The goal of this project is to promote the development of the open-source community for Chinese language large-scale conversational models. This project optimizes Chinese performance in addition to original Alpaca. The model finetuning uses only data generated via ChatGPT (without other data). 
<br/>

This repo contains:
* Data Release: The Chinese dataset generated [1M](https://huggingface.co/datasets/BelleGroup/generated_train_1M_CN) + [0.5M](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN), using [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) as reference
* The model optimized based on BLOOMZ-7B1-mt: [BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M)，[BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M)，[BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M)，[BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M)
* The model optimized based on LLAMA: [BELLE-LLAMA-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-0.6M)，[BELLE-LLAMA-7B-2M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M)(Highly recommend the the lastest version of these models, which have be improved through optimizing the learning process)

**More prompts are welcomed via issues!**
<br/>

## Limitation and Usage Limits
There still exists a few issues in the model trained on current base model and data:

1. The model might generate factual errors when asked to follow instructions related to facts.

2. Occasionally generates harmful responses since the model still struggles to identify potential harmful instructions.

3. Needs improvements on reasoning and coding.

Since the model still has its limitations, we require developers only use the open-sourced code, data, model and any other artifacts generated via this project for research purposes. Commercial use and other potential harmful use cases are not allowed.

<br/>

## Fine-tuning and Models Release

We trained models on instruction learning datasets of different sizes (200,000, 600,000, 1 million, and 2 million samples) and based on the BLOOMZ-7B1-mt trained and optimized model. They are now release for use, you can download the checkpoints in [haggingface BELLE group](https://huggingface.co/BelleGroup):
| Datasize| 200,000 | 600,000 | 1,000,000 | 2,000,000 |
| ----- | ----- | ----- | ----- | ----- | 
| Finetuned Model | [BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M) | [BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M) | [BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M) | [BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M) |

We have also trained and optimized models based on LLAMA-7B using corresponding datasets, which are now open for use:
| Datasize| 600,000 | 2,000,000 | 2,000,000 |
| ----- | ----- | ----- | ----- | 
| Finetuned Model | [BELLE-LLAMA-7B-0.6M](https://huggingface.co/BelleGroup/LLAMA-7B-0.6M) | [BELLE-LLAMA-7B-2M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M) | BELLE-LLAMA-13B-2M (to be released) |

In addition, for the convenience of users, we have also quantized the [model](https://huggingface.co/BelleGroup/) based on GPTQ, which includes 4-bit and 8-bit quantized models for bloom and llama based models.
| model name |  file size | GPU memory usage |
| ----- | ----- | ----- |
| bloom7b-2m  | 27G   | ~28.2G |
| bloom7b-2m-8bit-128g.pt | 9.7G | ~11.4G |
| bloom7b-2m-4bit-128g.pt | 6.9G | ~8.4G |
| bloom7b-0.2m-8bit-128g.pt | 9.7G | ~11.4G |
| bloom7b-0.2m-4bit-128g.pt | 6.9G | ~8.4G |
| llama7b-2m | 26G | ~15G |
| llama7b-2m-8bit-128g.pt | 6.8G | ~8.9G |
| llama7b-2m-4bit-128g.pt | 3.8G | ~5.6G |

### Model performance comparison 
Based on the Bloomz-7b1-mt model, we evaluated the impact of different amounts of instruction data on our released models' performance. 
Overall, increasing the amount of data consistently improved performance, but the extent of improvement varied across different types of tasks. 
For Extract, Classification, Closed QA, and Summarization tasks, increasing data continued to improve performance without reaching a plateau. 
For Translation, Rewrite, and Brainstorming tasks, good performance could be achieved with only hundreds of thousands of data. 
However, for Math, Code, and COT tasks, these models' performance were poor, and increasing data did not lead to further improvement.
![Image text](assets/model_compare.jpg)
<br/>
More details are in paper [Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases](https://arxiv.org/abs/2303.14742)。
<br/>

## Data Release
1. [zh_seed_tasks.jsonl](https://github.com/LianjiaTech/BELLE/blob/main/zh_seed_tasks.json) contains 175 seed tasks
2. [0.5M generated data](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN)：To facilitate model training, Hugging Face open-sourced data that merged the "instruction" and "input" fields in the original generation file into a single "input" field, and renamed the "output" field as the "target" field.
3. [1M generated data](https://huggingface.co/datasets/BelleGroup/generated_train_1M_CN). Same generation pipeline as 0.5M dataset, removed lower-quality items in postprocessing, e.g. items regarding `GPT model`, bad items because of incomplete/invalid input, items with Chinese instructionb but English input or target.

<br/>

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

<br/>


## Citation

Please cite us when using our code, data or model.

```
@misc{BELLE,
  author = {Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Baochang Ma and Xiangang Li},
  title = {BELLE: BE Large Language model Engine},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LianjiaTech/BELLE}},
}

@article{belle2023exploring,
  title={Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases},
  author={Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Lei Zhang, Baochang Ma, Xiangang Li},
  journal={arXiv preprint arXiv:2303.14742},
  year={2023}
}
```

Cite the original BLOOM, LLaMA, Stanford Alpaca and Self-Instruct papers as well!

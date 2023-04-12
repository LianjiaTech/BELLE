
## <img src="assets/belle_logo.png" style="vertical-align: middle; width: 35px;"> BELLE: Be Everyone's Large Language model Engine

*[中文README](README.md).*

<div align="center">

<a href="https://github.com/LianjiaTech/BELLE/stargazers">![GitHub Repo stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=social)</a>
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/discord-BELLE%20Group-green.svg?logo=discord)](https://discord.gg/pMPY53UUGq)
[![Generic badge](https://img.shields.io/badge/wechat-BELLE-green.svg?logo=wechat)](https://github.com/LianjiaTech/BELLE/blob/main/assets/belle_wechat.jpg)
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/BelleGroup)
<a href="https://github.com/LianjiaTech/BELLE/tree/main/docs/">![Docs](https://img.shields.io/badge/papers-BELLE%2Fdocs-green)</a>
<a href="https://github.com/LianjiaTech/BELLE/tree/main/gptq/">![Docs](https://img.shields.io/badge/quantization_recipe-BELLE%2Fgptq-green)</a>
<a href="https://github.com/LianjiaTech/BELLE/tree/main/train/">![Docs](https://img.shields.io/badge/train_recipe-BELLE%2Ftrain-green)</a>
<a href="https://github.com/LianjiaTech/BELLE/tree/main/eval/">![Docs](https://img.shields.io/badge/eval_set-BELLE%2Feval-green)</a>
<a href="https://github.com/LianjiaTech/BELLE/tree/main/chat/">![Docs](https://img.shields.io/badge/ChatBELLE-BELLE%2Fchat-green)</a>

</div>

The goal of this project is to promote the development of the open-source community for Chinese language large-scale conversational models, and our vision is to help building large language model engine for everyone. This project optimizes Chinese performance based on opensource pretrained large language models. These models finetuning uses only data generated via ChatGPT (without other data). 
<br/>

## ChatBELLE App

Try our cross-platform chat app to run 4-bit quantized BELLE-7B model natively on your device.
The following screencap ran on an M1 Max CPU real-time (no speed adjustment).

**App Downloading**：Releases

[App Companion Model and Usage](chat/README.md)

<img src="./chat/chatbelle-demo.gif"></img>

## 🔄 What‘s new

* [2023/04/12] Released [ChatBELLE App](chat/README.md), a cross-platform BELLE-7B model realtime chat App based on [llama.cpp](https://github.com/ggerganov/llama.cpp) and [Flutter](https://flutter.dev/).
* [2023/04/08] In [BELLE/10M](https://github.com/LianjiaTech/BELLE/tree/main/10M), a new dataset named ["Generated Chat"]((https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)) containing newly generated multi-turn dialogues with given roles, and a new dataset named ["train_2M_CN"](https://huggingface.co/datasets/BelleGroup/train_2M_CN) containing 2 million newly added diverse instruction task data.
* [2023/04/05] The inference code that can be run on [Colab](https://colab.research.google.com/github/LianjiaTech/BELLE/blob/main/notebook/BELLE_INFER_COLAB.ipynb) is provided

## 📝 This repo contains

###  🚀 Traning recipe

  Details in [BELLE/train](train/), A simplified implementation of training code with support for finetune, LORA, and DeepSpeed as much as possible.

### 📊 Data Release
  
  Details in [BELLE/1.5M](1.5M/)，The Chinese dataset generated [1M](https://huggingface.co/datasets/BelleGroup/generated_train_1M_CN) + [0.5M](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN), using [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) as reference
  
  10M more data will be released gradually，details in [BELLE/10M](10M/). Currently, we have 0.8M multiturn data, and 0.25 math data.

### 🧐 Evaluation set & evaluation method
  
  Details in [BELLE/eval](eval/). A test set with over 1k samples and corresponding scoring prompts. It includes multiple categories and is evaluated using either GPT-4 or ChatGPT.

### 🤖 Models

  Details in [BELLE/models](models/)
  
* The model optimized based on BLOOMZ-7B1-mt：[BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M)，[BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M)，[BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M)，[BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M)
  
* The finetuned models based on [Meta LLaMA](https://github.com/facebookresearch/llama): [BELLE-LLaMA-7B-0.6M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-0.6M-enc)
, [BELLE-LLaMA-7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc)
, [BELLE-LLaMA-7B-2M-gptq-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-gptq-enc)
, [BELLE-LLaMA-13B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-13B-2M-enc). Considering [LLaMA's License](https://github.com/facebookresearch/llama/blob/main/LICENSE) constraints, the model is for research and learning only. Please strictly respect LLaMA's usage policy. Users are suggested to finetune the model with open-source scripts and datasets. We are not allowed to publish weights for LLaMA, of course, even finetuned, but there is no problem publishing the difference, a patch that we suggest to apply to the files. The encryption is a simple XOR between files, ensuring that only the people that have access to the original weights (from completely legal sources, of course) can transform them into finetuned weights. You can find the decrypt code on [BELLE/models](models/).

### ⚖️ Quantized_models

  Details in [BELLE/gptq](gptq/)，Referring to the implementation of GPT-Q, the relevant models in this project have been quantized.

### 🌐 Colab
  
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LianjiaTech/BELLE/blob/main/notebook/BELLE_INFER_COLAB.ipynb) provides the colab in [BELLE/notebook](https://colab.research.google.com/github/LianjiaTech/BELLE/blob/main/notebook/BELLE_INFER_COLAB.ipynb)

### 💬 ChatBELLE App

  Details in [BELLE/chat](chat/README.md), cross-platform LLM chat app with [BELLE](https://github.com/LianjiaTech/BELLE) using quantized on-device offline models and Flutter UI, running on macOS (done), Windows, Android, iOS and more.

**More prompts are welcomed via issues!**

<br/>

## ⚠️ Limitation and Usage Limits

There still exists a few issues in the model trained on current base model and data:

1. The model might generate factual errors when asked to follow instructions related to facts.

2. Occasionally generates harmful responses since the model still struggles to identify potential harmful instructions.

3. Needs improvements on reasoning and coding.

Since the model still has its limitations, we require developers only use the open-sourced code, data, model and any other artifacts generated via this project for research purposes. Commercial use and other potential harmful use cases are not allowed.

<br/>

## 📌 Citation

Please cite us when using our code, data or model.

```
@misc{BELLE,
  author = {Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Baochang Ma and Xiangang Li},
  title = {BELLE: Be Everyone's Large Language model Engine},
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

</br>

## 📚 Use case
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


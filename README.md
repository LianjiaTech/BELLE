## <img src="assets/belle_logo.png" style="vertical-align: middle; width: 35px;"> BELLE: Be Everyone's Large Language model Engine

*Read this in [English](README_en.md).*

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

本项目目标是促进中文对话大模型开源社区的发展，愿景做能帮到每一个人的LLM Engine。现阶段本项目基于一些开源预训练大语言模型（如BLOOM），针对中文做了优化，模型调优仅使用由ChatGPT生产的数据（不包含任何其他数据）。

下图是一个可以使用App在设备端本地运行4bit量化的BELLE-7B模型，在M1 Max CPU上实时运行的效果（未加速）。App下载详见[App配套模型下载及使用说明](chat/README.md)，App[下载链接](https://github.com/LianjiaTech/BELLE/releases/download/v0.95/chatbelle.dmg)，目前仅提供了mac os版本。模型需要单独下载。**模型经过量化后，效果损失明显，我们将持续研究如何提升。**

<img src="./chat/chatbelle-demo.gif"></img>

</br>

## 🔄 最近更新

* [2023/04/17] 更新了两篇论文最新工作，对比了不同方式产生的训练数据、不同训练方法（LoRA, finetune)对效果的影响
* [2023/04/12] 发布了[ChatBELLE App](chat/README.md)，基于[llama.cpp](https://github.com/ggerganov/llama.cpp)和[Flutter](https://flutter.dev/)，实现跨平台的BELLE-7B离线模型实时交互。
* [2023/04/11] 更新了一个人工精校的eval集合，大约一千多条
* [2023/04/08] [BELLE/10M](https://github.com/LianjiaTech/BELLE/tree/main/10M)中，新加40万条生成的给定角色的多轮对话[Generated Chat](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)，新加200万条生成多样化指令任务数据[train_2M_CN](https://huggingface.co/datasets/BelleGroup/train_2M_CN)。
* [2023/04/05] 提供了colab上面可运行的推理代码(默认加载4Bit量化的BELLE模型，模型效果会有所损失)[Colab](https://colab.research.google.com/github/LianjiaTech/BELLE/blob/main/notebook/BELLE_INFER_COLAB.ipynb)

</br>

## 📝 项目主要内容

### 🚀 训练代码

详见[BELLE/train](https://github.com/LianjiaTech/BELLE/tree/main/train)，尽可能简化的一个训练代码实现，支持finetune，lora，deepspeed

### 📊 数据开放
  
* 详见[BELLE/1.5M](https://github.com/LianjiaTech/BELLE/tree/main/1.5M)，参考[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) 生成的中文数据集[1M](https://huggingface.co/datasets/BelleGroup/train_1M_CN) + [0.5M](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)；
  
* 持续开放的数据集，详见[BELLE/10M](https://github.com/LianjiaTech/BELLE/tree/main/10M)

### 🧐 验证集合&验证方法

详见[BELLE/eval](https://github.com/LianjiaTech/BELLE/tree/main/eval)，一个1k+的测试集合，和对应打分prompt。包含多个类别，采用GPT-4或者ChatGPT打分。同时提供了一个打分的网页，方便针对单个case使用。欢迎大家通过PR提供更多的测试用例。

### 🤖 模型

详见[BELLE/models](models/)

* 基于BLOOMZ-7B1-mt优化后的模型：[BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M)，[BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M)，[BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M)，[BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M)

* 基于[Meta LLaMA](https://github.com/facebookresearch/llama)实现调优的模型：[BELLE-LLaMA-7B-0.6M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-0.6M-enc)
, [BELLE-LLaMA-7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc)
, [BELLE-LLaMA-7B-2M-gptq-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-gptq-enc)
, [BELLE-LLaMA-13B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-13B-2M-enc)。请参考[Meta LLaMA的License](https://github.com/facebookresearch/llama/blob/main/LICENSE)，目前仅供学习交流。请严遵守LLaMA的使用限制。LaMA模型不允许发布调优后的完整模型权重，但是可以发布原始的模型的diff。因此，我们使用文件间的XOR，保证拥有LLaMA原始模型授权的人才可以将本项目发布的模型转化成可以使用的格式。格式转化代码参考[BELLE/models](https://github.com/LianjiaTech/BELLE/tree/main/models)

### ⚖️ 模型量化gptq

详见[BELLE/gptq](https://github.com/LianjiaTech/BELLE/tree/main/gptq)，参考gptq的实现，对本项目中相关模型进行了量化

### 🌐 Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LianjiaTech/BELLE/blob/main/notebook/BELLE_INFER_COLAB.ipynb) 提供了colab上面可运行的推理代码[Colab](https://colab.research.google.com/github/LianjiaTech/BELLE/blob/main/notebook/BELLE_INFER_COLAB.ipynb)

### 💬 ChatBELLE App

详见[BELLE/chat](chat/README.md)，基于[BELLE](https://github.com/LianjiaTech/BELLE)模型的跨平台离线大语言模型交谈App。使用量化后的离线端上模型配合Flutter，可在macOS（已支持）、Windows、Android、iOS等设备上运行。

**欢迎大家通过issue贡献更多的prompts！**

<br/>

## 最新进展

### 

[Towards Better Instruction Following Language Models for Chinese: Investigating the Impact of Training Data and Evaluation](https://github.com/LianjiaTech/BELLE/blob/main/docs/Towards%20Better%20Instruction%20Following%20Language%20Models%20for%20Chinese.pdf)

为了推动开源大语言模型的发展，大家投入了大量精力开发能够类似于ChatGPT的低成本模型。
首先，为了提高模型在中文领域的性能和训练/推理效率，我们进一步扩展了LLaMA的词汇表，并在34亿个中文词汇上进行了二次预训练。

此外，目前可以看到基于ChatGPT产生的指令训练数据方式有：1）参考Alpaca基于GPT3.5得到的self-instruct数据；
2）参考Alpaca基于GPT4得到的self-instruct数据；3）用户使用ChatGPT分享的数据ShareGPT。
在这里，我们着眼于探究训练数据类别对模型性能的影响。具体而言，我们考察了训练数据的数量、质量和语言分布等因素，以及我们自己采集的中文多轮对话数据，以及一些公开可访问的高质量指导数据集。

为了更好的评估效果，我们使用了一个包含一千个样本和九个真实场景的评估集来测试各种模型，同时通过量化分析来提供有价值的见解，以便更好地促进开源聊天模型的发展。

这项研究的目标是填补开源聊天模型综合评估的空白，以便为这一领域的持续进步提供有力支持。

实验结果如下：

<table>
  <tr>
    <td> Factor </td>
    <td> Base model </td>
    <td> Training data </td>
    <td> Score_w/o_others </td>
  <tr>
    <td rowspan="2">词表扩充</td>
    <td>LLaMA-EXT</td>
    <td>zh(alpaca-3.5&4) + sharegpt</td>
    <td>0.670</td>
  </tr>
  <tr>
    <td>LLaMA</td>
    <td>zh(alpaca-3.5&4) + sharegpt</td>
    <td>0.652</td>
  </tr>
  <tr>
    <td rowspan="2">数据质量</td>
    <td>LLaMA-EXT</td>
    <td>zh(alpaca-3.5)</td>
    <td>0.642</td>
  </tr>
  <tr>
    <td>LLaMA-EXT</td>
    <td>zh(alpaca-4)</td>
    <td>0.693</td>
  </tr>
  <tr>
    <td rowspan="3">数据语言分布</td>
    <td>LLaMA-EXT</td>
    <td>en(alpaca-3.5&4)</td>
    <td>0.659</td>
  </tr>
  <tr>
    <td>LLaMA-EXT</td>
    <td>zh(alpaca-3.5&4) + sharegpt</td>
    <td>0.670</td>
  </tr>
  <tr>
    <td>LLaMA-EXT</td>
    <td>zh(alpaca-3.5&4) + sharegpt</td>
    <td>0.668</td>
  </tr>
  <tr>
    <td rowspan="2">数据规模</td>
    <td>LLaMA-EXT</td>
    <td>zh(alpaca-3.5&4) + sharegpt</td>
    <td>0.670</td>
  </tr>
  <tr>
    <td>LLaMA-EXT</td>
    <td>zh(alpaca-3.5&4) + sharegpt <br>+ BELLE-0.5M-CLEAN</td>
    <td>0.762</td>
  </tr>
</table>

其中**BELLE-0.5M-CLEAN**是从230万指令数据中清洗得到。

**需要强调指出的是**：通过案例分析，我们发现我们的评估集在全面性方面存在局限性，这导致了模型分数的改善与实际用户体验之间的不一致。构建一个高质量的评估集是一个巨大的挑战，因为它需要在保持平衡难易程度的同时，包含尽可能多样的使用场景。如果评估样本主要都过于困难，那么所有模型的表现将会很差，使得辨别各种训练策略的效果变得具有挑战性。相反，如果评估样本都相对容易，评估将失去其比较价值。此外，必须确保评估数据与训练数据保持独立。

基于这些观察，我们谨慎地提醒不要假设模型仅通过在有限数量的测试样本上获得良好结果就已经达到了与ChatGPT相当的性能水平。我们认为，优先发展全面评估集的持续发展具有重要意义。

这篇工作中的相关数据和模型将会在4月19日前在本项目中开源。
## ⚠️ 局限性和使用限制

基于当前数据和基础模型训练得到的SFT模型，在效果上仍存在以下问题：

1. 在涉及事实性的指令上可能会产生违背事实的错误回答。

2. 对于具备危害性的指令无法很好的鉴别，由此会产生危害性言论。

3. 在一些涉及推理、代码、多轮对话等场景下模型的能力仍有待提高。

基于以上模型局限性，我们要求开发者仅将我们开源的代码、数据、模型及后续用此项目生成的衍生物用于研究目的，不得用于商业，以及其他会对社会带来危害的用途。

<br/>

## 📌引用

如果使用本项目的代码、数据或模型，请引用本项目。

```
@misc{BELLE,
  author = {Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Baochang Ma and Xiangang Li},
  title = {BELLE: Be Everyone's Large Language model Engine },
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

## 📚 模型使用例子

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

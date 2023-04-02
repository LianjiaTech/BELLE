## 已开放的模型

BELLE项目目标是促进中文对话大模型开源社区的发展，愿景做能帮到每一个人的LLM Engine。现阶段本项目基于一些开源预训练大语言模型（如BLOOM、LAMMA等），针对中文做了优化，模型调优仅使用由ChatGPT生产的数据（不包含任何其他数据）。

<br/>

## 局限性和使用限制

基于当前数据和基础模型训练得到的SFT模型，在效果上仍存在以下问题：

1. 在涉及事实性的指令上可能会产生违背事实的错误回答。

2. 对于具备危害性的指令无法很好的鉴别，由此会产生危害性言论。

3. 在一些涉及推理、代码等场景下模型的能力仍有待提高。

基于以上模型局限性，我们要求开发者仅将我们开源的代码、数据、模型及后续用此项目生成的衍生物用于研究目的，不得用于商业，以及其他会对社会带来危害的用途。

<br/>

## 调优BLOOMZ-7B1-mt模型

我们采取了不同大小规模（20万、60万、100万和200万样本）的指令学习的数据集训练模型，基于BLOOMZ-7B1-mt训练调优后的模型，现已开放:
| Datasize| 200,000 | 600,000 | 1,000,000 | 2,000,000 |
| ----- | ----- | ----- | ----- | ----- |
| Finetuned Model | [BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M) | [BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M) | [BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M) | [BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M) |

此外，方便大家使用，也对模型进行了量化[基于GPTQ量化后的模型](https://huggingface.co/BelleGroup/)，其中包含针对基础的模型上的4bit和8bit的量化模型。

### 模型效果比较

以Bloomz-7b1-mt为基础，我们评估了不同数量的instruction tuning数据，对模型效果的影响。总的来说，提升数据量能持续带来效果的提升，但是在不同类型的任务上表现有所不同。在Extract, Classification, Closed QA, 和Summarization任务上，增加数据能持续带来效果的提升，还未达到瓶颈。在Translation, Rewrite, 和Brainstorming任务上，几十万的数据量就能获得较好的效果。在Math, Code, 和COT任务上，模型效果较差，而且增加数据量已经无法带来效果的提升。
![Image text](../assets/model_compare.jpg)
<br/>
详见论文：[Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases](https://arxiv.org/abs/2303.14742)。
<br/>

## 基于[huggingface的LLaMA实例](https://huggingface.co/decapoda-research)LLAMA-HF调优了后的模型

请注意，不能保证是基于原版的LLaMA模型调优的结果，考虑到LLaMA的license约束，目前也仅供学习交流。请严遵守LLaMA的使用限制。建议大家给予训练脚本和开放数据调优模型。
| Datasize | 600,000 | 2,000,000 | 2,000,000 |
| ----- | ----- | ----- | ----- |
| Modelsize | 7B | 7B | 13B |
| Finetuned Model | [BELLE-LLAMA-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-0.6M) | [BELLE-LLAMA-7B-2M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M) | [BELLE-LLAMA-13B-2M](https://huggingface.co/BelleGroup/BELLE-LLAMA-13B-2M) |


***


## Models trained

The goal of this project is to promote the development of the open-source community for Chinese language large-scale conversational models, and our vision is to help building large language model engine for everyone. This project optimizes Chinese performance based on opensource pretrained large language models. These models finetuning uses only data generated via ChatGPT (without other data). 

<br/>


## Limitation and Usage Limits
There still exists a few issues in the model trained on current base model and data:

1. The model might generate factual errors when asked to follow instructions related to facts.

2. Occasionally generates harmful responses since the model still struggles to identify potential harmful instructions.

3. Needs improvements on reasoning and coding.

Since the model still has its limitations, we require developers only use the open-sourced code, data, model and any other artifacts generated via this project for research purposes. Commercial use and other potential harmful use cases are not allowed.


## Finetuned BLOOMZ-7B1-mt Model

We trained models on instruction learning datasets of different sizes (200,000, 600,000, 1 million, and 2 million samples) and based on the BLOOMZ-7B1-mt trained and optimized model. They are now release for use, you can download the checkpoints in [haggingface BELLE group](https://huggingface.co/BelleGroup):
| Datasize| 200,000 | 600,000 | 1,000,000 | 2,000,000 |
| ----- | ----- | ----- | ----- | ----- | 
| Finetuned Model | [BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M) | [BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M) | [BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M) | [BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M) |

In addition, for the convenience of users, we have also quantized the [model](https://huggingface.co/BelleGroup/) based on GPTQ, which includes 4-bit and 8-bit quantized models.


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


## Model based on [HuggingFace version of LLaMA](https://huggingface.co/decapoda-research) LLAMA-HF finetuning
Attention: It cannot be guaranteed that the model is based on the original LLaMA. Considering LLaMA's license constraints, the model is for research and learning only. Please strictly respect LLaMA's usage policy. Users are suggested to finetune the model with open-source scripts and datasets.

| Datasize | 600,000 | 2,000,000 | 2,000,000 |
| ----- | ----- | ----- | ----- |
| Modelsize | 7B | 7B | 13B |
| Finetuned Model | [BELLE-LLAMA-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-0.6M) | [BELLE-LLAMA-7B-2M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M) | [BELLE-LLAMA-13B-2M](https://huggingface.co/BelleGroup/BELLE-LLAMA-13B-2M) |

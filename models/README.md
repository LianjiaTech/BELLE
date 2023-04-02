## Models trained

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
![Image text](assets/model_compare.jpg)
<br/>
详见论文：[Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases](https://arxiv.org/abs/2303.14742)。
<br/>

## 基于[haggingface的llama实例](https://huggingface.co/decapoda-research)LLAMA-HF调优了后的模型

请注意，不能保证是基于原版的llama模型调优的结果，考虑到llama的license约束，目前也仅供学习交流。请严遵守LLaMA的使用限制。建议大家给予训练脚本和开放数据调优模型。
| Datasize | 600,000 | 2,000,000 | 2,000,000 |
| ----- | ----- | ----- | ----- |
| Modelsize | 7B | 7B | 13B |
| Finetuned Model | [BELLE-LLAMA-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-0.6M) | [BELLE-LLAMA-7B-2M](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M) | [BELLE-LLAMA-13B-2M](https://huggingface.co/BelleGroup/BELLE-LLAMA-13B-2M) |


*Read this in [English](README_en.md).*
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
| :-----: | :-----: | :-----: | :-----: | :-----: |
| Finetuned Model | [BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M) | [BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M) | [BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M) | [BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M) |

此外，方便大家使用，也对模型进行了量化[基于GPTQ量化后的模型](https://huggingface.co/BelleGroup/)，其中包含针对基础的模型上的4bit和8bit的量化模型。

### 模型效果比较

以Bloomz-7b1-mt为基础，我们评估了不同数量的instruction tuning数据，对模型效果的影响。总的来说，提升数据量能持续带来效果的提升，但是在不同类型的任务上表现有所不同。在Extract, Classification, Closed QA, 和Summarization任务上，增加数据能持续带来效果的提升，还未达到瓶颈。在Translation, Rewrite, 和Brainstorming任务上，几十万的数据量就能获得较好的效果。在Math, Code, 和COT任务上，模型效果较差，而且增加数据量已经无法带来效果的提升。
![Image text](../assets/model_compare.jpg)
<br/>
详见论文：[Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases](https://arxiv.org/abs/2303.14742)。
<br/>

## 调优LLaMA模型

考虑到LLaMA模型的限制，调优后的模型只能用作研究和学习使用，请严格遵守LLaMA的使用约束。LLaMA模型不允许发布调优后的完整模型权重，但是可以发布原始的模型的diff。因此，我们使用文件间的XOR，保证拥有LLaMA原始模型授权的人才可以将本项目发布的模型转化成可以使用的格式。文件XOR的代码参考[point-alpaca](https://github.com/pointnetwork/point-alpaca) 
### 模型列表
* [BELLE-LLaMA-7B-0.6M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-0.6M-enc)
* [BELLE-LLaMA-7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc)
* [BELLE-LLaMA-7B-2M-gptq-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-gptq-enc)
* [BELLE-LLaMA-13B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-13B-2M-enc)

### 使用说明
1. 从[LLaMA](https://github.com/facebookresearch/llama)官方获取7B/13B模型的pth文件，放到`/path/to_original_llama_7B`目录
2. 从[Huggingface Belle Group](https://huggingface.co/BelleGroup/) 下载发布的LLaMA模型diff，放到`/path/to_encrypted`目录
3. 运行下面的命令
```bash
mkdir /path/to_finetuned_model
for f in "/path/to_encrypted"/*; \
    do if [ -f "$f" ]; then \
       python3 decrypt.py "$f" "/path/to_original_llama_7B/consolidated.00.pth" "/path/to_finetuned_model/"; \
    fi; \
done
```
4. 参照Huggingface的README，检查`/path/to_finetuned_model/`目录文件的md5值
5. GPTQ量化模型推理代码参照[GPTQ推理代码](https://github.com/LianjiaTech/BELLE/tree/main/gptq)；非量化模型代码参照[基于transformers推理代码](https://github.com/LianjiaTech/BELLE/tree/main/train)

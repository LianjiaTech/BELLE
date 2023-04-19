### [Towards Better Instruction Following Language Models for Chinese: Investigating the Impact of Training Data and Evaluation](https://github.com/LianjiaTech/BELLE/blob/main/docs/Towards%20Better%20Instruction%20Following%20Language%20Models%20for%20Chinese.pdf)

# 论文简介

这篇论文研究了基于当前公开的指令数据训练得到的模型，能达到怎样的效果。我们在自有的1000条测试集上进行了量化评估，同时，为了提高模型在中文领域的性能和训练/推理效率，我们进一步扩展了LLaMA的词汇表，并在34亿个中文词汇上进行了二次预训练。

我们使用到的公开指令训练数据有：

1. GPT-3.5生成的Stanford alpaca 中文以及英文数据
2. GPT-4生成的Alpaca 中文以及英文数据
3. 用户分享的与ChatGPT的真实对话数据sharegpt

我们着眼于探究训练数据类别对模型性能的影响。具体而言，我们考察了训练数据的数量、质量和语言分布等因素。

实验结果如下：

<table>
  <tr>
    <td> Factor </td>
    <td> Base model </td>
    <td> Training data </td>
    <td> Score_w/o_others </td>
  <tr>
    <td rowspan="2">词表扩充</td>
    <td> LLaMA-7B-EXT </td>
    <td> zh(alpaca-3.5&4) + sharegpt </td>
    <td> 0.670 </td>
  </tr>
  <tr>
    <td> LLaMA-7B </td>
    <td> zh(alpaca-3.5&4) + sharegpt </td>
    <td> 0.652</td>
  </tr>
  <tr>
    <td rowspan="2">数据质量</td>
    <td> LLaMA-7B-EXT </td>
    <td> zh(alpaca-3.5) </td>
    <td> 0.642 </td>
  </tr>
  <tr>
    <td> LLaMA-7B-EXT </td>
    <td> zh(alpaca-4) </td>
    <td> 0.693 </td>
  </tr>
  <tr>
    <td rowspan="4">数据语言分布</td>
    <td> LLaMA-7B-EXT </td>
    <td> cn(alpaca-3.5&4) </td>
    <td> 0.679 </td>
  </tr>
  <tr>
    <td> LLaMA-7B-EXT </td>
    <td> en(alpaca-3.5&4) </td>
    <td> 0.659 </td>
  </tr>
  <tr>
    <td> LLaMA-7B-EXT </td>
    <td> zh(alpaca-3.5&4) + sharegpt </td>
    <td> 0.670 </td>
  </tr>
  <tr>
    <td> LLaMA-7B-EXT </td>
    <td> en(alpaca-3.5&4) + sharegpt </td>
    <td> 0.668 </td>
  </tr>
  <tr>
    <td rowspan="2">数据规模</td>
    <td> LLaMA-7B-EXT </td>
    <td> zh(alpaca-3.5&4) + sharegpt </td>
    <td> 0.670 </td>
  </tr>
  <tr>
    <td> LLaMA-7B-EXT </td>
    <td> zh(alpaca-3.5&4) + sharegpt <br>+ BELLE-0.5M-CLEAN</td>
    <td> 0.762</td>
  </tr>
  <tr>
    <td>-</td>
    <td>ChatGPT</td>
    <td>-</td>
    <td>0.824</td>
</table>

其中**BELLE-0.5M-CLEAN**是从我们内部的230万指令数据中清洗得到0.5M数据，其中包含单轮和多轮对话数据，和之前开放的0.5M数据不是同一批数据。这份数据还未开源，但是我们已经将实验中效果最好的模型 (score 0.762) 开源在[Hugging Face](https://huggingface.co/BelleGroup/BELLE-on-Open-Datasets).

# 准备数据集

### 下载数据集

我们的论文中使用了五个开源数据集：

| Data           | URL                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------- |
| alpaca-3.5-en  | https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json                            |
| alpaca-3.5-zh  | https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/dat                                        |
| alpaca-4-en    | https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json    |
| alpaca-4-zh    | https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json |
| sharegpt$^1$ | https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main                |

$^1$:请注意，sharegpt 数据集可能会持续更新，使得与我们论文中使用的略有不同，但不会对实验结论有大的影响。

### 预处理

##### 清洗 sharegpt 数据集

我们采用了[Vicuna](https://github.com/lm-sys/FastChat/blob/main/docs/commands/data_cleaning.md)中的数据清洗方法。

1. 通过 `fastchat.data.clean_sharegpt` 将 html 转换为 markdown
2. 通过 `fastchat.data.optional_clean` 删除除英文和中文之外的其他语言
3. 通过 `fastchat.data.split_long_conversation` 将最大长度为 2048 个令牌的对话分割开

#### 统一数据格式

我们将所有数据集统一为以下形式：

```python
{
    "id": "uniq_sample_id",
    "conversations": [
        {"from": "human", "value": "你好"},
        {"from": "assistant", "value": "你好，有什么可以帮助你的吗？"},
        {"from": "human", "value": "今天天气怎么样？"},
        {"from": "assistant", "value": "不好意思，我无法回答你的问题，因为我不知道你的位置信息，同时我目前还无法获取到最新的天气信息。"}
    ]
}

```

# 下载 LLaMA-EXT-7B

LLaMA-EXT-7B 是基于 [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) 的模型，进一步在 34 亿中文词汇上进行预训练，将模型词表大小扩展为 79,458。
现在它已经在[Hugging Face上线](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-7B)，你应该基于它进行后续的模型训练。

# 训练

以 LLaMA-EXT-7B 为基础模型，我们以如下超参数对模型进行训练。

| Hyper parameter   | Value  |
| ----------------- | ------ |
| Precision         | bf16   |
| Epochs            | 3      |
| Batch size        | 32     |
| Learning rate     | 5e-6   |
| Weight decay      | 0      |
| Warmup ratio      | 0.03   |
| LR scheduler type | cosine |
| Max length        | 2048   |

可以使用我们开源的[代码](../README.md)进行模型训练。

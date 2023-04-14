*[中文README](README.md).*

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
| :-----: | :-----: | :-----: | :-----: | :-----: | 
| Finetuned Model | [BELLE-7B-0.2M](https://huggingface.co/BelleGroup/BELLE-7B-0.2M) | [BELLE-7B-0.6M](https://huggingface.co/BelleGroup/BELLE-7B-0.6M) | [BELLE-7B-1M](https://huggingface.co/BelleGroup/BELLE-7B-1M) | [BELLE-7B-2M](https://huggingface.co/BelleGroup/BELLE-7B-2M) |

In addition, for the convenience of users, we have also quantized the [model](https://huggingface.co/BelleGroup/) based on GPTQ, which includes 4-bit and 8-bit quantized models.

### Model performance comparison 

Based on the Bloomz-7b1-mt model, we evaluated the impact of different amounts of instruction data on our released models' performance. 
Overall, increasing the amount of data consistently improved performance, but the extent of improvement varied across different types of tasks. 
For Extract, Classification, Closed QA, and Summarization tasks, increasing data continued to improve performance without reaching a plateau. 
For Translation, Rewrite, and Brainstorming tasks, good performance could be achieved with only hundreds of thousands of data. 
However, for Math, Code, and COT tasks, these models' performance were poor, and increasing data did not lead to further improvement.
![Image text](../assets/model_compare.jpg)
<br/>
More details are in paper [Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases](https://arxiv.org/abs/2303.14742)。
<br/>

## Finetuned LLaMA Model

Considering LLaMA's license constraints, the model is for research and learning only. Please strictly respect LLaMA's usage policy. We are not allowed to publish weights for LLaMA, of course, even finetuned, but there is no problem publishing the difference, a patch that we suggest to apply to the files. The encryption is a simple XOR between files, ensuring that only the people that have access to the original weights (from completely legal sources, of course) can transform them into finetuned weights. The encryption code is based on [point-alpaca](https://github.com/pointnetwork/point-alpaca) .

### Model list
* [BELLE-LLaMA-7B-0.6M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-0.6M-enc)
* [BELLE-LLaMA-7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc)
* [BELLE-LLaMA-7B-2M-gptq-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-gptq-enc)
* [BELLE-LLaMA-13B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-13B-2M-enc)
### Usage
1. From [LLaMA](https://github.com/facebookresearch/llama) download 7B/13B model's pth file，put it to `/path/to_original_llama_7B/` directory
2. From [Huggingface Belle Group](https://huggingface.co/BelleGroup/) download finetuned LLaMA model diff，put it to `/path/to_encrypted` directory
3. Run 
```bash
mkdir /path/to_finetuned_model
for f in "/path/to_encrypted"/*; \
    do if [ -f "$f" ]; then \
       python3 decrypt.py "$f" "/path/to_original_llama_7B/consolidated.00.pth" "/path/to_finetuned_model/"; \
    fi; \
done
```
4. Check the md5 value of `/path/to_finetuned_model/` directory
5. [GPTQ infer code](https://github.com/LianjiaTech/BELLE/tree/main/gptq)；[transformers infer code](https://github.com/LianjiaTech/BELLE/tree/main/train)

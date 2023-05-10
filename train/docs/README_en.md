# BELLE Training Code

 | [English](https://github.com/LianjiaTech/BELLE/blob/main/train/docs/README_en.md) | [中文](https://github.com/LianjiaTech/BELLE/blob/main/train/README.md) 

## 1. Set up the environment

### 1.1 Docker image

We provide a fully runnable Docker image, with the Dockerfile located in the docker folder.

Considering the difficulty of building the image, we offer an image download. You can use the following command to pull our image from DockerHub and run the code in the image environment.

```bash
docker pull belleagi/belle:v1.0
git clone https://github.com/LianjiaTech/BELLE.git
docker run -it --runtime=nvidia --shm-size="40g" -v /path/BELLE/train:/workspace/BELLE-train -v /path/huggingface_models/:/workspace/model_name_or_path belleagi/belle:v1.0 /bin/bash
```



The above command accomplishes the following:

1. Pull the docker image
2. Clone the BELLE repository
3. Mount the BELLE/train directory with the /workspace/BELLE-train directory in the Docker environment
4. Mount the huggingface_models directory with the /workspace/model_name_or_path directory in the Docker environment. The huggingface_models directory represents the save path for pre-trained language models, which stores all required pre-trained language models, such as llama-7b, bloomz-7b1-mt, etc.
5. Note: The mounted directories must be absolute paths

### 1.2 Conda

If you cannot use Docker due to machine constraints or other reasons, you can also create an environment with Conda and install the required packages using pip.

```bash
pip install -r requirements.txt
```



However, installing DeepSpeed with pip may fail during installation or runtime. The [FAQ](https://github.com/LianjiaTech/BELLE/blob/main/train/docs/FAQ.md) provides some tutorials on installing DeepSpeed and possible issues you may encounter.

## 2. Run

### 2.1 Download data

```bash
wget https://huggingface.co/datasets/BelleGroup/school_math_0.25M/resolve/main/school_math_0.25M.json
```



This dataset is for math problem-solving tasks, including the problem-solving process, with a total of about 250,000 entries. Example:

| Instruction                                                  | Answer                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？ | 一箱鸡蛋有25个，小明每天吃9个鸡蛋，用除法计算一下即可： 25 ÷ 9 = 2余7 也就是说，小明第1天到第2天共吃2个鸡蛋，第3天吃剩下的7个鸡蛋。 因此，一箱鸡蛋能够供小明吃2天。 |
| 小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？ | 该题可以用减法来解决。小明手中有15颗糖果，每人分得3颗，所以小红和小李一共分得6颗糖果（3+3=6）。 2. 然后再用原来的糖果总数减去分给小红和小李的糖果数，即可得到小明手中剩余的糖果数。 计算过程如下： 15 - 6 = 9 所以，小明还剩下9颗糖果。 |

#### 2.1.1 Convert data format

```bash
python scripts/convert_to_conv_data.py --orig_data school_math_0.25M.json --write_data school_math_0.25M_conv.json --dataset_name bellemath
head -n 1000 school_math_0.25M_conv.json > belleMath-dev1K.json
tail -n +1001 school_math_0.25M_conv.json > belleMath.json
```



We select the first 1000 entries as the validation set, and the rest as the training set.

**Note:** Currently, the repository code needs to be compatible with both single-turn and multi-turn data. Therefore, for previously open-sourced data (formatted as [instruction, output], most of the open-sourced single-turn instruction data is in this format), you need to use the convert_to_conv_data.py script to convert it to the following unified format:

```bash
{"id": "bellemath42", "conversations": [{"from": "human", "value": " 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？"}, {"from": "assistant", "value": "\n一箱鸡蛋有25个，小明每天吃9个鸡蛋，用除法计算一下即可：\n25 ÷ 9 = 2余7\n也就是说，小明第1天到第2天共吃2个鸡蛋，第3天吃剩下的7个鸡蛋。\n因此，一箱鸡蛋能够供小明吃2天。"}]}
{"id": "bellemath80", "conversations": [{"from": "human", "value": "小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？"}, {"from": "assistant", "value": " \n1. 该题可以用减法来解决。小明手中有15颗糖果，每人分得3颗，所以小红和小李一共分得6颗糖果（3+3=6）。\n2. 然后再用原来的糖果总数减去分给小红和小李的糖果数，即可得到小明手中剩余的糖果数。 \n计算过程如下：\n15 - 6 = 9\n所以，小明还剩下9颗糖果。"}]}
```




Other training data can be found at: https://huggingface.co/BelleGroup. Convert the format according to the above process.

For multi-turn conversation data, [shareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main) is an open-source large-scale multi-turn conversation dataset. You can refer to our work: [Towards Better Instruction Following Language Models for Chinese: Investigating the Impact of Training Data and Evaluation](https://arxiv.org/pdf/2304.07854.pdf) for specific performance.

The current code supports training this multi-turn conversation data. Download the data:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```



### 2.2 Model Training

The training start script is located in scripts/run.sh, and you need to modify the parameters in run.sh according to your actual requirements.

```bash
bash scripts/run.sh
```



- model_name_or_path represents the pre-trained model (if it is a LLaMA model, it needs to be converted to hf format beforehand to be loaded with from_pretrained)
- train_file represents the training data
- validation_file represents the validation data
- output_dir represents the path for saving training logs and models
- cache_dir represents the path for caching data processing
- cutoff_len represents the maximum input sequence length (for LLaMA models, it is recommended to set it to 1024 or above; for Bloom models, set it to 512 or above)

The run.sh script contains both full parameter fine-tuning and LoRA training start commands. Here, we will briefly explain the meaning of each parameter in the start command.

#### 2.2.1 Full parameter fine-tuning

The following command is for full-parameter fine-tuning on a single machine with multiple cards, using deepspeed and the base model LLaMA:

```bash
torchrun --nproc_per_node 8 train.py \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --deepspeed configs/deepspeed_config.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 8e-6 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --fp16 True \
    --seed 1234 \
    --gradient_checkpointing True \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir}
```



**Parameter Description**

1. If you want to train with a single card, just set nproc_per_node to 1.
2. If the pre-trained model is not LLaMA, remove --llama. If it is a LLaMA model, you need to specify --llama. This is because LLaMA models need to use LLamaTokenizer for loading, and using AutoTokenizer for llama may cause infinite recursion problems, which is related to the transformers version.
3. If the runtime environment does not support deepspeed, remove --deepspeed.

Deepspeed parameter configuration can refer to:

1. https://www.deepspeed.ai/docs/config-json/
2. https://huggingface.co/docs/accelerate/usage_guides/deepspeed

Training logs and models are saved in the output_dir directory, and the file structure in the directory should be as follows:

```Arduino
output_dir/
├── checkpoint-244/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── trainer_state.json
├── checkpoint-527/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── trainer_state.json
├── pytorch_model.bin
├── print_log.txt
└── config.json
```



trainer_state.json records the changes in loss and learning_rate.

#### 2.2.2 LoRA

```bash
torchrun --nproc_per_node 8 train.py \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora True \
    --use_int8_training \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 8e-6 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --fp16 True \
    --seed 1234 \
    --gradient_checkpointing True \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir}
```



**Parameter Description**

- use_lora represents using LoRA for training.
- use_int8_training represents using 8-bit quantization training, which can significantly reduce GPU memory usage.
- lora_config provides the parameter configuration for LoRA. If training the Bloom model, change it to configs/lora_config_bloom.json.

The file structure of the output_dir directory is as follows:

```Arduino
output_dir/
├── checkpoint-244/
│   ├── pytorch_model.bin
│   └── trainer_state.json
├── checkpoint-527/
│   ├── pytorch_model.bin
│   └── trainer_state.json
├── adapter_model.bin
├── print_log.txt
└── adapter_config.json
```



**Note: The saved model adapter_model.bin after LoRA training may be an empty file. In this case, you need to copy the pytorch_model.bin saved in other checkpoint-step to the output_dir directory. For example:**

```bash
cd output_dir
cp checkpoint-527/pytorch_model.bin adapter_model.bin
```



**Make sure adapter_model.bin is a valid LoRA weight.**

#### 2.2.3 Merging LoRA Weights

If you want to merge the LoRA weights with the pre-trained model, you can run the following command:

```bash
bash scripts/merge_lora.sh
```

The merged weights are saved in the output_path directory and can be loaded directly using from_pretrained later.

## 3. Inference

### 3.1 Inference

If you see this section, it means you have completed the training. Now we load the trained model to verify the text generation effect.

```bash
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --model_name_or_path model_name_or_path \
    --ckpt_path ckpt_path \
    --llama \
    --use_lora
```



**Parameter Description:**

- model_name_or_path is the path to the native pre-trained model.
- ckpt_path is the path to the saved model after training, which is the output_dir.
- llama indicates whether the base model is a LLaMA model.
- use_lora indicates whether the ckpt_path is LoRA weight.

**Note: The saved model adapter_model.bin after LoRA training may be an empty file. In this case, you need to copy the pytorch_model.bin saved in other checkpoint-step to the output_dir directory.**

In addition, if you have already merged the LoRA weights with the pre-trained model, you only need to specify ckpt_path as the path where the merged weights are saved, and there is no need to specify use_lora.

### 3.2 WebUI

We also provide a simple interactive web interface based on gradio. To start the service:

```bash
CUDA_VISIBLE_DEVICES=0 python src/interface.py \
    --model_name_or_path model_name_or_path \
    --ckpt_path ckpt_path \
    --llama \
    --use_lora
```



The service access address is hostip:17860.

![webUI](https://github.com/LianjiaTech/BELLE/blob/main/train/docs/interface.png)

## 4. Additional Notes

### 4.1 Usage of LLaMA Model

#### 4.1.1 Converting official LLaMA weights from Facebook to Hugging Face format

First, you need to obtain access to the LLaMA model from [facebookresearch/llama](https://github.com/facebookresearch/llama) and download the official checkpoints.

```bash
python training_scripts/convert_llama_weights_to_hf.py --input_dir download_official_llama_path --model_size 7B --output_dir xx/llama-7b-hf
```



When running the training script, change the `model_name_or_path` to `xx/llama-7b-hf`.

#### 4.1.2 Converting BELLE-LLaMA to Hugging Face format

Due to the usage constraints of the LLaMA model, we can only open-source the diff from the original model, such as [BELLE-LLaMA-7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc). Once you have obtained access to the LLaMA model from [facebookresearch/llama](https://github.com/facebookresearch/llama), you can refer to https://github.com/LianjiaTech/BELLE/tree/main/models to convert the model. The converted model will be our instruction-optimized LLaMA model.

## 5. Feedback and Questions

If you have any questions, please submit them in GitHub Issues. Before encountering a problem, please search for a solution to a similar issue in the [FAQ](https://github.com/LianjiaTech/BELLE/blob/main/train/docs/FAQ.md) .

The above text content is in a Chinese Markdown file. Please generate the corresponding English version of the Markdown file. The contents of tables and code cells do not need to be translated and do not need to be output.
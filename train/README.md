# Usage

## Install conda environment

```bash
conda env create -f environment.yml
conda activate Belle
conda install -c nvidia libcusolver-dev
```


## Download dataset

```bash
python download_data.py
```

The open source data will be downloaded into folder data_dir/, and will be randomly split into training and validation sets.

## Train
We support two training modes, one utilizes deepspeed to fine-tune the model, and the second one trains using Lora.

### Deepspeed 
#### Bloom
```bash
deepspeed --num_gpus=8 finetune.py --model_config_file run_config/Bloom_config.json  --deepspeed run_config/deepspeed_config.json 
```

#### Llama
```bash
deepspeed --num_gpus=8 finetune.py --model_config_file run_config/Llama_config.json  --deepspeed run_config/deepspeed_config.json 
```


### Lora
#### Bloom
```bash
torchrun --nproc_per_node=8 finetune.py --model_config_file run_config/Bloom_config.json --lora_hyperparams_file run_config/lora_hyperparams_bloom.json  --use_lora
```

#### Llama
```bash
torchrun --nproc_per_node=8 finetune.py --model_config_file run_config/Llama_config.json --lora_hyperparams_file run_config/lora_hyperparams_llama.json  --use_lora
```


## Generate

Assuming the model used in the training phase is bloomz, and the training data used is Belle_open_source_0.5M, you can directly run the following command to get the model's generation results.

```bash
python generate.py --dev_file data_dir/Belle_open_source_0.5M.dev.json --model_name_or_path trained_models/bloom/
```

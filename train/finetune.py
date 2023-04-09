import os
os.environ["WANDB_DISABLED"] = "true"
import sys
from typing import List
import argparse, logging

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
import transformers
import json

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

def get_logger(logger_name,output_dir):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) 
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir,'log.txt'),mode='w')
    file_handler.setLevel(logging.INFO) 
    file_handler.setFormatter(
            logging.Formatter(
                    fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
            )
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
            logging.Formatter(
                    fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
            )
    logger.addHandler(console_handler)
    return logger


def train(args):
    model_config = json.load(open(args.model_config_file))
    model_type = model_config['model_type']
    model_name_or_path = model_config['model_name_or_path']
    data_path = model_config['data_path']
    output_dir = model_config['output_dir']
    cutoff_len = model_config['cutoff_len']

    logger = get_logger("train", model_config['output_dir'])
    logger.info("args.__dict__ : {}".format(args.__dict__))
    for key, value in model_config.items():
        logger.info("{} : {}".format(key, value))
    assert (
        model_name_or_path
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    gradient_accumulation_steps = model_config['batch_size'] // model_config['per_device_train_batch_size'] if "gradient_accumulation_steps" not in model_config else model_config['gradient_accumulation_steps']

    logger.info("per_device_train_batch_size = {}, gradient_accumulation_steps = {}".format(model_config["per_device_train_batch_size"], gradient_accumulation_steps))
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(gradient_accumulation_steps // world_size, 1)

    load_in_8bit = True if args.use_lora else False
    if model_type.lower() == "llama":
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit = load_in_8bit,
            device_map=device_map,
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    elif model_type.lower() == "bloom":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit = load_in_8bit,
            device_map=device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    tokenizer.pad_token_id = 0 
    tokenizer.padding_side = "left" 

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if add_eos_token and len(result["input_ids"]) >= cutoff_len:
            result["input_ids"][cutoff_len - 1] = tokenizer.eos_token_id
            result["attention_mask"][cutoff_len - 1] = 1

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        instruction = data_point['instruction']
        input_text = data_point["input"]
        input_text = "Human: " + instruction + input_text + "\n\nAssistant: " 
        input_text = tokenizer.bos_token + input_text if tokenizer.bos_token!=None else input_text
        target_text = data_point["output"] + tokenizer.eos_token
        full_prompt = input_text+target_text
        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            user_prompt = input_text
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ] 
        return tokenized_full_prompt

    if args.use_lora:
        model = prepare_model_for_int8_training(model)
        lora_hyperparams = json.load(open(args.lora_hyperparams_file))
        for key, value in lora_hyperparams.items():
            logger.info("{} : {}".format(key, value))
        config = LoraConfig(
            r=lora_hyperparams['lora_r'],
            lora_alpha=lora_hyperparams['lora_alpha'],
            target_modules=lora_hyperparams['lora_target_modules'] if model_config['model_type']=="Llama" else ["query_key_value"],
            lora_dropout=lora_hyperparams['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(config)
        model = get_peft_model(model, config)

    data = load_dataset("json", data_files=data_path)
    print(data)

    val_set_size = model_config['val_set_size']
    training_nums = len(data['train'])
    if val_set_size > 0:
        val_set_size = min(val_set_size, int(training_nums*model_config['val_set_rate']))
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    print("start train...")
    num_gpus = torch.cuda.device_count()
    t_total = (training_nums//(gradient_accumulation_steps*num_gpus + 1))*model_config["num_epochs"]
    warmup_steps = int(t_total * model_config.get("warmup_rate", 0.1))
    logger.info("num_gpus = {}, training_nums = {}, t_total = {}, warmup_steps = {}".format(num_gpus, training_nums, t_total, warmup_steps))
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=model_config['per_device_train_batch_size'],
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=model_config['num_epochs'],
            learning_rate=model_config['learning_rate'],
            fp16=True,
            logging_steps=model_config['logging_steps'],
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=model_config["eval_steps"] if val_set_size > 0 else None,
            save_steps=model_config["save_steps"],
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            deepspeed=args.deepspeed if not args.use_lora else None,
            group_by_length=True
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    if args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    print("trainer.train")
    trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)
    logger.info("Save checkpointing...")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above when using lora to train, please disregard :)")
    logger.info("Training succeeded")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_file", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, help="deepspeed config")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="either training checkpoint or final adapter")
    parser.add_argument("--train_on_inputs", action="store_true", help="Target loss only. If False, masks out inputs in loss")
    #parser.add_argument("--group_by_length", action="store_true", help="Faster, but produces an odd training loss curve,")
    parser.add_argument("--lora_hyperparams_file", default="", type=str, help="Provide it when use_lora=True")
    parser.add_argument("--use_lora", action="store_true", default=False, help="Use lora")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    train(args)

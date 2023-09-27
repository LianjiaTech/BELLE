# imports
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal
import os, sys
import math
import numpy as np
from src.dpo_trainer import CustomDPOTrainer

import warnings
from transformers import PreTrainedModel
import torch.nn as nn
from typing import Union, Any, List

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.trainer_pt_utils import get_model_param_count

from trl.trainer.utils import DPODataCollatorWithPadding

IGNORE_INDEX=-100

def ignore_warnings():
    warnings.filterwarnings('ignore')

ignore_warnings()

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "the model name"})

    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "optimizer learning rate"})

    weight_decay: Optional[float] = field(default=0, metadata={"help": "weight_decay"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "batch size per device"})

    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "batch size per device"})

    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )

    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})

    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})

    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    llama: bool = field(
        default=False,
        metadata={"help": "Llama model"}
    )

    seed: int = field(
        default=42,
        metadata={"help": "seed"}
    )

    ddp_timeout: int = field(
        default=1800,
        metadata={"help": "ddp_timeout"}
    )

    save_total_limit: int = field(
        default=None,
        metadata={"help": "save_total_limit"}
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "trust_remote_code"}
    )

    bf16: bool = field(
        default=False,
        metadata={"help": "bf16"}
    )

    fp16: bool = field(
        default=False,
        metadata={"help": "fp16"}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    logging_dir: str = field(
        default=None,
        metadata={"help": "logging_dir"}
    )

    output_dir: str = field(
        default=None,
        metadata={"help": "output_dir"}
    )

    save_strategy: str = field(
        default='steps',
        metadata={"help": "save_strategy"}
    )

    evaluation_strategy: str = field(
        default='steps',
        metadata={"help": "evaluation_strategy"}
    )
    metric_for_best_model: str = field(
        default='loss',
        metadata={"help": "metric_for_best_model"}
    )

    lr_scheduler_type: str = field(
        default='linear',
        metadata={"help": "lr_scheduler_type"}
    )

    eval_and_save_ratio_per_epoch: float = field(
        default=0.1,
        metadata={"help": "eval_and_save_ratio_per_epoch"}
    )

    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "num_train_epochs"}
    )
    
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "warmup_ratio"}
    )

    logging_steps: float = field(
        default=500,
        metadata={"help": "logging_steps"}
    )

    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "gradient_checkpointing"}
    )

    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "load_best_model_at_end"}
    )

    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "prediction_loss_only"}
    )

    deepspeed: str = field(
        default=None,
        metadata={"help": "deepspeed"}
    )

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

# Custom Defined Metric
def compute_metrics(eval_preds):
    # preds shape: (sample, )
    # labels shape: (sample, )
    preds, labels = eval_preds

    acc_mean = preds.mean()
    eval_dict = {
        'rewards/accuracies':acc_mean
    }
    
    return eval_dict


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch_dtype = (
        script_args.torch_dtype
        if script_args.torch_dtype in ["auto", None]
        else getattr(torch, script_args.torch_dtype)
    )

    # load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=script_args.trust_remote_code)

    # The use_cache=True option is incompatible with gradient checkpointing. Disable it for training.
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # load ref model & tokenizer
    model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=script_args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=script_args.trust_remote_code)
    tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.unk_token_id # https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py#L254

    # Load the dataset
    """dataset format: json
    example:
    ... ...
    {'chosen': " I haven't even thought about it.",
    'rejected': ' Ass.',
    'prompt': "\n\nHuman: What are some cuss words in english?\n\nAssistant: Here’s an incomplete list.\n\nAss, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch,\n\nHuman: What's your favorite one?\n\nAssistant:"}
    ... ...
    """
    train_dataset = load_dataset("json", data_files=script_args.train_file)['train'].shuffle()
    eval_dataset = load_dataset("json", data_files=script_args.validation_file)['train'].shuffle() if script_args.validation_file else None

    # cal some hyperparameters
    training_nums = len(train_dataset)
    eval_nums = len(eval_dataset) if script_args.validation_file else None
    num_gpus = torch.cuda.device_count()

    batch_size = script_args.per_device_train_batch_size * world_size * script_args.gradient_accumulation_steps
    t_total_per_epoch = int(math.ceil(training_nums/batch_size))
    t_total = t_total_per_epoch * script_args.num_train_epochs
    eval_and_save_steps = int(t_total_per_epoch * script_args.eval_and_save_ratio_per_epoch)
    eval_steps = eval_and_save_steps if script_args.validation_file else None
    save_steps = eval_and_save_steps
    warmup_steps = int(t_total*script_args.warmup_ratio)


    # initialize training arguments
    training_args = TrainingArguments(
        seed=script_args.seed,
        num_train_epochs=script_args.num_train_epochs,
        eval_steps=eval_steps,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        logging_steps=script_args.logging_steps,
        lr_scheduler_type=script_args.lr_scheduler_type,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        remove_unused_columns=False,
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        gradient_checkpointing=script_args.gradient_checkpointing,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        save_strategy=script_args.save_strategy,
        evaluation_strategy=script_args.evaluation_strategy,
        metric_for_best_model=script_args.metric_for_best_model,
        output_dir=script_args.output_dir,
        logging_dir=script_args.logging_dir,
        report_to=script_args.report_to,
        ddp_timeout=script_args.ddp_timeout,
        save_total_limit=script_args.save_total_limit,
        load_best_model_at_end=script_args.load_best_model_at_end,
        prediction_loss_only=script_args.prediction_loss_only,
        deepspeed=script_args.deepspeed
    )

    # initialize the DPO trainer
    dpo_trainer = CustomDPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if script_args.validation_file else None,
        tokenizer=tokenizer,
        data_collator=DPODataCollatorWithPadding(tokenizer=tokenizer, 
                                                 padding=True, 
                                                 max_length=script_args.max_length, 
                                                 max_prompt_length=script_args.max_prompt_length, 
                                                 label_pad_token_id=IGNORE_INDEX, 
                                                 padding_value=0,
                                                 truncation_mode='keep_end')
    )
    dpo_trainer.compute_metrics = compute_metrics
    
    global_rank = torch.distributed.get_rank()

    print_rank_0("*** *** Training configs *** ***", global_rank)
    print_rank_0(f"train example nums: {training_nums}", global_rank)
    if script_args.validation_file:
        print_rank_0(f"train example nums: {eval_nums}", global_rank)
    print_rank_0(f"***", global_rank)
    print_rank_0(f"num gpu: {num_gpus}", global_rank)
    print_rank_0(f"world size: {world_size}", global_rank)
    print_rank_0(f"***", global_rank)
    print_rank_0(f"epoch: {training_args.num_train_epochs}", global_rank)
    print_rank_0(f"per device train batch size: {training_args.per_device_train_batch_size}", global_rank)
    print_rank_0(f"gradient accumulation: {training_args.gradient_accumulation_steps}", global_rank)
    print_rank_0(f"warmup strategy: {training_args.lr_scheduler_type}", global_rank)
    print_rank_0(f"warmup ratio: {training_args.warmup_ratio}", global_rank)
    print_rank_0(f"eval and save ratio: {script_args.eval_and_save_ratio_per_epoch}", global_rank)
    print_rank_0(f"***", global_rank)
    print_rank_0(f"t_total per epoch: {t_total_per_epoch}", global_rank)
    print_rank_0(f"t_total: {t_total}", global_rank)
    print_rank_0(f"***", global_rank)
    print_rank_0(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True)}", global_rank)
    print_rank_0("*** *** Training configs *** ***", global_rank)

    # train
    dpo_trainer.train()

    print_rank_0("\n Training completed!!! If there's a warning about missing keys above, please disregard :)", global_rank)

if __name__ == "__main__":
    main()
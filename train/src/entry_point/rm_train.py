# coding=utf-8
from dataclasses import dataclass, field
from functools import partial
import math
import os
import sys
from typing import Any, Dict, List, Optional, Union

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    LlamaTokenizer,
)
from transformers.utils import PaddingStrategy
from transformers.trainer_utils import get_last_checkpoint
from trl import RewardConfig, RewardTrainer
from trl.trainer.utils import RewardDataCollatorWithPadding
import logging
from multiprocessing import cpu_count

tqdm.pandas()
accelerator = Accelerator()
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def print_rank_0(msg, log_file):
    if accelerator.is_main_process:
        with open(log_file, "a") as f:
            print(msg)
            f.write(msg + "\n")


@dataclass
class ScriptArguments:
    """
    Hyperparameters to fine-tune a reward model on a given dataset with the `RewardTrainer`.
    """

    # Training arguments
    report_to: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    logging_steps: Optional[int] = field(
        default=500, metadata={"help": "the number of update steps between two logs"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the batch size"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "evaluating batch size"}
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "the number of training epochs"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )
    fp16: Optional[bool] = field(default=False, metadata={"help": "float16"})
    bf16: Optional[bool] = field(default=True, metadata={"help": "bfloat16"})
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    weight_decay: float = field(
        default=0.001, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    warmup_steps: int = field(
        default=1000, metadata={"help": "Linear warmup over warmup_steps."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to a folder with a valid checkpoint for your model."
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    dataloader_drop_last: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Drop the last incomplete batch if it is not divisible by the batch size."
        },
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    # Other arguments
    model_name: Optional[str] = field(
        default="facebook/opt-350m", metadata={"help": "the model name"}
    )
    train_data: str = field(default="", metadata={"help": "train data path"})
    eval_data: str = field(default="", metadata={"help": "eval data path"})
    cache_dir: str = field(default="", metadata={"help": "cache dir"})
    use_llama: Optional[bool] = field(default=True, metadata={"help": "bfloat16"})
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    use_lora: Optional[bool] = field(
        default=False, metadata={"help": "Wether to use LoRA or not to train adapters"}
    )
    trust_remote_code: Optional[bool] = field(
        default=True, metadata={"help": "Enable `trust_remote_code`"}
    )
    seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )


# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
def preprocess_function(tokenizer: PreTrainedTokenizerBase, examples: Dict[str, Any]):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, add_special_tokens=False)
        tokenized_rejected = tokenizer(rejected, add_special_tokens=False)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(
            tokenized_rejected["attention_mask"]
        )

    return new_examples

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    log_file = os.path.join(script_args.output_dir, "print_log.txt")
    local_rank = accelerator.local_process_index

    # Load the dataset and pre-process it
    if script_args.use_llama:
        tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
        tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<unk>",
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
        tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})
    tokenizer.padding_side = "left"
    print_rank_0(
        f"unk token: {tokenizer.unk_token}, "
        f"unk token id: {tokenizer.unk_token_id}, "
        f"pad token: {tokenizer.pad_token}, "
        f"pad token id: {tokenizer.pad_token_id}",
        log_file,
    )

    with accelerator.main_process_first():
        train_dataset = load_dataset(
            "json", data_files=script_args.train_data, cache_dir=script_args.cache_dir
        )["train"]
        eval_dataset = load_dataset(
            "json", data_files=script_args.eval_data, cache_dir=script_args.cache_dir
        )["train"]

        # Preprocess the dataset and filter out examples that are longer than script_args.max_length
        train_dataset = train_dataset.map(
            partial(preprocess_function, tokenizer),
            batched=True,
            num_proc=max(cpu_count() // 2, 1),
            remove_columns=["chosen", "rejected"],
        )
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
            and len(x["input_ids_rejected"]) <= script_args.seq_length
        )

        eval_dataset = eval_dataset.map(
            partial(preprocess_function, tokenizer),
            batched=True,
            num_proc=max(cpu_count() // 2, 1),
            remove_columns=["chosen", "rejected"],
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
            and len(x["input_ids_rejected"]) <= script_args.seq_length
        )

    for i in range(2):
        print_rank_0("Eval tokenized example: {}".format(train_dataset[i]), log_file)
    for i in range(2):
        print_rank_0("Train tokenized example: {}".format(eval_dataset[i]), log_file)

    # Define the training arguments
    training_nums = len(train_dataset)
    global_batch_size = (
        accelerator.num_processes
        * script_args.gradient_accumulation_steps
        * script_args.per_device_train_batch_size
    )
    if script_args.dataloader_drop_last:
        num_steps = (
            math.floor(training_nums / global_batch_size) * script_args.num_train_epochs
        )
    else:
        num_steps = (
            math.ceil(training_nums / global_batch_size) * script_args.num_train_epochs
        )
    eval_steps = max(num_steps // (script_args.num_train_epochs * 4), 5)
    print_rank_0(
        "num_gpus = {}, training_nums = {}, num_steps = {}, warmup_steps = {}, eval_steps = {}, save_steps = {}".format(
            accelerator.num_processes,
            training_nums,
            num_steps,
            script_args.warmup_steps,
            eval_steps,
            eval_steps,
        ),
        log_file,
    )
    # `TrainingArguments` must be instantiated before loading model!!!
    training_args = RewardConfig(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        report_to="wandb" if script_args.report_to == "wandb" else "tensorboard",
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_steps=script_args.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        max_length=script_args.seq_length,
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        weight_decay=script_args.weight_decay,
        lr_scheduler_type=script_args.lr_scheduler_type,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        warmup_steps=script_args.warmup_steps,
        overwrite_output_dir=script_args.overwrite_output_dir,
        resume_from_checkpoint=script_args.resume_from_checkpoint,
        save_total_limit=script_args.save_total_limit,
        load_best_model_at_end=True,
        ddp_timeout=3600,
        seed=script_args.seed,
        dataloader_drop_last=script_args.dataloader_drop_last,
    )

    print_rank_0(
        "world_size = {}".format(training_args.world_size),
        log_file,
    )

    # Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError(
            "You can't load the model in 8 bits and 4 bits at the same time"
        )
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": local_rank}
    else:
        device_map = None
        quantization_config = None

    # Model must be loaded after create `TrainingArguments`!!!
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        num_labels=1,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Define the LoraConfig
    if script_args.use_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["scores"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Define the Trainer
    model.config.use_cache = False
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, pad_to_multiple_of=8
        ),
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    accelerator.wait_for_everyone()
    print_rank_0(
        "\n Training completed!!! If there's a warning about missing keys above, please disregard :)",
        log_file,
    )


if __name__ == "__main__":
    main()

# coding=utf-8
from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Union

import torch
from accelerate import Accelerator
from accelerate.utils import DummyOptim
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    HfArgumentParser,
    LlamaTokenizer,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
)
from transformers.utils import PaddingStrategy
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, set_seed
from trl.core import LengthSampler
from multiprocessing import cpu_count
from src.utils import prepare_deepspeed, zero_infer
from src.ppo_trainer import PPOTrainerForZero3 as PPOTrainer

accelerator = Accelerator()

tqdm.pandas()
import logging

logging.basicConfig(
    format=f"[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] [Rank {accelerator.process_index}] %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def print_rank_0(msg, log_file):
    if accelerator.is_main_process:
        with open(log_file, "a") as f:
            print(msg)
            f.write(msg + "\n")


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    reward_model_name: Optional[str] = field(
        default="", metadata={"help": "the reward model name"}
    )
    log_with: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    output_max_length: Optional[int] = field(
        default=128, metadata={"help": "maximum length for generation"}
    )
    mini_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the PPO minibatch size"}
    )
    eval_batch_size: Optional[int] = field(
        default=1,
        metadata={
            "help": "the batch size for reward model rating and actor generating"
        },
    )
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(
        default=4, metadata={"help": "the number of ppo epochs"}
    )
    data_epochs: Optional[int] = field(
        default=1, metadata={"help": "the number of data epochs"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(
        default=False, metadata={"help": "whether to use the adafactor optimizer"}
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "whether to early stop"}
    )
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "kl target for early stopping"}
    )
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    save_freq: Optional[int] = field(
        default=None, metadata={"help": "n steps to save the model"}
    )
    output_dir: Optional[str] = field(
        default="runs/", metadata={"help": "n steps to save the model"}
    )
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Initial KL penalty coefficient (used for adaptive and linear control)"
        },
    )
    adap_kl_ctrl: Optional[bool] = field(
        default=True, metadata={"help": "Use adaptive KL control, otherwise linear"}
    )
    do_sample: Optional[bool] = field(
        default=True, metadata={"help": "Do sample when generating"}
    )
    logging_dir: Optional[str] = field(default="logs", metadata={"help": "Logging dir"})
    use_llama: Optional[bool] = field(default=True, metadata={"help": "Use llama"})
    reward_model_use_llama: Optional[bool] = field(
        default=True, metadata={"help": "Reward model use llama"}
    )
    use_lora: Optional[bool] = field(default=False, metadata={"help": "Use lora"})
    train_data: str = field(default="", metadata={"help": "Train file"})
    cache_dir: Optional[str] = field(
        default="hf_cache_dir", metadata={"help": "Dataset cache dir"}
    )
    input_length: Optional[int] = field(
        default=512, metadata={"help": "Input token length"}
    )


def build_dataset(tokenizer, dataset_name, input_length, cache_dir="hf_cache_dir"):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    train_dataset = load_dataset(
        "json",
        data_files=dataset_name,
        cache_dir=cache_dir,
    )["train"]
    original_columns = train_dataset.column_names
    num_proc = max(cpu_count() // 2, 1)

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for text in examples["text"]:
            tokens = tokenizer(text, add_special_tokens=False, truncation=True)[
                "input_ids"
            ]
            new_examples["query"].append(text)
            new_examples["input_ids"].append(tokens)

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) <= input_length)
    return ds


@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {"query": [feature["query"] for feature in features]}
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        input_ids = self.tokenizer.pad(
            input_ids,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )["input_ids"].unbind(0)
        batch["input_ids"] = list(input_ids)
        return batch


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    log_file = os.path.join(script_args.output_dir, "print_log.txt")

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
        # We retrieve the dataloader by calling the `build_dataset` function.
        dataset = build_dataset(
            tokenizer,
            dataset_name=script_args.train_data,
            input_length=script_args.input_length,
            cache_dir=script_args.cache_dir,
        )
        for i in range(2):
            print_rank_0("Train tokenized example: {}".format(dataset[i]), log_file)
        print_rank_0(f"Train dataset length: {len(dataset)}", log_file)

    config = PPOConfig(
        steps=len(dataset) // accelerator.num_processes,
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=script_args.early_stopping,
        target_kl=script_args.target_kl,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed,
        init_kl_coef=script_args.init_kl_coef,
        adap_kl_ctrl=script_args.adap_kl_ctrl,
        project_kwargs={"logging_dir": script_args.logging_dir},
    )
    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.

    if script_args.use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        lora_config = None

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        peft_config=lora_config,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = True

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        peft_config=lora_config,
    )
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.use_cache = False

    optimizer = DummyOptim(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate
    )
    if script_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        optimizer=optimizer,
    )
    ppo_trainer.current_device = accelerator.device
    logger.debug("After ppo_trainer initialized")
    # We then build the text classification pipeline using our reward model, passing the
    # model name and the text classification pipeline arguments. Let's also make sure to
    # set the device to the same device as the PPOTrainer.
    if script_args.reward_model_use_llama:
        rw_tokenizer = LlamaTokenizer.from_pretrained(script_args.reward_model_name)
        rw_tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<unk>",
            }
        )
    else:
        rw_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_model_name)
        rw_tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})

    # 使用deepspeed做reward model的inference
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.reward_model_name, num_labels=1
    )
    reward_model.config.pad_token_id = rw_tokenizer.pad_token_id
    reward_model = prepare_deepspeed(accelerator, reward_model)

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        # "min_length": -1,
        "top_k": 0,
        "top_p": 1.0,
        "do_sample": script_args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    output_min_length = 32
    output_max_length = script_args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for data_epoch in tqdm(
        range(script_args.data_epochs),
        desc=f"rank: {accelerator.process_index}, data_epoch",
    ):
        # batch: 因为dataloader用accelerate prepare，prepare调用prepare_data_loader，prepare_data_loader会为dataloader加入分布式采样，因此每个进程都是不一样的batch
        for ppo_epoch, batch in tqdm(
            enumerate(ppo_trainer.dataloader),
            total=config.total_ppo_epochs,
            desc=f"rank: {accelerator.process_index}, ppo_epoch",
        ):
            # actor生成回复
            question_tensors = batch["input_ids"]
            logger.debug("Begin ppo_trainer.generate")
            output_strs = []
            response_tensors = []
            for i in range(0, script_args.batch_size, script_args.eval_batch_size):
                input_ids = batch["input_ids"][i : i + script_args.eval_batch_size]
                output_ids = ppo_trainer.generate(
                    input_ids,
                    return_prompt=False,
                    length_sampler=output_length_sampler,
                    **generation_kwargs,
                )
                response_tensors.extend(output_ids)
                output_strs.extend(
                    tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                )
            batch["response"] = output_strs
            logger.debug("After ppo_trainer.generate")

            # 获得rewards
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            outputs: List[Dict[str, Any]] = zero_infer(
                accelerator,
                reward_model,
                rw_tokenizer,
                texts,
                script_args.eval_batch_size,
            )
            rewards = []
            for output in outputs:
                rewards.extend(output.logits.tolist())
            rewards = [
                torch.tensor(reward, device=accelerator.device)
                - script_args.reward_baseline
                for reward in rewards
            ]

            # Run PPO step
            logger.debug("Begin ppo_trainer.step")
            stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
            logger.debug("After ppo_trainer.step")
            ppo_trainer.log_stats(stats, batch, rewards)
            accelerator.wait_for_everyone()
            # 保存
            total_ppo_epoch = data_epoch * config.total_ppo_epochs + ppo_epoch
            if (
                script_args.save_freq
                and (total_ppo_epoch + 1) % script_args.save_freq == 0
            ):
                ppo_trainer.save_pretrained(
                    f"{script_args.output_dir}/step_{total_ppo_epoch}"
                )
    ppo_trainer.save_pretrained(script_args.output_dir)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

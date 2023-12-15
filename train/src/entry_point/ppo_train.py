from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from accelerate.utils import DummyOptim
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_xpu_available
from multiprocessing import cpu_count


tqdm.pandas()


@dataclass
class ScriptArguments:
    data_epochs: Optional[int] = field(
        default=1, metadata={"help": "The number of overall training epochs."}
    )
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to save the model."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Dataset cache dir."}
    )
    input_length: Optional[int] = field(
        default=128, metadata={"help": "Maximum input token length."}
    )
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name=None,
            query_dataset=None,
            reward_model=None,
            learning_rate=1.41e-5,
            log_with=None,
            mini_batch_size=1,
            batch_size=2,
            gradient_accumulation_steps=2,
            optimize_device_cache=True,
            early_stopping=False,
            target_kl=0.1,
            ppo_epochs=4,
            kl_penalty="kl",
            seed=0,
            init_kl_coef=0.2,
            adap_kl_ctrl=True,
            project_kwargs={"logging_dir": "logs"},
            use_score_scaling=False,
            use_score_norm=False,
            score_clip=None,
        )
    )
    use_seq2seq: bool = field(default=False, metadata={"help": "Whether to use seq2seq models."})
    use_peft: bool = field(default=False, metadata={"help": "Whether to use peft."})
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})


args = tyro.cli(ScriptArguments)


# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead

# Let's begin building the datset!
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
    ds.set_format(type="torch")
    return ds


tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name, trust_remote_code=args.trust_remote_code)
# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, args.ppo_config.query_dataset, args.input_length, args.cache_dir)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# set seed before initializing value head for deterministic eval
set_seed(args.ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(args.ppo_config.model_name, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = args.peft_config
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = trl_model_class.from_pretrained(
    args.ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)

# dummy optimizer
optimizer = DummyOptim(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.ppo_config.learning_rate
)

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
# ppo_trainer = PPOTrainer(args.ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer)
ppo_trainer = PPOTrainer(
    args.ppo_config,
    model,
    ref_model,
    tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis (text-classification) pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
# sentiment-analysis is an alias of text-classification
# https://huggingface.co/docs/transformers/v4.36.0/en/main_classes/pipelines#transformers.pipeline.task
task = 'text-classification'
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        reward_pipe = pipeline(task, model=args.ppo_config.reward_model, device=device)
else:
    reward_pipe = pipeline(task, model=args.ppo_config.reward_model, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if reward_pipe.tokenizer.pad_token_id is None:
    reward_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if reward_pipe.model.config.pad_token_id is None:
    reward_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

# Train every data epoch
for data_epoch in tqdm(
    range(args.data_epochs),
    desc=f"Epoch",
):
    # Each epoch training
    for batch in tqdm(ppo_trainer.dataloader, desc='Training'):
        query_tensors = batch["input_ids"]

        # Get response from model
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute score score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_pipe_outputs = reward_pipe(ref_texts, **sent_kwargs)
        ref_rewards = [torch.tensor(output[0]["score"]) for output in ref_pipe_outputs]
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
    
    # Save for every epoch
    ppo_trainer._save_pretrained(args.output_dir)
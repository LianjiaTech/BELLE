import numpy as np
from transformers.utils import add_start_docstrings
from transformers.trainer_pt_utils import torch_distributed_zero_first
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    LlamaTokenizer,
    TrainingArguments,
    set_seed,
)
from transformers import GenerationConfig as HFGenerationConfig
from peft import PeftModel
from datasets import load_dataset
import transformers
import torch

from typing import Optional
from functools import partial
from dataclasses import dataclass, field
import os
import logging
import sys
import pandas as pd
from src.models.llama.modeling_llama import LlamaForCausalLM

from src.sample_generator import inference_generate
from src.trainer import MySeq2SeqTrainer as Seq2SeqTrainer

logger = logging.getLogger(__name__)


# import pudb
# import traceback
# import sys
# # 异常时中断
# def debug_on_exception(exctype, value, tb):
#     traceback.print_exception(exctype, value, tb)
#     pudb.post_mortem(tb)
# sys.excepthook = debug_on_exception


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    ckpt_path: str = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    lora_path: Optional[str] = field(default=None, metadata={"help": "Checkpoint path."})
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
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
    llama: bool = field(default=False, metadata={"help": "Llama model."})
    use_flash_attention: bool = field(default=False, metadata={"help": "Enable flash attention."})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    infer_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )


@dataclass
class GenerationConfig:
    max_new_tokens: int = field(
        default=512,
        metadata={
            "help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."
        },
    )
    min_new_tokens: int = field(
        default=0,
        metadata={
            "help": "The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt."
        },
    )
    do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use sampling ; use greedy decoding otherwise."
        },
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_k: int = field(
        default=50,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."
        },
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "The parameter for repetition penalty. 1.0 means no penalty."
        },
    )


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArguments(TrainingArguments):
    predict_with_generate: bool = field(
        default=True,
        metadata={"help": "Enable generation"},
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."},
    )
    report_to: str = field(
        default="wandb",
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )
    deepspeed: str = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )


def print_rank_0(msg, log_file, rank=0):
    if rank <= 0:
        with open(log_file, "a") as f:
            print(msg)
            f.write(msg + "\n")


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationConfig)
    )
    (
        model_args,
        data_args,
        training_args,
        generation_config,
    ) = parser.parse_args_into_dataclasses()

    global_rank = torch.distributed.get_rank()
    log_file = os.path.join(training_args.output_dir, "print_log.txt")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if not model_args.llama and model_args.use_flash_attention:
        logger.warning(
            "Only implement flash attention in llama-based model currently, "
            "set use_flash_attention = False"
        )
        model_args.use_flash_attention = False
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, fp16-bits training: {training_args.fp16}, bf16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    training_args.data_seed = training_args.seed

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    if model_args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.ckpt_path)
        print_rank_0(
            "Set the eos_token_id and bos_token_id of LLama model tokenizer",
            log_file,
            global_rank,
        )
        tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<unk>",
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.ckpt_path)
        tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})
    tokenizer.padding_side = "left"  # Allow batched inference

    if model_args.llama:
        model = LlamaForCausalLM.from_pretrained(
            model_args.ckpt_path,
            torch_dtype=torch_dtype
        )
        model.config.use_flash_attention = model_args.use_flash_attention
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.ckpt_path,
            torch_dtype=torch_dtype
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # peft model
    if model_args.use_lora:
        model = PeftModel.from_pretrained(
            model, 
            model_args.lora_path, 
            torch_dtype=torch_dtype
        )

    
    generation_config = vars(generation_config)
    generation_config["bos_token_id"] = tokenizer.bos_token_id
    generation_config["eos_token_id"] = tokenizer.eos_token_id
    generation_config["pad_token_id"] = tokenizer.pad_token_id

    print_rank_0(
        "tokenizer.eos_token_id = {}".format(tokenizer.eos_token_id),
        log_file,
        global_rank,
    )
    print_rank_0(
        "tokenizer.pad_token_id = {}".format(tokenizer.pad_token_id),
        log_file,
        global_rank,
    )
    print_rank_0(
        "tokenizer.bos_token_id = {}".format(tokenizer.bos_token_id),
        log_file,
        global_rank,
    )

    assert os.path.exists(data_args.infer_file), "{} file not exists".format(
        data_args.infer_file
    )

    with torch_distributed_zero_first(global_rank):
        infer_data = load_dataset(
            "json", data_files=data_args.infer_file, cache_dir=model_args.cache_dir
        )
        infer_data = infer_data["train"].map(
            partial(
                inference_generate,
                training_args.model_max_length,
                tokenizer,
                lambda input_text: f"Human: \n" + input_text + "\n\nAssistant:\n",
            )
        )

    for i in range(2):
        print_rank_0(
            "Eval tokenized example: {}".format(infer_data[i]), log_file, global_rank
        )

    num_gpus = torch.cuda.device_count()
    print_rank_0(
        f"val data nums = {len(infer_data)}, num gpus = {num_gpus}, total batch size = {num_gpus * training_args.eval_batch_size}",
        log_file,
        global_rank,
    )

    print_rank_0(
        f"Using {training_args.half_precision_backend} half precision backend",
        log_file,
        global_rank,
    )

    training_args.generation_config = HFGenerationConfig(**generation_config)
    # Trainer
    # https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
    # https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
    # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    # https://www.deepspeed.ai/docs/config-json/
    # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
    # https://huggingface.co/transformers/v4.10.1/main_classes/deepspeed.html
    # https://github.com/tatsu-lab/stanford_alpaca/issues/176    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8 , return_tensors="pt", 
            padding=True
        ),
    )
    
    predict_results = trainer.predict(
        infer_data,
        metric_key_prefix="predict",
        **training_args.generation_config.to_dict(),
    )

    print_rank_0(
        f"predict_runtime = {predict_results.metrics['predict_runtime']}, predict_samples_per_second = {predict_results.metrics['predict_samples_per_second']}, predict_steps_per_second = {predict_results.metrics['predict_steps_per_second']}",
        log_file,
        global_rank,
    )

    if global_rank in [0, -1]:
        predictions = predict_results.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]
        pd.DataFrame({"output": predictions}).to_json(
            f"{training_args.output_dir}/predictions.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )


if __name__ == "__main__":
    main()

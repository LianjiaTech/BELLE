#!/usr/bin/env python
# https://huggingface.co/docs/transformers/main/en/main_classes/deepspeed#custom-deepspeed-zero-inference
from functools import partial
import json
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    LlamaTokenizer,
    PreTrainedTokenizer,
    GenerationConfig,
)
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import os
import torch
from dataclasses import dataclass, field
from typing import Optional
from peft import PeftModel
import gradio as gr

from src.models.llama.modeling_llama import LlamaForCausalLM
from src.models.generation_utils import GenerationMixin
from src.utils import bind_methods_from_class_to_instance

SEP_LINE = "=" * 20

@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    ckpt_path: str = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    deepspeed: str = field(
        default=None,
        metadata={"help": "Deepspeed config."},
    )
    local_rank: int = field(
        default=None,
        metadata={"help": "Local rank."},
    )
    lora_path: Optional[str] = field(default=None, metadata={"help": "Checkpoint path."})
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA."})
    llama: bool = field(default=False, metadata={"help": "Llama model."})
    base_port: int = field(default=17860, metadata={"help": "Multi process bose port."})


def generate_prompt(input_text):
    return input_text


def evaluate(
    model,
    deepspeed_model,
    tokenizer: PreTrainedTokenizer,
    local_rank: int,
    input: str,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    do_sample=False,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=1.2,
):
    prompt = generate_prompt(input)

    print(SEP_LINE)
    print(f"local_rank: {local_rank}\nprompt:\n{prompt}")
    print(SEP_LINE)

    input_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device=local_rank)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,  # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens,  # min_length=min_new_tokens+input_sequence
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
    )
    
    with torch.no_grad():
        try:
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                deepspeed_model=deepspeed_model,
                synced_gpus=True,
                return_dict_in_generate=True,
                output_scores=False,
            )
        except Exception as e:
            print(e)
            sys.exit(-1)
        output = generation_output.sequences[0]
        output = tokenizer.decode(
            output, 
            skip_special_tokens=True
        )[len(prompt):].strip()

        print(SEP_LINE)
        print(f"local_rank: {local_rank}\nresult:\n{prompt + output}")
        print(SEP_LINE)
        return output


def main():
    parser = HfArgumentParser((Arguments,))
    args = parser.parse_args_into_dataclasses()[0]
    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    with open(args.deepspeed, "rt") as f:
        ds_config = json.load(f)
    # batch size has to be divisible by world_size, but can be bigger than world_size
    ds_config["train_batch_size"] = int(os.getenv("WORLD_SIZE", "1"))
    ds_config["train_micro_batch_size_per_gpu"] = 1

    # next line instructs transformers to partition the model directly over multiple gpus using
    # deepspeed.zero.Init when model's `from_pretrained` method is called.
    #
    # **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
    #
    # otherwise the model will first be loaded normally and only partitioned at forward time which is
    # less efficient and when there is little CPU RAM may fail
    dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

    if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_path)
        tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<unk>",
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
        tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})
    tokenizer.padding_side = "left"

    # now a model can be loaded.
    if args.llama:
        model = LlamaForCausalLM.from_pretrained(args.ckpt_path)
        model.config.use_flash_attention = True
    else:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # peft model
    if args.use_lora:
        model = PeftModel.from_pretrained(model, args.lora_path)

    # initialise Deepspeed ZeRO and store only the engine object
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    # inference
    ds_engine.module.eval()

    bind_methods_from_class_to_instance(
        ds_engine.module,
        GenerationMixin,
        include=[
            "contrastive_search",
            "greedy_search",
            "sample",
            "beam_search",
            "beam_sample",
            "group_beam_search",
            "constrained_beam_search",
            "assisted_decoding",
            "generate",
        ],
    )

    gr.Interface(
        fn=partial(evaluate, ds_engine.module, ds_engine, tokenizer, local_rank),
        inputs=[
            gr.components.Textbox(
                lines=2, label="Input", placeholder="Welcome to the BELLE model"
            ),
            gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=1, label="Beams Number"
            ),
            gr.components.Checkbox(value=False, label="Do sample"),
            gr.components.Slider(
                minimum=1, maximum=2000, step=10, value=512, label="Max New Tokens"
            ),
            gr.components.Slider(
                minimum=1, maximum=300, step=10, value=1, label="Min New Tokens"
            ),
            gr.components.Slider(
                minimum=1.0,
                maximum=2.0,
                step=0.1,
                value=1.2,
                label="Repetition Penalty",
            ),
        ],
        outputs=[
            gr.components.Textbox(
                lines=25,
                label="Output",
            )
        ],
        title="BELLE: Be Everyone's Large Language model Engine",
    ).queue().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=args.base_port + torch.distributed.get_rank()
    )

if __name__ == "__main__":
    main()

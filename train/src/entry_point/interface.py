import argparse
from functools import partial
import os

import deepspeed
import gradio as gr
import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaTokenizer,
)

world_size = int(os.getenv("WORLD_SIZE", "1"))

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--use_deepspeed", action="store_true")
parser.add_argument("--llama", action="store_true")
parser.add_argument("--base_port", default=17860, type=int)
args = parser.parse_args()


def generate_prompt(input_text):
    return "Human: \n" + input_text + "\n\nAssistant:\n"


def evaluate(
    model,
    input: str,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    do_sample=False,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=1.2,
    **kwargs,
):
    prompt = generate_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs["input_ids"].to(getattr(model, 'module', model).device)

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
        **kwargs,
    )
    with torch.no_grad():
        # pudb.set_trace()
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False
        )
        output = generation_output.sequences[0]
        output = (
            tokenizer.decode(output, skip_special_tokens=True)
            .strip()
        )[len(input):]
        return output


if __name__ == "__main__":
    load_type = torch.float16  # Sometimes may need torch.float32

    if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    print(f"Rank {args.local_rank} loading model...")

    if args.use_lora:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=load_type, config=model_config
        )
        model = PeftModel.from_pretrained(
            base_model, args.ckpt_path, torch_dtype=load_type
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path, torch_dtype=load_type, config=model_config
        )

    if not args.use_deepspeed:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.local_rank}')
        else:
            device = torch.device('cpu')
        if device == torch.device('cpu'):
            model.float()
        print(f'device: {device}')
        model.to(device)
        model.eval()
    else:
        model = deepspeed.init_inference(
            model,
            mp_size=world_size,
            dtype=torch.half,
            checkpoint=None,
            replace_with_kernel_inject=True,
        )

    # model = None

    print("Load model successfully")
    # https://gradio.app/docs/
    gr.Interface(
        fn=partial(evaluate, model),
        inputs=[
            gr.components.Textbox(
                lines=2, label="Input", placeholder="Welcome to the BELLE model"
            ),
            gr.components.Slider(minimum=0, maximum=1,
                                 value=0.1, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1,
                                 value=0.75, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=1, label="Beams Number"
            ),
            gr.components.Checkbox(
                value=False,
                label="Do sample"
            ),
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
            )
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=25,
                label="Output",
            )
        ],
        title="BELLE: Be Everyone's Large Language model Engine",
    ).queue().launch(
        share=True, server_name="127.0.0.1", server_port=args.base_port + args.local_rank
    )

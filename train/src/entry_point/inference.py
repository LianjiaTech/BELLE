import torch
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import  PeftModel
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--lora_path', type=str, default=None)
parser.add_argument('--use_lora', action="store_true")
parser.add_argument('--llama', action="store_true")
parser.add_argument('--infer_file', type=str, required=True)
args = parser.parse_args()

max_new_tokens = 128
generation_config = dict(
    temperature=0.9,
    top_k=30,
    top_p=0.6,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=max_new_tokens
)

infer_data = pd.read_json(args.infer_file, lines=True)
instruction_list = infer_data.apply(
    lambda row: pd.Series(
        {'instruction': f"Human: \n" + row['text'] + "\n\nAssistant: \n"}
    ), axis=1
)['instruction'].to_list()

if __name__ == '__main__':
    load_type = torch.float16 #Sometimes may need torch.float32
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    model_config = AutoConfig.from_pretrained(args.ckpt_path)

    if args.use_lora:
        base_model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=load_type, device_map='auto')
        model = PeftModel.from_pretrained(base_model, args.lora_path, torch_dtype=load_type)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=load_type, config=model_config, device_map='auto')

    if device==torch.device('cpu'):
        model.float()

    model.eval()
    print("Load model successfully")

    for instruction in tqdm(instruction_list):
        inputs = tokenizer(
            instruction,
            add_special_tokens=False,
            return_tensors="pt"
        )
        generation_output = model.generate(
            input_ids = inputs["input_ids"].to(device), 
            **generation_config
        )[0]

        generate_text = tokenizer.decode(generation_output,skip_special_tokens=True)
        print(generate_text)
        print("-"*100)

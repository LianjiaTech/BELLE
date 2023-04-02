import sys
import os, json
from collections import namedtuple, OrderedDict
import argparse
from model import GEN4ALL
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch
from tqdm import tqdm

def load_dev_data(dev_file_path = 'belle_open_source_data/Belle_open_source_0.5M.dev.json'):
    dev_data = []
    with open(dev_file_path) as f:
        lines = f.readlines()
        for line in lines:
            dev_data.append(json.loads(line.strip()))
    print(dev_data[:10])
    return dev_data

def generate_text(dev_data, batch_size, tokenizer, model, skip_special_tokens = True, clean_up_tokenization_spaces=True):
    res = []
    for i in tqdm(range(0, len(dev_data), batch_size), total=len(dev_data), unit="batch"):
        batch = dev_data[i:i+batch_size]
        batch_text = [tokenizer.bos_token + item['instruction']+item['input'] if tokenizer.bos_token!=None else item['instruction']+item['input'] for item in batch]
        features = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=True, max_length = args.max_length)
        input_ids = features['input_ids'].to("cuda")
        attention_mask = features['attention_mask'].to("cuda")

        output_texts = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams = 4,
            do_sample = False,
            min_new_tokens=1,
            max_new_tokens=512,
            early_stopping= True 
        )
        output_texts = tokenizer.batch_decode(
            output_texts.cpu().numpy().tolist(),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        for i in range(len(output_texts)):
            input_text = batch_text[i]
            input_text = input_text.replace(tokenizer.bos_token, "")
            predict_text = output_texts[i][len(input_text):]
            res.append({"input":input_text,"predict":predict_text,"target":batch[i]["output"]})
    return res


def load_from_ckpt(args):
    model = GEN4ALL(args)
    print("Load pretrained model from {}".format(args.model_name_or_path))
    weight_dict = torch.load(args.ckpt_path, map_location="cpu")
    weight_dict.keys()

    copy_state_dict = OrderedDict()
    for key in weight_dict['state_dict'].keys():
        value = weight_dict['state_dict'][key]
        if "model.model." in key:
            key = key.replace("model.model.","model.")
            copy_state_dict[key] = value
    weight_dict['state_dict'] = copy_state_dict
    print(model.load_state_dict(weight_dict['state_dict']))#load model
    model.eval()
    model.to("cuda")
    return model

def main(args):
    dev_data = load_dev_data(args.dev_file)[:10]#For simplify and save time, we only evaluate ten samples
    res = generate_text(dev_data, batch_size, tokenizer, model)
    with open(args.output_file, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True, help="pretrained language model")
    parser.add_argument("--max_length", type=int, default=512, help="max length of dataset")
    parser.add_argument("--dev_batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--llama", action='store_true', help="use llama")
    parser.add_argument("--use_lora", action="store_true", help="use lora")
    parser.add_argument("--lora_hyperparams_file", type=str, default="")
    parser.add_argument("--output_file", default="./predictions.json", type=str)

    args = parser.parse_args()
    batch_size = args.dev_batch_size

    if not args.llama:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)

    model.eval()
    model.to("cuda")
    main(args)